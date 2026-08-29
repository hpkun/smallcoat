from __future__ import annotations

import numpy as np

from train import build_training_env
from src.cmaddpg import CMADDPGSystem
from src.observation_builder import LINK_FEATURE_DIM
from src.observation_builder import MAX_NEIGHBOR_LINKS
from src.observation_builder import NODE_LOAD_DIM
from src.observation_builder import TASK_FEATURE_DIM


def test_training_env_only_builds_agents_for_cluster_heads_or_isolated_uavs() -> None:
    env = build_training_env()
    observations, action_specs = env.reset()

    logical_agent_ids = {
        env._logical_agent_id(uav)
        for uav in env.base_env.uavs
        if uav.is_cluster_head or uav.is_isolated
    }

    assert set(observations) <= logical_agent_ids
    assert set(action_specs) == set(observations)
    assert all(env.pending_contexts[agent_id].task_slots for agent_id in observations)
    assert all(
        env.pending_contexts[agent_id].decision_uav_id != agent_id
        for agent_id in observations
    )
    assert len(observations) < len(env.base_env.uavs)


def test_logical_ch_agent_id_is_stable_across_physical_head_candidates() -> None:
    env = build_training_env()
    cluster = next(
        cluster
        for cluster in env.base_env.clustering_manager.cluster_infos.values()
        if len(cluster.member_uav_ids) >= 2
    )
    first = env.base_env.get_uav_by_id(cluster.member_uav_ids[0])
    second = env.base_env.get_uav_by_id(cluster.member_uav_ids[1])

    assert first.node_id != second.node_id
    assert env._logical_agent_id(first) == env._logical_agent_id(second)
    assert env._logical_agent_id(first) == cluster.logical_agent_id


def test_failed_physical_head_is_replaced_consistently_for_rl_and_execution() -> None:
    env = build_training_env()
    manager = env.base_env.clustering_manager
    assert manager is not None
    cluster = next(
        cluster
        for cluster in manager.cluster_infos.values()
        if len(cluster.member_uav_ids) >= 2
    )
    original_head = env.base_env.get_uav_by_id(cluster.head_uav_id)
    original_head.remaining_energy_j = 0.0
    serviceable_members = [
        env.base_env.get_uav_by_id(uav_id)
        for uav_id in cluster.member_uav_ids
        if uav_id != original_head.node_id
    ]
    for uav in serviceable_members:
        uav.remaining_energy_j = uav.battery_capacity_j
    expected_head = min(
        serviceable_members,
        key=lambda uav: (
            uav.position.distance_to(cluster.centroid),
            uav.node_id,
        ),
    )

    bindings = {binding.agent_id: binding for binding in env._decision_agents()}
    updated_cluster = manager.cluster_infos[cluster.cluster_id]
    execution_head = env.base_env.get_decision_uav(original_head)

    assert updated_cluster.logical_agent_id == cluster.logical_agent_id
    assert updated_cluster.head_uav_id == expected_head.node_id
    assert bindings[cluster.logical_agent_id].decision_uav_id == expected_head.node_id
    assert execution_head.node_id == expected_head.node_id
    assert expected_head.is_cluster_head
    assert not original_head.is_cluster_head


def test_context_and_replay_use_logical_agent_ids() -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    for _ in range(20):
        observations, action_specs = env.reset()
        if observations:
            break
    else:
        raise AssertionError("Environment did not generate tasks for the test.")

    bindings = {binding.agent_id: binding for binding in env._decision_agents()}
    for agent_id, context in env.pending_contexts.items():
        assert agent_id in bindings
        assert context.agent_id == agent_id
        assert context.decision_uav_id == bindings[agent_id].decision_uav_id
        assert all(task.owner_agent_id == agent_id for task in context.task_slots)
        assert all(
            task.owner_ch_id == context.decision_uav_id for task in context.task_slots
        )

    system = CMADDPGSystem()
    for agent_id, observation in observations.items():
        system.ensure_agent(agent_id, observation.size, action_specs[agent_id])
    raw_actions = system.act(observations, add_noise=False)
    env_actions, critic_actions = system.decode_actions(raw_actions)
    next_observations, _, done, _ = env.step(env_actions)
    system.store_transitions(
        observations,
        critic_actions,
        shared_reward=0.0,
        next_observations=next_observations,
        done=done,
    )

    transition = system.replay_buffer.buffer[-1]
    assert all(not agent_id.startswith("uav-") for agent_id in transition.agent_ids)
    assert set(transition.local_states) == set(observations)


def test_reclustering_keeps_existing_actor_objects() -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    for _ in range(20):
        observations, action_specs = env.reset()
        if observations:
            break
    else:
        raise AssertionError("Environment did not generate tasks for the test.")

    system = CMADDPGSystem()
    for agent_id, observation in observations.items():
        system.ensure_agent(agent_id, observation.size, action_specs[agent_id])
    actors_before = dict(system.actors)
    logical_ids_before = {
        cluster.logical_agent_id
        for cluster in env.base_env.clustering_manager.cluster_infos.values()
    }

    env.base_env.clustering_manager.centralized_clustering(
        env.base_env.uavs, env.base_env.rng
    )
    next_observations, next_action_specs = env._build_contexts_and_states()
    for agent_id, observation in next_observations.items():
        system.ensure_agent(agent_id, observation.size, next_action_specs[agent_id])

    logical_ids_after = {
        cluster.logical_agent_id
        for cluster in env.base_env.clustering_manager.cluster_infos.values()
    }
    assert logical_ids_after == logical_ids_before
    assert set(system.actors) <= logical_ids_after
    assert all(
        system.actors[agent_id] is actor
        for agent_id, actor in actors_before.items()
    )
    assert all(not agent_id.startswith("uav-") for agent_id in system.actors)


def test_workflow_records_logical_owner_and_physical_ch_separately() -> None:
    env = build_training_env(task_mode="workflow")
    ingress_uav = env.base_env.uavs[0]
    workflow = env.workflow_generator._generate_one_workflow(
        uavs=[ingress_uav],
        current_time_s=0.0,
        rng=np.random.default_rng(9),
        delay_sensitivity_lambda=None,
    )

    env._assign_owner_ch_to_workflows([workflow])

    expected_agent_id = env.base_env.clustering_manager.get_logical_agent_id(
        ingress_uav.node_id
    )
    assert expected_agent_id is not None
    assert workflow.owner_agent_id == expected_agent_id
    assert workflow.owner_ch_id is not None
    assert workflow.owner_ch_id.startswith("uav-")
    assert all(
        spec.task_instance.owner_agent_id == expected_agent_id
        and spec.task_instance.owner_ch_id == workflow.owner_ch_id
        for spec in workflow.task_specs.values()
    )


def test_reset_clears_node_queues_between_episodes() -> None:
    env = build_training_env()
    observations, _ = env.reset()
    env.base_env.uavs[0].queue_manager.commit(
        task_id="stale-task",
        arrival_time_s=0.0,
        service_time_s=10.0,
        priority_eta=0.5,
        current_time_s=0.0,
    )

    assert env.base_env.uavs[0].queue_manager.pending_entries

    observations, _ = env.reset()

    assert observations
    assert not env.base_env.uavs[0].queue_manager.pending_entries
    assert env.base_env.uavs[0].queue_manager.busy_until_s == 0.0


def test_actor_link_features_align_with_fixed_target_logits() -> None:
    env = build_training_env()
    observations, action_specs = env.reset()
    agent_id = next(iter(observations))
    state = observations[agent_id]
    spec = action_specs[agent_id]

    assert spec.num_discrete_targets == 11
    assert spec.num_discrete_targets <= MAX_NEIGHBOR_LINKS
    per_slot_dim = NODE_LOAD_DIM + TASK_FEATURE_DIM + LINK_FEATURE_DIM * MAX_NEIGHBOR_LINKS
    assert state.size == spec.num_task_slots * per_slot_dim

    context = env.pending_contexts[agent_id]
    for node_ids, mask in zip(context.slot_target_node_ids, spec.slot_target_masks):
        assert len(node_ids) == 11
        assert len(mask) == 11
        assert all(bool(node_id) == allowed for node_id, allowed in zip(node_ids, mask))
