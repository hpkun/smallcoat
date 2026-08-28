from __future__ import annotations

from train import build_training_env
from src.observation_builder import LINK_FEATURE_DIM
from src.observation_builder import MAX_NEIGHBOR_LINKS
from src.observation_builder import NODE_LOAD_DIM
from src.observation_builder import TASK_FEATURE_DIM


def test_training_env_only_builds_agents_for_cluster_heads_or_isolated_uavs() -> None:
    env = build_training_env()
    observations, action_specs = env.reset()

    decision_uav_ids = {
        uav.node_id
        for uav in env.base_env.uavs
        if uav.is_cluster_head or uav.is_isolated
    }

    assert set(observations) <= decision_uav_ids
    assert set(action_specs) == set(observations)
    assert all(env.pending_contexts[agent_id].task_slots for agent_id in observations)
    assert len(observations) < len(env.base_env.uavs)


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
