from __future__ import annotations

import torch

from src.action_space import build_action_spec
from src.cmaddpg import CMADDPGSystem
from src.maddpg_agent import AgentHyperParameters
from src.networks import ActorNetwork
from src.networks import VariableTaskActorNetwork
from src.observation_builder import LINK_FEATURE_DIM
from src.observation_builder import MAX_NEIGHBOR_LINKS
from src.observation_builder import NODE_LOAD_DIM
from src.observation_builder import OBSERVATION_INPUT_DIM
from src.observation_builder import TASK_FEATURE_DIM
from train import build_training_env


def test_attention_actor_preserves_action_output_shape() -> None:
    num_task_slots = 2
    batch_size = 4
    state_dim = num_task_slots * OBSERVATION_INPUT_DIM
    action_dim = 18
    actor = ActorNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        num_task_slots=num_task_slots,
        use_self_attention=True,
    )

    output = actor(torch.zeros(batch_size, state_dim))

    assert output.shape == (batch_size, action_dim)


def test_attention_actor_batches_slots_without_changing_slot_order() -> None:
    num_task_slots = 3
    actor = ActorNetwork(
        state_dim=num_task_slots * OBSERVATION_INPUT_DIM,
        action_dim=18,
        num_task_slots=num_task_slots,
        use_self_attention=True,
    )
    state = torch.randn(4, num_task_slots * OBSERVATION_INPUT_DIM)
    slot_states = state.view(4, num_task_slots, OBSERVATION_INPUT_DIM)

    expected = torch.cat(
        [
            actor._encode_slot_tokens(slot_states[:, slot_index, :])
            for slot_index in range(num_task_slots)
        ],
        dim=-1,
    )
    actual = actor._encode_with_attention(state)

    torch.testing.assert_close(actual, expected)


def test_plain_actor_can_still_be_used_for_ablation() -> None:
    num_task_slots = 2
    batch_size = 4
    state_dim = num_task_slots * OBSERVATION_INPUT_DIM
    action_dim = 18
    actor = ActorNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        num_task_slots=num_task_slots,
        use_self_attention=False,
    )

    output = actor(torch.zeros(batch_size, state_dim))

    assert output.shape == (batch_size, action_dim)


def test_cmaddpg_system_can_enable_attention_actor() -> None:
    system = CMADDPGSystem(
        agent_hyper_params=AgentHyperParameters(use_actor_self_attention=True),
    )
    action_spec = build_action_spec(
        ["uav-0", "bs-0"],
        [[True, True], [True, True]],
    )

    system.ensure_agent(
        agent_id="uav-0",
        state_dim=2 * OBSERVATION_INPUT_DIM,
        action_spec=action_spec,
    )

    assert system.agents["uav-0"].actor.use_self_attention


def test_resource_aware_actor_responds_to_candidate_resource_state() -> None:
    torch.manual_seed(7)
    actor = VariableTaskActorNetwork(
        per_task_state_dim=OBSERVATION_INPUT_DIM,
        per_task_action_dim=24,
        use_self_attention=True,
        use_resource_awareness=True,
    )
    state = torch.zeros(1, 1, OBSERVATION_INPUT_DIM)
    candidate_start = NODE_LOAD_DIM + TASK_FEATURE_DIM
    candidates = state[..., candidate_start:].view(
        1, 1, MAX_NEIGHBOR_LINKS, LINK_FEATURE_DIM
    )
    candidates[..., 0, :] = 0.1
    changed = state.clone()
    changed_candidates = changed[..., candidate_start:].view(
        1, 1, MAX_NEIGHBOR_LINKS, LINK_FEATURE_DIM
    )
    changed_candidates[..., 0, :] = 0.9

    original_action = actor(state)
    changed_action = actor(changed)

    assert not torch.allclose(original_action, changed_action)


def test_resource_awareness_observes_candidate_queue_pressure() -> None:
    env = build_training_env(
        enable_resource_awareness=True,
        arrival_rate_tasks_per_s=25.0,
        seed=42,
    )
    observations, _ = env.reset()
    agent_id = next(iter(observations))
    context = env.pending_contexts[agent_id]
    candidate_id = next(
        node_id
        for node_id in context.slot_target_node_ids[0]
        if node_id.startswith("bs-")
    )
    candidate = env.base_env.get_compute_node_by_id(candidate_id)
    before = observations[agent_id].copy()
    candidate.queue_manager.commit(
        task_id="synthetic-load",
        arrival_time_s=0.0,
        service_time_s=2.0,
        priority_eta=0.5,
        current_time_s=0.0,
    )
    after, _ = env._build_contexts_and_states()

    assert not torch.equal(
        torch.as_tensor(before),
        torch.as_tensor(after[agent_id]),
    )
