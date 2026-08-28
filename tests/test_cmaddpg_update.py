from __future__ import annotations

import numpy as np

from src.action_space import build_action_spec
from src.cmaddpg import CMADDPGSystem
from src.observation_builder import OBSERVATION_INPUT_DIM


def test_update_reuses_target_actions_and_refreshes_only_soft_updated_actor() -> None:
    system = CMADDPGSystem()
    agent_ids = ["agent-0", "agent-1", "agent-2"]
    action_spec = build_action_spec(agent_ids, [[True, True, True]])

    for agent_id in agent_ids:
        system.ensure_agent(agent_id, OBSERVATION_INPUT_DIM, action_spec)

    observations = {
        agent_id: np.full(OBSERVATION_INPUT_DIM, index, dtype=np.float32)
        for index, agent_id in enumerate(agent_ids)
    }
    next_observations = {
        agent_id: observation + 0.5
        for agent_id, observation in observations.items()
    }
    actions = {
        agent_id: np.zeros(action_spec.actor_output_dim, dtype=np.float32)
        for agent_id in agent_ids
    }
    for _ in range(2):
        system.store_transitions(
            observations,
            actions,
            shared_reward=1.0,
            next_observations=next_observations,
            done=False,
        )

    target_actor_calls = dict.fromkeys(agent_ids, 0)
    hooks = []
    for agent_id in agent_ids:
        def count_call(_module, _inputs, _output, *, tracked_agent_id=agent_id):
            target_actor_calls[tracked_agent_id] += 1

        hooks.append(system.agents[agent_id].target_actor.register_forward_hook(count_call))

    try:
        result = system.update(batch_size=2)
    finally:
        for hook in hooks:
            hook.remove()

    assert result is not None
    assert all(call_count > 0 for call_count in target_actor_calls.values())
    assert all(
        parameter.requires_grad
        for agent in system.agents.values()
        for parameter in agent.critic.parameters()
    )


def test_task_count_changes_without_rebuilding_agent_networks() -> None:
    system = CMADDPGSystem()
    specs = {
        count: build_action_spec(
            ["uav-0", "bs-0", "leo-0"],
            [[True, True, True] for _ in range(count)],
        )
        for count in (1, 3)
    }
    system.ensure_agent("agent-0", OBSERVATION_INPUT_DIM, specs[1])
    actor = system.agents["agent-0"].actor
    critic = system.agents["agent-0"].critic
    for count in (1, 3):
        observation = np.ones(count * OBSERVATION_INPUT_DIM, dtype=np.float32)
        system.ensure_agent("agent-0", observation.size, specs[count])
        raw = system.act({"agent-0": observation}, add_noise=False)
        assert raw["agent-0"].size == specs[count].actor_output_dim
    assert system.agents["agent-0"].actor is actor
    assert system.agents["agent-0"].critic is critic
