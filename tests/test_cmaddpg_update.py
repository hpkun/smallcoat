from __future__ import annotations

import numpy as np
import pytest
import torch

from src.action_space import build_action_spec
from src.cmaddpg import CMADDPGSystem
from src.observation_builder import OBSERVATION_INPUT_DIM


def test_update_builds_targets_once_and_steps_one_global_critic(monkeypatch) -> None:
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

    assert system.global_critic_optimizer is not None
    optimizer_step = system.global_critic_optimizer.step
    critic_step_calls = 0

    def count_critic_step(*args, **kwargs):
        nonlocal critic_step_calls
        critic_step_calls += 1
        return optimizer_step(*args, **kwargs)

    monkeypatch.setattr(system.global_critic_optimizer, "step", count_critic_step)

    try:
        result = system.update(batch_size=2)
    finally:
        for hook in hooks:
            hook.remove()

    assert result is not None
    assert target_actor_calls == dict.fromkeys(agent_ids, 2)
    assert critic_step_calls == 1
    assert system.global_critic is not None
    assert all(
        parameter.requires_grad
        for parameter in system.global_critic.parameters()
    )
    assert all(not hasattr(actor, "critic") for actor in system.actors.values())


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
    critic = system.global_critic
    for count in (1, 3):
        observation = np.ones(count * OBSERVATION_INPUT_DIM, dtype=np.float32)
        system.ensure_agent("agent-0", observation.size, specs[count])
        raw = system.act({"agent-0": observation}, add_noise=False)
        assert raw["agent-0"].size == specs[count].actor_output_dim
    assert system.agents["agent-0"].actor is actor
    assert system.global_critic is critic


def test_checkpoint_stores_actors_and_one_global_critic(tmp_path) -> None:
    system = CMADDPGSystem()
    agent_ids = ["ch-agent-0", "ch-agent-1"]
    action_spec = build_action_spec(agent_ids, [[True, True]])
    for agent_id in agent_ids:
        system.ensure_agent(agent_id, OBSERVATION_INPUT_DIM, action_spec)

    checkpoint_path = system.save(tmp_path / "global-critic.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["architecture"] == "multi_actor_global_critic_v2"
    assert checkpoint["actor_count"] == 2
    assert set(checkpoint["actors"]) == set(agent_ids)
    assert "global_critic" in checkpoint
    assert all("critic_state_dict" not in state for state in checkpoint["actors"].values())

    restored = CMADDPGSystem()
    for agent_id in agent_ids:
        restored.ensure_agent(agent_id, OBSERVATION_INPUT_DIM, action_spec)
    restored.load(checkpoint_path, strict=True)
    assert restored.global_critic is not None
    assert system.global_critic is not None
    for expected, actual in zip(
        system.global_critic.parameters(), restored.global_critic.parameters()
    ):
        torch.testing.assert_close(actual, expected)


def test_fixed_actor_pool_rejects_unexpected_identity() -> None:
    system = CMADDPGSystem()
    system.configure_agent_pool(["ch-agent-0", "ch-agent-1"])
    action_spec = build_action_spec(["target-0"], [[True]])
    system.ensure_agent("ch-agent-0", OBSERVATION_INPUT_DIM, action_spec)
    system.ensure_agent("ch-agent-1", OBSERVATION_INPUT_DIM, action_spec)

    with pytest.raises(RuntimeError, match="unexpected Actor"):
        system.ensure_agent("ch-agent-2", OBSERVATION_INPUT_DIM, action_spec)

    assert system.total_actor_count == 2
