from __future__ import annotations

import numpy as np
import torch

from src.action_space import build_action_spec
from src.cmppo import CMPPOConfig
from src.cmppo import CMPPOSystem
from src.observation_builder import OBSERVATION_INPUT_DIM


def _action_spec():
    return build_action_spec(
        ["target-slot-0", "target-slot-1", "target-slot-2"],
        [[True, True, True], [True, False, True]],
        slot_target_node_ids=[
            ["uav-0", "bs-0", "leo-0"],
            ["uav-1", "", "leo-0"],
        ],
    )


def test_cmppo_samples_only_legal_targets() -> None:
    torch.manual_seed(42)
    spec = _action_spec()
    system = CMPPOSystem(
        state_dim=2 * OBSERVATION_INPUT_DIM,
        action_spec=spec,
        device="cpu",
        redundancy_mode="hybrid",
        config=CMPPOConfig(update_epochs=1, minibatch_size=4),
    )
    observations = {"uav-0": np.zeros(2 * OBSERVATION_INPUT_DIM, dtype=np.float32)}
    actions, step = system.sample_actions(observations, {"uav-0": spec})

    assert actions["uav-0"].slot_actions[0].target_node_id in {
        "uav-0",
        "bs-0",
        "leo-0",
    }
    assert actions["uav-0"].slot_actions[1].target_node_id in {"uav-1", "leo-0"}
    assert len(step.agent_samples) == 1


def test_cmppo_keeps_independent_actor_and_critic_per_agent() -> None:
    torch.manual_seed(11)
    spec = _action_spec()
    system = CMPPOSystem(
        state_dim=2 * OBSERVATION_INPUT_DIM,
        action_spec=spec,
        device="cpu",
        redundancy_mode="hybrid",
        config=CMPPOConfig(update_epochs=1, minibatch_size=4),
    )
    observations = {
        "uav-0": np.zeros(2 * OBSERVATION_INPUT_DIM, dtype=np.float32),
        "uav-1": np.ones(2 * OBSERVATION_INPUT_DIM, dtype=np.float32),
    }
    system.sample_actions(observations, {"uav-0": spec, "uav-1": spec})

    assert system.agents["uav-0"].actor is not system.agents["uav-1"].actor
    assert system.agents["uav-0"].critic is not system.agents["uav-1"].critic
    assert (
        next(system.agents["uav-0"].actor.parameters()).data_ptr()
        != next(system.agents["uav-1"].actor.parameters()).data_ptr()
    )


def test_cmppo_none_mode_disables_redundancy_and_updates() -> None:
    torch.manual_seed(7)
    spec = _action_spec()
    system = CMPPOSystem(
        state_dim=2 * OBSERVATION_INPUT_DIM,
        action_spec=spec,
        device="cpu",
        redundancy_mode="none",
        config=CMPPOConfig(update_epochs=1, minibatch_size=4),
    )
    observations = {"uav-0": np.ones(2 * OBSERVATION_INPUT_DIM, dtype=np.float32)}
    actions, step = system.sample_actions(observations, {"uav-0": spec})
    step.shared_reward = 1.0
    result = system.update([step])

    assert all(slot.redundancy_eta == 0.0 for slot in actions["uav-0"].slot_actions)
    assert all(
        slot.backup_target_node_id is None for slot in actions["uav-0"].slot_actions
    )
    assert result is not None
    assert np.isfinite(result.actor_loss)
    assert np.isfinite(result.critic_loss)


def test_cmppo_updates_variable_task_trajectory() -> None:
    two_task_spec = _action_spec()
    one_task_spec = build_action_spec(
        ["target-slot-0", "target-slot-1", "target-slot-2"],
        [[True, True, True]],
        slot_target_node_ids=[["uav-0", "bs-0", "leo-0"]],
    )
    system = CMPPOSystem(
        state_dim=2 * OBSERVATION_INPUT_DIM,
        action_spec=two_task_spec,
        device="cpu",
        config=CMPPOConfig(update_epochs=1, minibatch_size=2),
    )
    trajectory = []
    for count, spec in ((2, two_task_spec), (1, one_task_spec)):
        _, step = system.sample_actions(
            {"uav-0": np.ones(count * OBSERVATION_INPUT_DIM, dtype=np.float32)},
            {"uav-0": spec},
        )
        step.shared_reward = 1.0
        trajectory.append(step)
    result = system.update(trajectory)
    assert result is not None
    assert np.isfinite(result.actor_loss)
    assert np.isfinite(result.critic_loss)
