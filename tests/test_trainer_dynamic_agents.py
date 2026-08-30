from __future__ import annotations

import numpy as np
import pytest

from train import build_training_env
from src.cmaddpg import CMADDPGSystem
from src.trainer import CMADDPGTrainer
from src.trainer import TrainerConfig


def test_unexpected_next_agent_is_rejected_before_store_and_update(monkeypatch) -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    for _ in range(20):
        observations, action_specs = env.reset()
        if observations:
            break
    else:
        raise AssertionError("Environment did not generate tasks for the test.")

    source_agent_id = next(iter(observations))
    source_observation = observations[source_agent_id]
    source_action_spec = action_specs[source_agent_id]
    monkeypatch.setattr(env, "reset", lambda: (observations, action_specs))

    new_agent_id = "ch-agent-new"
    original_step = env.step

    def step_with_new_agent(env_actions):
        next_observations, rewards, done, info = original_step(env_actions)
        next_observations = dict(next_observations)
        next_observations[new_agent_id] = np.zeros_like(source_observation)
        next_action_specs = dict(info["action_specs"])
        next_action_specs[new_agent_id] = source_action_spec
        info = dict(info)
        info["action_specs"] = next_action_specs
        return next_observations, rewards, done, info

    monkeypatch.setattr(env, "step", step_with_new_agent)

    system = CMADDPGSystem()
    call_order: list[str] = []

    def checked_store(*args, **kwargs):
        call_order.append("store")

    def checked_update(*args, **kwargs):
        assert new_agent_id in system.actors
        call_order.append("update")
        return None

    monkeypatch.setattr(system, "store_transitions", checked_store)
    monkeypatch.setattr(system, "update", checked_update)

    trainer = CMADDPGTrainer(
        env=env,
        system=system,
        config=TrainerConfig(
            num_episodes=1,
            steps_per_episode=1,
            update_every=1,
            batch_size=1,
            progress_print_interval=0,
        ),
    )
    with pytest.raises(RuntimeError, match="unexpected Actor"):
        trainer.train()

    assert new_agent_id not in system.actors
    assert call_order == []
