from __future__ import annotations

from src.cmaddpg import CMADDPGSystem
from src.observation_builder import OBSERVATION_INPUT_DIM
from train import build_training_env


def _reset_with_tasks(env):
    for _ in range(20):
        observations, action_specs = env.reset()
        if env.pending_tasks:
            return observations, action_specs
    raise AssertionError("Environment did not generate tasks for the test.")


def test_arrival_rate_is_sampled_once_for_the_whole_system(monkeypatch) -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    calls = []

    def fake_generate_tasks(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        env.base_env.task_generator, "generate_tasks", fake_generate_tasks
    )
    assert env._generate_next_tasks() == []
    assert len(calls) == 1
    assert calls[0]["uavs"] == env.base_env.uavs
    assert env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s == 25.0


def test_every_pending_task_has_exactly_one_actor_task_row() -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)

    assert sum(spec.num_task_slots for spec in action_specs.values()) == len(
        env.pending_tasks
    )
    assert sum(
        observation.size // OBSERVATION_INPUT_DIM
        for observation in observations.values()
    ) == len(env.pending_tasks)


def test_actor_actions_bypass_environment_expert(monkeypatch) -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)
    system = CMADDPGSystem()
    for agent_id, observation in observations.items():
        system.ensure_agent(agent_id, observation.size, action_specs[agent_id])
    raw_actions = system.act(observations, add_noise=False)
    env_actions, _ = system.decode_actions(raw_actions)
    current_task_count = len(env.pending_tasks)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Environment expert must not schedule RL tasks.")

    monkeypatch.setattr(env.base_env, "select_best_plan", fail_if_called)
    _, _, _, info = env.step(env_actions)

    assert len(info["records"]) == current_task_count
    assert info["uncollected_task_count"] == 0


def test_missing_actor_action_stays_pending_without_expert(monkeypatch) -> None:
    env = build_training_env(arrival_rate_tasks_per_s=25.0)
    _reset_with_tasks(env)
    original_task_ids = {task.task_id for task in env.pending_tasks}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Environment expert must not schedule RL tasks.")

    monkeypatch.setattr(env.base_env, "select_best_plan", fail_if_called)
    _, _, _, info = env.step({})

    assert info["records"] == []
    assert original_task_ids <= {task.task_id for task in env.pending_tasks}
    assert info["uncollected_task_count"] == len(original_task_ids)
