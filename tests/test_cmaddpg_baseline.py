from __future__ import annotations

import numpy as np
import torch

from src.action_space import build_action_spec
from src.baseline_action_space import BaselineActionCodec
from src.baseline_cmaddpg import BASELINE_ARCHITECTURE
from src.baseline_cmaddpg import BaselineCMADDPGSystem
from src.observation_builder import CANDIDATE_FEATURE_DIM
from src.observation_builder import MAX_NEIGHBOR_LINKS
from src.observation_builder import NODE_LOAD_DIM
from src.observation_builder import OBSERVATION_INPUT_DIM
from src.observation_builder import TASK_FEATURE_DIM
from train import build_training_env
from train_baseline import build_baseline_env
from train_baseline import build_baseline_reward_config


def _reset_with_tasks(env):
    for _ in range(30):
        observations, action_specs = env.reset()
        if observations:
            return observations, action_specs
    raise AssertionError("Environment did not generate baseline tasks.")


def test_baseline_observation_hides_proposed_reliability_and_energy_state() -> None:
    env = build_baseline_env(seed=42, arrival_rate_tasks_per_s=25.0)
    observations, _ = _reset_with_tasks(env)
    blocks = np.concatenate(
        [value.reshape(-1, OBSERVATION_INPUT_DIM) for value in observations.values()]
    )
    node = blocks[:, :NODE_LOAD_DIM]
    task = blocks[:, NODE_LOAD_DIM : NODE_LOAD_DIM + TASK_FEATURE_DIM]
    candidates = blocks[:, NODE_LOAD_DIM + TASK_FEATURE_DIM :].reshape(
        -1,
        MAX_NEIGHBOR_LINKS,
        CANDIDATE_FEATURE_DIM,
    )

    assert env.observation_builder.observation_profile == "baseline"
    assert np.all(node[:, 4:6] == 0.0)
    assert np.all(task[:, 4] == 0.0)
    assert np.all(candidates[:, :, 3:] == 0.0)


def test_proposed_observation_and_action_contract_remain_unchanged() -> None:
    env = build_training_env(seed=42, arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)
    blocks = np.concatenate(
        [value.reshape(-1, OBSERVATION_INPUT_DIM) for value in observations.values()]
    )
    spec = next(iter(action_specs.values()))

    assert env.observation_builder.observation_profile == "proposed"
    assert np.any(blocks[:, 4] > 0.0)
    assert np.any(blocks[:, 5] > 0.0)
    assert np.any(blocks[:, NODE_LOAD_DIM + 4] > 0.0)
    assert spec.per_slot_output_dim == 3 + 3 * spec.num_discrete_targets + 1


def test_baseline_action_is_exactly_one_target_plus_priority() -> None:
    env = build_baseline_env(seed=42, arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)
    spec = next(iter(action_specs.values()))
    target_count = spec.num_discrete_targets
    assert spec.per_slot_output_dim == target_count + 1

    raw = np.zeros(spec.actor_output_dim, dtype=np.float32)
    decoded = BaselineActionCodec(spec).decode_numpy(raw)
    env_action = decoded.to_multi_task_action()
    critic_action = BaselineActionCodec(spec).encode_for_critic(
        decoded.slot_target_indices,
        decoded.slot_priority_etas,
    )

    assert all(action.replica_count == 1 for action in env_action.slot_actions)
    assert all(
        len(action.replica_target_node_ids) == 1
        for action in env_action.slot_actions
    )
    assert critic_action.size == spec.num_task_slots * (target_count + 1)
    assert observations


def test_baseline_reward_is_profit_only_with_energy_dual_disabled() -> None:
    config = build_baseline_reward_config()

    assert config.deadline_failure_penalty == 0.0
    assert config.capacity_drop_penalty == 0.0
    assert config.reliability_failure_penalty == 0.0
    assert config.completion_delay_penalty == 0.0
    assert config.energy_penalty_weight == 0.0
    assert config.completion_constraint_dual_lr == 0.0
    assert config.long_term_energy_budget_j_per_step is None
    assert config.energy_constraint_dual_lr == 0.0
    assert config.advantage_reward_weight == 0.0


def test_baseline_keeps_physical_failures_battery_and_energy_accounting() -> None:
    env = build_baseline_env(seed=42, arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)
    system = BaselineCMADDPGSystem()
    for agent_id, observation in observations.items():
        system.ensure_agent(agent_id, observation.size, action_specs[agent_id])
    env_actions, _ = system.decode_actions(system.act(observations, add_noise=False))
    _, _, _, info = env.step(env_actions)

    assert not env.base_env.enable_redundancy
    assert any(uav.execution_failure_rate > 0.0 for uav in env.base_env.uavs)
    assert all(uav.battery_capacity_j > 0.0 for uav in env.base_env.uavs)
    assert all(record.requested_replica_count == 1 for record in info["records"])
    assert all(record.total_energy_j >= 0.0 for record in info["records"])
    expected_reward = info["equation8_objective"].total_profit / 1_000_000_000.0
    assert np.isclose(info["shared_reward"], expected_reward)
    assert info["energy_constraint_multiplier"] == 0.0
    assert info["energy_budget_violation_j"] == 0.0


def test_baseline_system_updates_and_checkpoint_is_distinct(tmp_path) -> None:
    env = build_baseline_env(seed=42, arrival_rate_tasks_per_s=25.0)
    observations, action_specs = _reset_with_tasks(env)
    system = BaselineCMADDPGSystem()
    for agent_id, observation in observations.items():
        system.ensure_agent(agent_id, observation.size, action_specs[agent_id])
    raw_actions = system.act(observations, add_noise=False)
    _, critic_actions = system.decode_actions(raw_actions)
    for _ in range(2):
        system.store_transitions(
            observations,
            critic_actions,
            shared_reward=1.0,
            next_observations=observations,
            done=False,
        )
    assert system.update(batch_size=2) is not None

    path = system.save(tmp_path / "baseline.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["algorithm"] == "cmaddpg-baseline"
    assert checkpoint["architecture"] == BASELINE_ARCHITECTURE
    assert checkpoint["observation_profile"] == "baseline"
    assert checkpoint["action_profile"] == "single-copy"
    assert checkpoint["reward_profile"] == "profit-only"

    proposed = build_action_spec(["target-0"], [[True]])
    assert proposed.per_slot_output_dim == 7
