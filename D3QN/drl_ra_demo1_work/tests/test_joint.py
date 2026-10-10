from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from drl_ra.agent import JointD3QNAgent
from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv, Task
from drl_ra.experiment import evaluate_joint_agent, train_joint_agent


def config():
    cfg = deepcopy(load_config())
    cfg["environment"].update(episode_steps=4, sample_reliability_failures=False)
    cfg["training"].update(episodes=2, batch_size=2, hidden_sizes=[16], target_update_steps=2)
    return cfg


def prepared_env():
    env = SAGINEnv(config(), seed=2)
    env.current_task = Task("safety", 800_000, 500_000_000, 3.0, 0.99, 0)
    env.edge_positions[0] = env.device_positions[0]
    env._observation()
    return env


def test_resource_changes_physics_and_reservation():
    env = prepared_env()
    candidate = env.candidates[1]
    low = env._candidate_with_resource(candidate, 0.25)
    high = env._candidate_with_resource(candidate, 1.0)
    assert low.delay_s > high.delay_s
    assert low.energy_mj > high.energy_mj
    assert low.reliability < high.reliability
    assert low.compute_capacity == pytest.approx(0.25 * env.available_capacity[1])
    env._reserve([low])
    assert env.reserved_capacity[1] == pytest.approx(low.compute_capacity)
    env._release([low])
    assert env.reserved_capacity.sum() == 0
    local_low = env._candidate_with_resource(env.candidates[0], 0.25)
    local_high = env._candidate_with_resource(env.candidates[0], 1.0)
    assert local_low.energy_mj < local_high.energy_mj
    assert local_low.delay_s > local_high.delay_s


def test_joint_context_preserves_primary_identity():
    env = prepared_env()
    state = env.joint_observation([0, 1], 2)
    assert state.shape == (env.joint_state_dim,)
    assert np.isfinite(state).all()
    assert not np.array_equal(state, env.joint_observation([1, 0], 2))
    assert np.array_equal(state[:4 + 7 * env.action_dim], env.joint_observation()[:4 + 7 * env.action_dim])


def test_resource_mask_has_no_fake_feasible_level():
    env = prepared_env()
    env.used_capacity[1] = env.total_capacity[1] - env.candidates[1].required_capacity * 1.2
    mask = env.joint_resource_mask([1], (0.25, 0.5, 0.75, 1.0))
    np.testing.assert_array_equal(mask, [False, False, False, True])
    with pytest.raises(ValueError, match="under-provisions"):
        env.step_joint([1], 0.25)
    env.used_capacity[1] = env.total_capacity[1]
    assert not env.joint_resource_mask([1], (0.25, 0.5, 0.75, 1.0)).any()
    # Local execution always provides a fallback, including an infeasible deadline.
    env.current_task = Task("safety", 1e6, 1e12, 1.0, 0.99, 0)
    env._observation()
    assert env.joint_resource_mask([0], (1.0,)).any()


def test_joint_reward_uses_gain_and_resource_cost():
    env = prepared_env()
    env.reward_cfg.update(latency=0.0, energy=0.0, reliability=2.0, violation=3.0, resource=0.4)
    _, reward, _, _, info = env.step_joint([0, 1], 0.5)
    expected = (2 * (info["reliability"] - info["primary_reliability"])
                - 3 * max(0, info["required_reliability"] - info["reliability"])
                - 0.4 * 0.5 * 2 / env.env_cfg["max_replicas"])
    assert reward == pytest.approx(expected)
    assert info["reserved_after_event"] == 0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_autoregressive_replay_uses_successor_head_and_masks(count):
    env = prepared_env()
    agent = JointD3QNAgent(env, config(), seed=2)
    # Force all branches, including the zero-additional-replica branch.
    with patch.object(agent, "_act", side_effect=lambda state, mask, head, eps:
                      count - 1 if head == "replica_count" else int(np.flatnonzero(mask)[0])):
        _, selected, ratio, transitions = agent.decide(env)
    assert len(selected) == count and len(set(selected)) == count
    _, reward, done, _, info = env.step_joint(selected, ratio)
    next_state = env.joint_observation()
    next_mask = agent._node_mask(env)
    agent.observe_sequence(transitions, reward, next_state, done, next_mask, cost=info["cost"])
    for i, transition in enumerate(transitions):
        head = transition[0]
        last = i == len(transitions) - 1
        successor_head = "primary" if last else transitions[i + 1][0]
        row = agent.buffers[head, successor_head]._data[-1]
        np.testing.assert_array_equal(row[3], next_state if last else transitions[i + 1][1])
        np.testing.assert_array_equal(row[5], next_mask if last else transitions[i + 1][4])
        assert row[6] == (agent.gamma if last else 1.0)
        assert row[2] == pytest.approx(reward - agent.lagrange * info["cost"] if last else 0.0)


def test_training_evaluation_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    cfg = config()
    agent, history = train_joint_agent(cfg, 5, progress=False, metrics_only=True)
    assert len(history) == 2 and agent.training_steps == 8
    assert all(np.isfinite(row["loss"]) for row in history)
    assert history[-1]["tasks"] == 4
    probe = SAGINEnv(cfg, seed=5)
    expected = agent.decide(probe)[:3]
    checkpoint = tmp_path / "joint.pt"
    agent.save(checkpoint, {"method": "joint-d3qn"})
    restored = JointD3QNAgent(probe, cfg, 123)
    assert restored.load(checkpoint)["method"] == "joint-d3qn"
    assert restored.decide(probe)[:3] == expected
    assert restored.lagrange == agent.lagrange
    assert restored.training_steps == agent.training_steps
    rows, aggregate = evaluate_joint_agent(cfg, restored, [5, 6])
    assert len(rows) == 2 and aggregate["tasks"]["mean"] == 4
    assert all(np.isfinite(value) for row in rows for value in row.values())
    wrong = deepcopy(cfg)
    wrong["joint"]["resource_levels"] = [1.0, 0.75, 0.5, 0.25]
    with pytest.raises(ValueError, match="semantics"):
        JointD3QNAgent(probe, wrong, 1).load(checkpoint)


def test_double_q_selects_online_action_in_successor_head():
    env = prepared_env()
    agent = JointD3QNAgent(env, config(), 1)
    state = env.joint_observation()
    mask = np.array([True, True, False])
    pair = ("primary", "replica_count")
    for _ in range(2):
        agent.buffers[pair].add(state, 0, 0.0, state, False, mask, 1.0)
    with torch.no_grad():
        for net in (agent.online, agent.target):
            for p in net.parameters():
                p.zero_()
        agent.online.advantages["replica_count"].bias.copy_(torch.tensor([1., 2., 100.]))
        agent.target.advantages["replica_count"].bias.copy_(torch.tensor([9., 3., 6.]))
    # Online picks action 1; target evaluates it as 3 - mean(9,3,6) = -3.
    # Huber(0, -3) = 2.5, whereas target-max would give a different result.
    assert agent._learn_head(pair) == pytest.approx(2.5)


def test_local_only_mask_and_terminal_reward():
    env = prepared_env()
    env._last_candidates = [env.candidates[0]] + [replace(item, available=False) for item in env.candidates[1:]]
    agent = JointD3QNAgent(env, config(), 1)
    np.testing.assert_array_equal(agent._count_mask(env, 0), [True, False, False])
    _, selected, _, transitions = agent.decide(env, epsilon=1.0)
    assert selected == [0]
    agent.observe_sequence(transitions, 2.0, env.joint_observation(), True, agent._node_mask(env), cost=0.5)
    row = agent.buffers["resource", "primary"]._data[-1]
    assert row[4] is True
    assert row[2] == pytest.approx(1.5)


def test_resource_baseline_retains_analytic_replica_plan():
    env = prepared_env()
    cfg = config()
    agent = JointD3QNAgent(env, cfg, 1, analytic_replicas=True)
    with patch.object(agent, "_act", side_effect=lambda state, mask, head, eps: int(np.flatnonzero(mask)[0])):
        primary, selected, ratio, transitions = agent.decide(env)
    assert selected == [item.action for item in env._replica_plan(env.candidates[primary])]
    assert [row[0] for row in transitions] == ["primary", "resource"]
    _, _, _, _, info = env.step_joint(selected, ratio)
    assert info["resource_ratio"] == ratio
    trained, history = train_joint_agent(cfg, 3, progress=False, metrics_only=True, analytic_replicas=True)
    assert trained.analytic_replicas and np.isfinite(history[-1]["loss"])
