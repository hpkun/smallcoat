from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import HeuristicLatencyBaseline
from .baselines import RandomOffloadingBaseline
from .rl_env import CMADDPGEnv


@dataclass(frozen=True)
class EvaluationSummary:
    """评估结果摘要。"""

    avg_delay_s: float
    completion_rate: float
    system_profit: float
    shared_reward: float


def evaluate_baseline(
    env: CMADDPGEnv,
    policy_name: str = "heuristic",
    num_steps: int = 10,
) -> EvaluationSummary:
    """评估一个非学习基线。"""

    if policy_name == "random":
        policy = RandomOffloadingBaseline(env.base_env.rng)
    else:
        policy = HeuristicLatencyBaseline()

    states, action_specs = env.reset()
    shared_rewards: list[float] = []
    delay_values: list[float] = []
    completion_values: list[float] = []
    profit_values: list[float] = []

    for _ in range(num_steps):
        actions = {}
        for agent_id, state in states.items():
            action_spec = action_specs[agent_id]
            if policy_name == "random":
                actions[agent_id] = policy.act(action_spec)
            else:
                actions[agent_id] = policy.act(action_spec, state)

        states, _, _, info = env.step(actions)
        action_specs = info["action_specs"]
        metrics = env.extract_record_metrics(info["records"])
        shared_rewards.append(info["shared_reward"])
        delay_values.append(metrics["avg_delay_s"])
        completion_values.append(metrics["completion_rate"])
        profit_values.append(metrics["system_profit"])

    return EvaluationSummary(
        avg_delay_s=float(np.mean(delay_values) if delay_values else 0.0),
        completion_rate=float(np.mean(completion_values) if completion_values else 0.0),
        system_profit=float(np.mean(profit_values) if profit_values else 0.0),
        shared_reward=float(np.mean(shared_rewards) if shared_rewards else 0.0),
    )
