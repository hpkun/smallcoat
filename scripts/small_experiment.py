from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import CMADDPGSystem
from src import CMADDPGTrainer
from src import TrainerConfig
from src.evaluation import evaluate_baseline
from train import build_small_scale_env


def evaluate_trained_policy(env, system: CMADDPGSystem, num_steps: int = 5) -> dict[str, float]:
    """评估短训练后的策略表现。"""

    states, action_specs = env.reset()
    shared_rewards: list[float] = []
    delay_values: list[float] = []
    completion_values: list[float] = []
    profit_values: list[float] = []

    for _ in range(num_steps):
        for agent_id, observation in states.items():
            system.ensure_agent(
                agent_id=agent_id,
                state_dim=int(observation.shape[0]),
                action_spec=action_specs[agent_id],
            )

        raw_actions = {}
        for agent_id, observation in states.items():
            raw_actions[agent_id] = system.agents[agent_id].act(observation, add_noise=False)

        env_actions, _ = system.decode_actions(raw_actions)
        states, _, _, info = env.step(env_actions)
        action_specs = info["action_specs"]
        metrics = env.extract_record_metrics(info["records"])
        shared_rewards.append(info["shared_reward"])
        delay_values.append(metrics["avg_delay_s"])
        completion_values.append(metrics["completion_rate"])
        profit_values.append(metrics["system_profit"])

    return {
        "avg_delay_s": float(np.mean(delay_values) if delay_values else 0.0),
        "completion_rate": float(np.mean(completion_values) if completion_values else 0.0),
        "system_profit": float(np.mean(profit_values) if profit_values else 0.0),
        "shared_reward": float(np.mean(shared_rewards) if shared_rewards else 0.0),
    }


def main() -> None:
    """运行一个小规模实验。"""

    heuristic_env = build_small_scale_env()
    random_env = build_small_scale_env()
    training_env = build_small_scale_env()
    trained_eval_env = build_small_scale_env()

    heuristic = evaluate_baseline(heuristic_env, policy_name="heuristic", num_steps=5)
    random_result = evaluate_baseline(random_env, policy_name="random", num_steps=5)

    system = CMADDPGSystem(device="cpu")
    trainer = CMADDPGTrainer(
        env=training_env,
        system=system,
        config=TrainerConfig(
            num_episodes=200,
            steps_per_episode=50,
            update_every=1,
            batch_size=8,
        ),
    )
    logger = trainer.train()
    trained_result = evaluate_trained_policy(trained_eval_env, system, num_steps=5)

    summary = {
        "heuristic": {
            "avg_delay_s": heuristic.avg_delay_s,
            "completion_rate": heuristic.completion_rate,
            "system_profit": heuristic.system_profit,
            "shared_reward": heuristic.shared_reward,
        },
        "random": {
            "avg_delay_s": random_result.avg_delay_s,
            "completion_rate": random_result.completion_rate,
            "system_profit": random_result.system_profit,
            "shared_reward": random_result.shared_reward,
        },
        "trained_short_run": trained_result,
        "training_log_count": len(logger.records),
    }

    output_dir = PROJECT_ROOT / "outputs" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "small_experiment.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("small_experiment_summary")
    for name, result in summary.items():
        if isinstance(result, dict):
            print(
                name,
                f"delay={result['avg_delay_s']:.6f}",
                f"completion={result['completion_rate']:.4f}",
                f"profit={result['system_profit']:.2f}",
                f"reward={result['shared_reward']:.4f}",
            )
        else:
            print(name, result)

    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
