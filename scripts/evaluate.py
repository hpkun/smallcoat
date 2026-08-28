from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_baseline
from train import build_training_env


def main() -> None:
    """评估入口。"""

    env = build_training_env()
    heuristic_result = evaluate_baseline(env, policy_name="heuristic", num_steps=5)
    random_result = evaluate_baseline(env, policy_name="random", num_steps=5)

    print(
        "heuristic",
        f"delay={heuristic_result.avg_delay_s:.6f}",
        f"completion={heuristic_result.completion_rate:.4f}",
        f"profit={heuristic_result.system_profit:.2f}",
        f"reward={heuristic_result.shared_reward:.4f}",
    )
    print(
        "random",
        f"delay={random_result.avg_delay_s:.6f}",
        f"completion={random_result.completion_rate:.4f}",
        f"profit={random_result.system_profit:.2f}",
        f"reward={random_result.shared_reward:.4f}",
    )


if __name__ == "__main__":
    main()
