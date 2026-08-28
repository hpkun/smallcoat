from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_episode_rewards(metrics_path: Path, *, average_per_step: bool) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [record for record in records if "episode_shared_reward" in record]
    if not episode_records:
        raise ValueError(f"No episode-level reward records found in {metrics_path}")

    episodes = np.asarray([int(record["episode"]) + 1 for record in episode_records], dtype=int)
    rewards = np.asarray([record["episode_shared_reward"] for record in episode_records], dtype=float)
    if average_per_step:
        episode_steps: dict[int, set[int]] = {}
        for record in records:
            if "episode" in record and "step" in record:
                episode_steps.setdefault(int(record["episode"]), set()).add(int(record["step"]))
        denominators = np.asarray(
            [max(len(episode_steps.get(int(ep) - 1, set())), 1) for ep in episodes],
            dtype=float,
        )
        rewards = rewards / denominators
    return episodes, rewards


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def moving_average(
    episodes: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if values.size < window:
        raise ValueError("window cannot exceed the number of episodes")
    if window == 1:
        return episodes, values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return episodes[window - 1 :], np.convolve(values, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig.4-style reward curve from training metrics.")
    parser.add_argument("metrics_path", help="Path to the training metrics JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output image. Defaults to outputs/figures/<metrics>_fig4_reward.png.",
    )
    parser.add_argument(
        "--average-per-step",
        action="store_true",
        help="Plot per-episode average reward instead of summed episode reward.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Min-max normalize rewards to [0, 1], matching the visual scale of the paper figure.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Moving-average window in episodes; defaults to 20.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs") / "figures" / f"{metrics_path.stem}_fig4_reward.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes, rewards = load_episode_rewards(metrics_path, average_per_step=args.average_per_step)
    if args.normalize:
        rewards = minmax_normalize(rewards)
    smooth_episodes, smooth_rewards = moving_average(episodes, rewards, args.window)

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(
        episodes,
        rewards,
        color="#9ebbd0",
        linewidth=0.8,
        alpha=0.35,
        label="Episode Reward",
    )
    plt.plot(
        smooth_episodes,
        smooth_rewards,
        color="#1479b8",
        linewidth=2.0,
        label=f"{args.window}-Episode Moving Average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Step Reward" if args.average_per_step else "Episode Reward")
    plt.xlim(1, int(episodes[-1]))
    plt.grid(True, linestyle="-", alpha=0.55)
    plt.legend(loc="upper left", frameon=True)
    if args.normalize:
        plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(output_path)


if __name__ == "__main__":
    main()
