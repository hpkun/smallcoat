from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_episode_completion(
    metrics_path: Path,
    *,
    as_percentage: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [
        record for record in records if "episode_task_completion_rate" in record
    ]
    if not episode_records:
        raise ValueError(f"No episode completion records found in {metrics_path}")
    episode_records.sort(key=lambda record: int(record["episode"]))
    episodes = np.asarray(
        [int(record["episode"]) + 1 for record in episode_records], dtype=int
    )
    rates = np.asarray(
        [float(record["episode_task_completion_rate"]) for record in episode_records],
        dtype=float,
    )
    if as_percentage:
        rates *= 100.0
    return episodes, rates


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if values.size < window or window == 1:
        return np.arange(values.size), values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.arange(window - 1, values.size), np.convolve(values, kernel, mode="valid")


def plot_completion_comparison(
    metrics_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    window: int = 20,
    as_percentage: bool = True,
) -> Path:
    if len(metrics_paths) != len(labels):
        raise ValueError("labels must have the same count as metrics_paths")
    if not metrics_paths:
        raise ValueError("At least one metrics file is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#1479b8", "#d97706", "#17864b", "#a23e8c"]

    for index, (metrics_path, label) in enumerate(zip(metrics_paths, labels)):
        episodes, rates = load_episode_completion(
            metrics_path,
            as_percentage=as_percentage,
        )
        smooth_indices, smooth_rates = moving_average(rates, window)
        smooth_episodes = episodes[smooth_indices]
        color = colors[index % len(colors)]
        ax.plot(
            episodes,
            rates,
            color=color,
            linewidth=0.7,
            alpha=0.16,
        )
        ax.plot(
            smooth_episodes,
            smooth_rates,
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Task Completion Rate (%)" if as_percentage else "Task Completion Rate")
    ax.set_xlim(left=1)
    ax.set_ylim(0.0, 100.0 if as_percentage else 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot episode completion-rate curves for multiple schemes."
    )
    parser.add_argument("metrics_paths", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--fraction",
        action="store_false",
        dest="as_percentage",
        help="Plot completion rate on its native [0, 1] scale instead of percent.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/completion_comparison.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_paths = [Path(path) for path in args.metrics_paths]
    labels = args.labels or [path.stem for path in metrics_paths]
    saved_path = plot_completion_comparison(
        metrics_paths,
        labels,
        Path(args.output),
        window=args.window,
        as_percentage=args.as_percentage,
    )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
