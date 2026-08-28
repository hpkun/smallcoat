from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAILURE_METRICS = (
    ("episode_task_timeout_or_drop_rate", "Timeout / Drop Rate", "#b5483a"),
    ("episode_task_deadline_failure_rate", "Deadline Failure Rate", "#d97706"),
    ("episode_task_capacity_drop_rate", "Capacity Drop Rate", "#1479b8"),
    ("episode_reliability_failure_rate", "Reliability Failure Rate", "#7a4fa3"),
)


def load_episode_failure_rates(
    metrics_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [
        record
        for record in records
        if "episode_task_timeout_or_drop_rate" in record
    ]
    if not episode_records:
        raise ValueError(f"No episode failure-rate records found in {metrics_path}")
    episode_records.sort(key=lambda record: int(record["episode"]))
    episodes = np.asarray(
        [int(record["episode"]) + 1 for record in episode_records], dtype=int
    )
    rates = {
        key: np.asarray(
            [float(record.get(key, 0.0)) * 100.0 for record in episode_records],
            dtype=float,
        )
        for key, _, _ in FAILURE_METRICS
    }
    return episodes, rates


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if window == 1 or values.size < window:
        return np.arange(values.size), values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.arange(window - 1, values.size), np.convolve(values, kernel, mode="valid")


def plot_episode_failure_rates(
    metrics_path: Path,
    output_path: Path,
    *,
    window: int = 20,
) -> Path:
    episodes, rates = load_episode_failure_rates(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_values = np.concatenate(list(rates.values()))
    common_top = min(
        100.0,
        max(5.0, float(np.ceil((float(all_values.max()) + 2.0) / 5.0) * 5.0)),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, (key, title, color) in zip(axes.flat, FAILURE_METRICS):
        values = rates[key]
        smooth_indices, smooth_values = moving_average(values, window)
        axis.plot(
            episodes,
            values,
            color=color,
            linewidth=0.7,
            alpha=0.2,
            label="Raw",
        )
        axis.plot(
            episodes[smooth_indices],
            smooth_values,
            color=color,
            linewidth=2.0,
            label=f"Moving Average ({window})",
        )
        axis.set_title(title)
        axis.set_xlim(left=1)
        axis.set_ylim(0.0, common_top)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    axes[0, 0].set_ylabel("Rate (%)")
    axes[1, 0].set_ylabel("Rate (%)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot episode-level task failure rates from a metrics JSON file."
    )
    parser.add_argument("metrics_path")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--output",
        default="outputs/figures/episode_failure_rates.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_episode_failure_rates(
        Path(args.metrics_path),
        Path(args.output),
        window=args.window,
    )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
