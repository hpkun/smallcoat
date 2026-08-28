from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_episode_profit(
    metrics_path: Path,
    *,
    per_task: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [
        record for record in records if "episode_system_profit" in record
    ]
    if not episode_records:
        raise ValueError(f"No episode_system_profit records found in {metrics_path}")
    episode_records.sort(key=lambda record: int(record["episode"]))
    episodes = np.asarray(
        [int(record["episode"]) + 1 for record in episode_records], dtype=int
    )
    profits = np.asarray(
        [float(record["episode_system_profit"]) for record in episode_records],
        dtype=float,
    )
    if per_task:
        task_counts = np.asarray(
            [int(record.get("episode_total_tasks", 0)) for record in episode_records],
            dtype=float,
        )
        profits = np.divide(
            profits,
            task_counts,
            out=np.zeros_like(profits),
            where=task_counts > 0,
        )
    return episodes, profits


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if window == 1 or values.size < window:
        return np.arange(values.size), values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.arange(window - 1, values.size), np.convolve(values, kernel, mode="valid")


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def plot_profit_comparison(
    metrics_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    window: int = 20,
    scale: float = 1e9,
    per_task: bool = False,
    normalize: bool = False,
) -> Path:
    if not metrics_paths:
        raise ValueError("At least one metrics file is required")
    if len(metrics_paths) != len(labels):
        raise ValueError("labels must have the same count as metrics_paths")
    if scale <= 0:
        raise ValueError("scale must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#1479b8", "#d97706", "#17864b", "#a23e8c", "#b5483a"]

    for index, (metrics_path, label) in enumerate(zip(metrics_paths, labels)):
        episodes, profits = load_episode_profit(metrics_path, per_task=per_task)
        scaled_profits = minmax_normalize(profits) if normalize else profits / scale
        smooth_indices, smooth_profits = moving_average(scaled_profits, window)
        color = colors[index % len(colors)]
        axis.plot(
            episodes,
            scaled_profits,
            color=color,
            linewidth=0.7,
            alpha=0.18,
        )
        axis.plot(
            episodes[smooth_indices],
            smooth_profits,
            color=color,
            linewidth=2.0,
            label=label,
        )

    quantity = "Profit per Task" if per_task else "System Profit"
    if normalize:
        quantity = f"Normalized {quantity}"
        scale_suffix = ""
    elif scale == 1.0:
        scale_suffix = ""
    elif np.isclose(scale, 1e9):
        scale_suffix = " (billion units)"
    else:
        scale_suffix = f" (scaled by {scale:g})"
    axis.set_xlabel("Episode")
    axis.set_ylabel(f"{quantity}{scale_suffix}")
    axis.set_xlim(left=1)
    if normalize:
        axis.set_ylim(0.0, 1.05)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axis.legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_paper_cumulative_profit(
    metrics_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    steps_per_episode: int = 50,
    max_time: int | None = None,
) -> Path:
    """Plot cumulative profit against the discrete simulation time-slot index."""

    if not metrics_paths:
        raise ValueError("At least one metrics file is required")
    if len(metrics_paths) != len(labels):
        raise ValueError("labels must have the same count as metrics_paths")
    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be positive")
    if max_time is not None and max_time <= 0:
        raise ValueError("max_time must be positive when provided")

    series: list[tuple[np.ndarray, np.ndarray]] = []
    for metrics_path in metrics_paths:
        episodes, profits = load_episode_profit(metrics_path)
        time_values = episodes * steps_per_episode
        if max_time is not None:
            selected = time_values <= max_time
            time_values = time_values[selected]
            profits = profits[selected]
        if profits.size == 0:
            raise ValueError(
                f"No profit records remain for {metrics_path} at max_time={max_time}"
            )
        cumulative_profit = np.cumsum(profits)
        series.append(
            (
                np.concatenate([np.array([0], dtype=int), time_values]),
                np.concatenate([np.array([0.0]), cumulative_profit]),
            )
        )

    shared_denominator = max(float(values[-1]) for _, values in series)
    if shared_denominator <= 0.0:
        raise ValueError("Final cumulative profit must be positive for normalization")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#7a4fa3", "#d97706", "#1479b8", "#17864b", "#b5483a"]
    markers = ["o", "P", "d", "s", "^"]
    for index, ((time_values, cumulative_profit), label) in enumerate(
        zip(series, labels)
    ):
        marker_interval = max(1, len(time_values) // 50)
        axis.plot(
            time_values,
            cumulative_profit / shared_denominator,
            color=colors[index % len(colors)],
            linestyle="-",
            linewidth=1.6,
            marker=markers[index % len(markers)],
            markersize=4.5,
            markeredgewidth=0.8,
            markevery=marker_interval,
            label=label,
        )

    axis.set_xlabel("Time Slot")
    axis.set_ylabel("Profit")
    axis.set_xlim(left=0, right=max_time if max_time is not None else None)
    axis.set_ylim(0.0, 1.03)
    axis.grid(True, linestyle="-", linewidth=0.6, alpha=0.45)
    axis.legend(loc="upper left", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot episode system-profit curves from one or more metrics files."
    )
    parser.add_argument("metrics_paths", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--scale",
        type=float,
        default=1e9,
        help="Divide profit values by this number before plotting.",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Plot episode profit divided by episode task count.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Min-max normalize each episode-profit series to [0, 1].",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help=(
            "Plot cumulative system profit over time, normalized by the largest "
            "final cumulative profit across all input schemes."
        ),
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=50,
        help="Simulation time slots represented by each episode record in paper-style mode.",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        help="Optional maximum time to include in paper-style mode.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/profit_comparison.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_paths = [Path(path) for path in args.metrics_paths]
    labels = args.labels or [path.stem for path in metrics_paths]
    if args.paper_style:
        if args.per_task:
            raise ValueError("--per-task cannot be combined with --paper-style")
        saved_path = plot_paper_cumulative_profit(
            metrics_paths,
            labels,
            Path(args.output),
            steps_per_episode=args.steps_per_episode,
            max_time=args.max_time,
        )
    else:
        saved_path = plot_profit_comparison(
            metrics_paths,
            labels,
            Path(args.output),
            window=args.window,
            scale=args.scale,
            per_task=args.per_task,
            normalize=args.normalize,
        )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
