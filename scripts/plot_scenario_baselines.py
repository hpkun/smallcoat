from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIO_INPUTS = (
    ("delay_sensitive", "Delay-sensitive"),
    ("computation_intensive", "Computation-intensive"),
    ("balanced", "Balanced"),
)


@dataclass(frozen=True)
class ScenarioMetrics:
    label: str
    u_base: float
    u_net: float
    reliable_on_time_completion_rate: float
    episode_count: int
    task_count: int


def _load_episode_records(metrics_path: Path) -> list[dict]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    episode_records = [
        record for record in records if record.get("record_type") == "episode"
    ]
    if not episode_records:
        raise ValueError(f"No episode records found in {metrics_path}.")
    return sorted(episode_records, key=lambda record: int(record.get("episode", 0)))


def summarize_scenario(
    metrics_path: Path,
    *,
    label: str,
    tail_episodes: int,
) -> ScenarioMetrics:
    episode_records = _load_episode_records(metrics_path)
    if tail_episodes > 0:
        episode_records = episode_records[-tail_episodes:]

    required_fields = {
        "episode_u_base",
        "episode_u_net",
        "episode_total_tasks_for_utility",
        "episode_reliable_on_time_tasks",
    }
    missing_fields = required_fields.difference(episode_records[0])
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"{metrics_path} does not contain the new baseline fields: {missing}. "
            "Rerun training with the updated trainer."
        )

    total_tasks = int(
        sum(int(record["episode_total_tasks_for_utility"]) for record in episode_records)
    )
    reliable_tasks = int(
        sum(int(record["episode_reliable_on_time_tasks"]) for record in episode_records)
    )
    reliable_rate = float(reliable_tasks / total_tasks) if total_tasks > 0 else 0.0

    return ScenarioMetrics(
        label=label,
        u_base=float(np.mean([record["episode_u_base"] for record in episode_records])),
        u_net=float(np.mean([record["episode_u_net"] for record in episode_records])),
        reliable_on_time_completion_rate=reliable_rate,
        episode_count=len(episode_records),
        task_count=total_tasks,
    )


def _annotate_bars(ax: plt.Axes, bars, *, digits: int) -> None:
    for bar in bars:
        height = float(bar.get_height())
        offset = 3 if height >= 0 else -14
        vertical_alignment = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
        )


def plot_utility_comparison(
    summaries: list[ScenarioMetrics],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [summary.label for summary in summaries]
    u_base = np.asarray([summary.u_base for summary in summaries], dtype=float)
    u_net = np.asarray([summary.u_net for summary in summaries], dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    base_bars = ax.bar(
        x - width / 2.0,
        u_base,
        width,
        label=r"$U_{base}$",
        color="#2F6B9A",
    )
    net_bars = ax.bar(
        x + width / 2.0,
        u_net,
        width,
        label=r"$U_{net}$",
        color="#D98C2B",
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized Utility")
    ax.set_title("System Utility under Three Task Distributions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    _annotate_bars(ax, base_bars, digits=3)
    _annotate_bars(ax, net_bars, digits=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reliable_completion_comparison(
    summaries: list[ScenarioMetrics],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [summary.label for summary in summaries]
    rates = np.asarray(
        [summary.reliable_on_time_completion_rate for summary in summaries],
        dtype=float,
    )
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.bar(x, rates, width=0.52, color=["#4477AA", "#228833", "#CC6677"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Reliable On-Time Completion Rate")
    ax.set_title("Reliable On-Time Completion under Three Task Distributions")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _annotate_bars(ax, bars, digits=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot baseline-compatible utility and reliable on-time completion "
            "comparisons for the three task distributions."
        )
    )
    parser.add_argument("--delay-sensitive", required=True, type=Path)
    parser.add_argument("--computation-intensive", required=True, type=Path)
    parser.add_argument("--balanced", required=True, type=Path)
    parser.add_argument(
        "--tail-episodes",
        type=int,
        default=100,
        help="Use the last N episodes from each log; use 0 for all episodes.",
    )
    parser.add_argument(
        "--utility-output",
        type=Path,
        default=Path("outputs/figures/scenario_utility.png"),
    )
    parser.add_argument(
        "--reliability-output",
        type=Path,
        default=Path("outputs/figures/scenario_reliable_completion.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tail_episodes < 0:
        raise ValueError("tail_episodes must be non-negative.")

    paths = {
        "delay_sensitive": args.delay_sensitive,
        "computation_intensive": args.computation_intensive,
        "balanced": args.balanced,
    }
    summaries = [
        summarize_scenario(
            paths[key],
            label=label,
            tail_episodes=args.tail_episodes,
        )
        for key, label in SCENARIO_INPUTS
    ]
    print(plot_utility_comparison(summaries, args.utility_output))
    print(plot_reliable_completion_comparison(summaries, args.reliability_output))
    for summary in summaries:
        print(
            f"{summary.label}: episodes={summary.episode_count} "
            f"tasks={summary.task_count} U_base={summary.u_base:.6f} "
            f"U_net={summary.u_net:.6f} "
            "reliable_on_time_completion_rate="
            f"{summary.reliable_on_time_completion_rate:.6f}"
        )


if __name__ == "__main__":
    main()
