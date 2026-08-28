from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class SchemeSummary:
    label: str
    total_tasks: int
    success_tasks: int
    total_redundant_tasks: int
    success_redundant_tasks: int

    @property
    def task_success_rate(self) -> float:
        if self.total_tasks <= 0:
            return 0.0
        return self.success_tasks / self.total_tasks

    @property
    def redundancy_success_rate(self) -> float:
        if self.total_redundant_tasks <= 0:
            return 0.0
        return self.success_redundant_tasks / self.total_redundant_tasks


def load_records(metrics_path: Path) -> list[dict]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summarize_scheme(metrics_path: Path, label: str, *, episode_level: bool) -> SchemeSummary:
    records = load_records(metrics_path)
    if episode_level:
        records = [record for record in records if "episode_task_completion_rate" in record]
    else:
        records = [record for record in records if "task_completion_rate" in record]
    if not records:
        raise ValueError(f"No matching metric records found in {metrics_path}")

    prefix = "episode_" if episode_level else ""
    total_tasks = int(sum(record.get(f"{prefix}total_tasks", 0) for record in records))
    success_tasks = int(sum(record.get(f"{prefix}completed_tasks", 0) for record in records))
    admitted_key = f"{prefix}admitted_redundant_tasks"
    legacy_redundant_key = f"{prefix}redundant_tasks"
    total_redundant_tasks = int(
        sum(
            record.get(admitted_key, record.get(legacy_redundant_key, 0))
            for record in records
        )
    )
    success_redundant_tasks = int(
        sum(record.get(f"{prefix}redundancy_success_tasks", 0) for record in records)
    )
    if episode_level and total_tasks == 0 and any(
        "task_completion_rate" in record for record in load_records(metrics_path)
    ):
        return summarize_scheme(metrics_path, label, episode_level=False)
    return SchemeSummary(
        label=label,
        total_tasks=total_tasks,
        success_tasks=success_tasks,
        total_redundant_tasks=total_redundant_tasks,
        success_redundant_tasks=success_redundant_tasks,
    )


def annotate_rates(ax: plt.Axes, bars, rates: list[float]) -> None:
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_summaries(summaries: list[SchemeSummary], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [summary.label for summary in summaries]
    x = np.arange(len(labels))
    width = 0.58

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    total_tasks = [summary.total_tasks for summary in summaries]
    success_tasks = [summary.success_tasks for summary in summaries]
    task_rates = [summary.task_success_rate for summary in summaries]
    total_bars = axes[0].bar(
        x,
        total_tasks,
        width,
        color="#9bd9e8",
        label="Total Task",
    )
    axes[0].bar(
        x,
        success_tasks,
        width,
        color="#0b7fba",
        label="Success Task",
    )
    annotate_rates(axes[0], total_bars, task_rates)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Task Number")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].set_title("(a) Different redundancy schemes", y=-0.24)

    redundancy_summaries = [
        summary for summary in summaries if summary.total_redundant_tasks > 0
    ]
    if redundancy_summaries:
        redundancy_labels = [summary.label for summary in redundancy_summaries]
        rx = np.arange(len(redundancy_labels))
        total_redundant = [summary.total_redundant_tasks for summary in redundancy_summaries]
        success_redundant = [
            summary.success_redundant_tasks for summary in redundancy_summaries
        ]
        redundancy_rates = [
            summary.redundancy_success_rate for summary in redundancy_summaries
        ]
        redundant_bars = axes[1].bar(
            rx,
            total_redundant,
            width,
            color="#b8e5ef",
            hatch="/",
            edgecolor="#3b9fb2",
            linewidth=0.0,
            label="Admitted Redundancy",
        )
        axes[1].bar(
            rx,
            success_redundant,
            width,
            color="#ff8b8b",
            hatch="\\",
            edgecolor="#d65d5d",
            linewidth=0.0,
            label="Any-Replica Success",
        )
        annotate_rates(axes[1], redundant_bars, redundancy_rates)
        axes[1].set_xticks(rx, redundancy_labels)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No redundancy records",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xticks([])
    axes[1].set_ylabel("Redundancy Number")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].set_title("(b) Different redundancy schemes", y=-0.24)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Fig.5-style task/redundancy success bars from metrics JSON files.",
    )
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        help="Metrics JSON files, one per scheme.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Scheme labels. Must match the number of metrics files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/redundancy_scheme_bars.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--step-level",
        action="store_true",
        help="Aggregate step-level records instead of episode-level records.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics_paths = [Path(path) for path in args.metrics_paths]
    labels = args.labels or [path.stem for path in metrics_paths]
    if len(labels) != len(metrics_paths):
        raise ValueError("--labels must have the same count as metrics_paths")

    summaries = [
        summarize_scheme(path, label, episode_level=not args.step_level)
        for path, label in zip(metrics_paths, labels)
    ]
    saved_path = plot_summaries(summaries, Path(args.output))
    print(saved_path)


if __name__ == "__main__":
    main()
