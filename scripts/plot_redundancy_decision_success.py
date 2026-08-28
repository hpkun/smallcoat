from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RedundancyDecisionSummary:
    label: str
    admitted_tasks: int
    success_tasks: int

    @property
    def success_rate(self) -> float:
        if self.admitted_tasks <= 0:
            return 0.0
        return self.success_tasks / self.admitted_tasks

    @property
    def decision_tasks(self) -> int:
        """Backward-compatible alias for callers using the old field name."""

        return self.admitted_tasks


def summarize_metrics(
    metrics_path: Path,
    label: str,
    *,
    episode_level: bool = True,
) -> RedundancyDecisionSummary:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    prefix = "episode_" if episode_level else ""
    admitted_key = f"{prefix}admitted_redundant_tasks"
    legacy_admitted_key = f"{prefix}redundant_tasks"
    success_key = f"{prefix}redundancy_success_tasks"
    matching_records = [record for record in records if success_key in record]
    if not matching_records:
        raise ValueError(
            f"{metrics_path} does not contain {success_key}."
        )

    admitted_tasks = int(
        sum(
            int(record.get(admitted_key, record.get(legacy_admitted_key, 0)))
            for record in matching_records
        )
    )
    success_tasks = int(
        sum(int(record.get(success_key, 0)) for record in matching_records)
    )
    if success_tasks > admitted_tasks:
        raise ValueError(
            f"Invalid redundancy metrics in {metrics_path}: success tasks "
            f"({success_tasks}) exceed admitted tasks ({admitted_tasks})."
        )
    return RedundancyDecisionSummary(
        label=label,
        admitted_tasks=admitted_tasks,
        success_tasks=success_tasks,
    )


def plot_summaries(
    summaries: list[RedundancyDecisionSummary],
    output_path: Path,
) -> Path:
    if not summaries:
        raise ValueError("At least one metrics summary is required.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [summary.label for summary in summaries]
    admitted_tasks = [summary.admitted_tasks for summary in summaries]
    success_tasks = [summary.success_tasks for summary in summaries]
    x = np.arange(len(labels))
    width = 0.58

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    admitted_bars = ax.bar(
        x,
        admitted_tasks,
        width,
        color="#b8e5ef",
        hatch="/",
        edgecolor="#3b9fb2",
        linewidth=0.0,
        label="Admitted Redundancy",
    )
    ax.bar(
        x,
        success_tasks,
        width,
        color="#ff8b8b",
        hatch="\\",
        edgecolor="#d65d5d",
        linewidth=0.0,
        label="Any-Replica Success",
    )
    for bar, summary in zip(admitted_bars, summaries):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{summary.success_rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Redundancy Number")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("(b) Admitted redundancy outcomes", y=-0.24)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot admitted redundant tasks completed by at least one replica."
        )
    )
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        help="Metrics JSON files containing redundancy success fields.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels matching the metrics files. Defaults to file stems.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/redundancy_decision_success.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--step-level",
        action="store_true",
        help="Aggregate step-level records instead of episode records.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_paths = [Path(path) for path in args.metrics_paths]
    labels = args.labels or [path.stem for path in metrics_paths]
    if len(labels) != len(metrics_paths):
        raise ValueError("--labels must have the same count as metrics_paths.")

    summaries = [
        summarize_metrics(
            metrics_path,
            label,
            episode_level=not args.step_level,
        )
        for metrics_path, label in zip(metrics_paths, labels)
    ]
    output_path = plot_summaries(summaries, Path(args.output))
    for summary in summaries:
        print(
            f"{summary.label}: admitted={summary.admitted_tasks} "
            f"completed={summary.success_tasks} "
            f"success_rate={summary.success_rate:.6f}"
        )
    print(f"figure={output_path}")


if __name__ == "__main__":
    main()
