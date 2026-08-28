from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_redundancy_scheme_bars import SchemeSummary
from scripts.plot_redundancy_scheme_bars import summarize_scheme


def plot_task_completion(
    summaries: list[SchemeSummary],
    output_path: Path,
) -> Path:
    if not summaries:
        raise ValueError("At least one metrics summary is required.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [summary.label for summary in summaries]
    total_tasks = [summary.total_tasks for summary in summaries]
    success_tasks = [summary.success_tasks for summary in summaries]
    x = np.arange(len(labels))
    width = 0.58

    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    total_bars = ax.bar(
        x,
        total_tasks,
        width,
        color="#9bd9e8",
        label="Total Tasks",
    )
    ax.bar(
        x,
        success_tasks,
        width,
        color="#0b7fba",
        label="Successful Tasks",
    )
    for bar, summary in zip(total_bars, summaries):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{summary.task_success_rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Task Number")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
    )
    ax.set_title("(a) Task completion outcomes", y=-0.24)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot total and successful task counts with completion rates."
    )
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        help="Metrics JSON files, one per scheme.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels matching the metrics files. Defaults to file stems.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/task_completion_bars.png",
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
        summarize_scheme(
            metrics_path,
            label,
            episode_level=not args.step_level,
        )
        for metrics_path, label in zip(metrics_paths, labels)
    ]
    output_path = plot_task_completion(summaries, Path(args.output))
    for summary in summaries:
        print(
            f"{summary.label}: total={summary.total_tasks} "
            f"successful={summary.success_tasks} "
            f"completion_rate={summary.task_success_rate:.6f}"
        )
    print(f"figure={output_path}")


if __name__ == "__main__":
    main()
