from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_overall_completion_rate(metrics_path: Path) -> tuple[int, int, float]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episodes = [
        record
        for record in records
        if "episode_total_tasks" in record and "episode_completed_tasks" in record
    ]
    if not episodes:
        raise ValueError(f"No episode completion records found in {metrics_path}")
    total_tasks = int(sum(int(record["episode_total_tasks"]) for record in episodes))
    completed_tasks = int(
        sum(int(record["episode_completed_tasks"]) for record in episodes)
    )
    rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
    return total_tasks, completed_tasks, float(rate)


def plot_fig9_completion(
    metrics_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    scenario: str = "Balanced",
) -> Path:
    if not metrics_paths or len(metrics_paths) != len(labels):
        raise ValueError("metrics_paths and labels must have the same non-zero length")

    rates = [load_overall_completion_rate(path)[2] for path in metrics_paths]
    width = 0.24
    offsets = (np.arange(len(labels), dtype=float) - (len(labels) - 1) / 2.0) * width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    colors = ["#8ec5f4", "#f5f5f5", "#347bb7", "#dc8f32"]
    hatches = [None, "//", "xx", "\\\\"]
    bars = []
    for index, (offset, rate, label) in enumerate(zip(offsets, rates, labels)):
        bar = ax.bar(
            offset,
            rate,
            width,
            color=colors[index % len(colors)],
            edgecolor="#333333",
            linewidth=0.7,
            hatch=hatches[index % len(hatches)],
            label=label,
        )
        bars.append(bar[0])
        ax.text(
            offset,
            rate + 0.018,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks([0.0], [scenario])
    ax.set_ylabel("Task Completion Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.45)
    ax.legend(loc="upper center", frameon=True)
    ax.set_title("Overall Task Completion Rate", pad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a Fig.9-style overall task-completion comparison."
    )
    parser.add_argument("metrics_paths", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--scenario", default="Balanced")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = [Path(path) for path in args.metrics_paths]
    labels = args.labels or [path.stem for path in paths]
    output = plot_fig9_completion(
        paths,
        labels,
        Path(args.output),
        scenario=args.scenario,
    )
    for path, label in zip(paths, labels):
        total, completed, rate = load_overall_completion_rate(path)
        print(
            f"{label}: total={total} completed={completed} "
            f"completion_rate={rate:.6f}"
        )
    print(f"figure={output}")


if __name__ == "__main__":
    main()
