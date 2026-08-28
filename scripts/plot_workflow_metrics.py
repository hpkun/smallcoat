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

from src.plotting import moving_average


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot workflow-level metrics, including makespan and SLA violations.",
    )
    parser.add_argument(
        "metrics_path",
        help="Path to the training metrics JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output image. Defaults to outputs/figures/<metrics>_workflow_metrics.png.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving-average smoothing window.",
    )
    return parser


def default_output_path(metrics_path: Path) -> Path:
    return Path("outputs") / "figures" / f"{metrics_path.stem}_workflow_metrics.png"


def load_episode_records(metrics_path: Path) -> list[dict]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [
        record
        for record in records
        if "episode_avg_completed_workflow_makespan_s" in record
    ]
    if not episode_records:
        raise ValueError(f"No workflow episode records found in {metrics_path}")
    return episode_records


def smooth_series(x_values: list[int], y_values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    smoothed_x, smoothed_y = moving_average(y_values, window)
    if smoothed_x.size == 0:
        return np.asarray(x_values, dtype=int), np.asarray(y_values, dtype=float)
    x = np.asarray(x_values, dtype=int)
    return x[smoothed_x], smoothed_y


def plot_workflow_metrics(
    metrics_path: Path,
    *,
    output_path: Path,
    window: int,
) -> Path:
    episode_records = load_episode_records(metrics_path)
    episodes = [int(record["episode"]) + 1 for record in episode_records]

    series = {
        "Avg Workflow Makespan": (
            [float(record.get("episode_avg_completed_workflow_makespan_s", 0.0)) for record in episode_records],
            "Seconds",
        ),
        "Max Workflow Makespan": (
            [float(record.get("episode_max_completed_workflow_makespan_s", 0.0)) for record in episode_records],
            "Seconds",
        ),
        "Completed Workflows": (
            [float(record.get("episode_completed_workflows", 0.0)) for record in episode_records],
            "Count",
        ),
        "Failed Workflows": (
            [float(record.get("episode_failed_workflows", 0.0)) for record in episode_records],
            "Count",
        ),
        "Workflow SLA Violation Rate": (
            [float(record.get("episode_workflow_sla_violation_rate", 0.0)) for record in episode_records],
            "Rate",
        ),
        "Task Completion Rate": (
            [float(record.get("episode_task_completion_rate", 0.0)) for record in episode_records],
            "Rate",
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=150)
    fig.suptitle(f"Workflow Metrics: {metrics_path.name}", fontsize=14)

    for ax, (title, (values, ylabel)) in zip(axes.ravel(), series.items()):
        x, y = smooth_series(episodes, values, window)
        ax.plot(x, y, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = Path(args.output) if args.output else default_output_path(metrics_path)
    saved_path = plot_workflow_metrics(
        metrics_path,
        output_path=output_path,
        window=args.window,
    )
    print(saved_path)


if __name__ == "__main__":
    main()

