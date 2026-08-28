from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def load_cumulative_profit(
    metrics_path: Path,
    *,
    steps_per_episode: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation_records = [
        record
        for record in records
        if record.get("record_type") == "evaluation_step"
        and "time_slot" in record
        and "cumulative_system_profit" in record
    ]
    training_records = [
        record
        for record in records
        if record.get("record_type") in {"training_step", "battery_step"}
        and "time_slot" in record
        and "cumulative_system_profit" in record
    ]
    step_records = evaluation_records or training_records
    if step_records:
        step_records.sort(key=lambda record: int(record["time_slot"]))
        time_slots = np.asarray(
            [int(record["time_slot"]) for record in step_records], dtype=int
        )
        cumulative_profit = np.asarray(
            [float(record["cumulative_system_profit"]) for record in step_records],
            dtype=float,
        )
        return time_slots, cumulative_profit

    episode_records = [
        record
        for record in records
        if record.get("record_type") == "episode" and "episode_system_profit" in record
    ]
    if not episode_records:
        raise ValueError(
            f"No cumulative step or episode profit records found in {metrics_path}."
        )
    episode_records.sort(key=lambda record: int(record["episode"]))
    if steps_per_episode is None:
        logged_steps = [
            int(record["step"])
            for record in records
            if record.get("record_type") == "battery_step" and "step" in record
        ]
        if not logged_steps:
            raise ValueError(
                "steps_per_episode is required when it cannot be inferred from step records."
            )
        steps_per_episode = max(logged_steps) + 1
    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be positive.")
    time_slots = np.asarray(
        [(int(record["episode"]) + 1) * steps_per_episode for record in episode_records],
        dtype=int,
    )
    cumulative_profit = np.cumsum(
        [float(record["episode_system_profit"]) for record in episode_records],
        dtype=float,
    )
    return time_slots, cumulative_profit


def plot_fig6(
    metrics_path: Path,
    output_path: Path,
    *,
    label: str = "Proposed",
    max_time: int | None = None,
    steps_per_episode: int | None = None,
    normalize: bool = True,
) -> Path:
    time_slots, cumulative_profit = load_cumulative_profit(
        metrics_path,
        steps_per_episode=steps_per_episode,
    )
    if max_time is not None:
        if max_time <= 0:
            raise ValueError("max_time must be positive when provided.")
        selected = time_slots <= max_time
        time_slots = time_slots[selected]
        cumulative_profit = cumulative_profit[selected]
    if cumulative_profit.size == 0:
        raise ValueError("No evaluation time-slot records selected for plotting.")
    plotted_max_time = int(time_slots[-1])
    time_slots = np.concatenate(([0], time_slots))
    if normalize:
        denominator = float(cumulative_profit[-1])
        if denominator <= 0.0:
            raise ValueError("Final cumulative system profit must be positive.")
        plotted_profit = cumulative_profit / denominator
    else:
        plotted_profit = cumulative_profit
    plotted_profit = np.concatenate(([0.0], plotted_profit))
    marker_interval = max(1, len(time_slots) // 50)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(
        time_slots,
        plotted_profit,
        color="#8c62bd",
        linewidth=1.6,
        marker="o",
        markersize=4.0,
        markeredgewidth=0.0,
        markevery=marker_interval,
        label=label,
    )
    axis.set_xlabel("Time Slot")
    axis.set_ylabel("Normalized Cumulative Profit" if normalize else "Cumulative System Profit")
    axis.set_xlim(0, plotted_max_time)
    axis.set_ylim(bottom=0.0, top=1.03 if normalize else None)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    if normalize:
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    else:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.grid(True, linestyle="-", linewidth=0.6, alpha=0.45)
    axis.legend(loc="upper left", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 6 cumulative system profit over evaluation time slots."
    )
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--label", default="Proposed")
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Optional time-slot cutoff; defaults to all records in the metrics file.",
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=None,
        help="Training episode length; inferred from step records when omitted.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Plot raw cumulative system profit instead of scaling the final value to 1.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_fig6(
        args.metrics_path,
        args.output,
        label=args.label,
        max_time=args.max_time,
        steps_per_episode=args.steps_per_episode,
        normalize=args.normalize,
    )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
