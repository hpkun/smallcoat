from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def load_instant_profit(metrics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation_records = [
        record
        for record in records
        if record.get("record_type") == "evaluation_step"
        and "time_slot" in record
        and "system_profit" in record
    ]
    training_records = [
        record
        for record in records
        if record.get("record_type") in {"training_step", "battery_step"}
        and "time_slot" in record
        and "system_profit" in record
    ]
    step_records = evaluation_records or training_records
    if not step_records:
        raise ValueError(
            f"No evaluation or training step-profit records found in {metrics_path}."
        )
    step_records.sort(key=lambda record: int(record["time_slot"]))
    time_slots = np.asarray(
        [int(record["time_slot"]) for record in step_records], dtype=int
    )
    profits = np.asarray(
        [float(record["system_profit"]) for record in step_records], dtype=float
    )
    return time_slots, profits


def moving_average(
    time_slots: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if values.size < window:
        raise ValueError("window cannot exceed the number of selected time slots")
    kernel = np.ones(window, dtype=float) / float(window)
    return time_slots[window - 1 :], np.convolve(values, kernel, mode="valid")


def plot_instant_profit(
    metrics_path: Path,
    output_path: Path,
    *,
    label: str = "Proposed",
    max_time: int | None = None,
    window: int = 100,
) -> Path:
    time_slots, profits = load_instant_profit(metrics_path)
    if max_time is not None:
        if max_time <= 0:
            raise ValueError("max_time must be positive when provided.")
        selected = time_slots <= max_time
        time_slots = time_slots[selected]
        profits = profits[selected]
    if profits.size == 0:
        raise ValueError("No evaluation time-slot records selected for plotting.")
    plotted_max_time = int(time_slots[-1])

    smooth_time, smooth_profit = moving_average(time_slots, profits, window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(
        time_slots,
        profits,
        color="#7b8a97",
        linewidth=0.65,
        alpha=0.24,
        label="Instantaneous Profit",
    )
    axis.plot(
        smooth_time,
        smooth_profit,
        color="#1479b8",
        linewidth=2.0,
        label=f"{window}-slot Moving Average",
    )
    axis.set_xlabel("Time Slot")
    axis.set_ylabel("System Profit")
    axis.set_xlim(0, plotted_max_time)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.grid(True, linestyle="-", linewidth=0.6, alpha=0.35)
    axis.legend(loc="upper right", frameon=True, title=label)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot instantaneous system profit and its moving average over "
            "evaluation time slots."
        )
    )
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--label", default="Proposed")
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Optional time-slot cutoff; defaults to all records in the metrics file.",
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_instant_profit(
        args.metrics_path,
        args.output,
        label=args.label,
        max_time=args.max_time,
        window=args.window,
    )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
