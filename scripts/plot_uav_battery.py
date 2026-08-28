from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_battery_series(
    metrics_path: Path,
    *,
    level: str = "step",
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load remaining energy and battery level for every UAV."""

    records: list[dict[str, Any]] = json.loads(metrics_path.read_text(encoding="utf-8"))
    status_key = "battery_status" if level == "step" else "episode_battery_status"
    selected = [record for record in records if status_key in record]
    if not selected:
        raise ValueError(
            f"No {status_key} records found. Train again with battery logging enabled."
        )

    if level == "step":
        selected.sort(key=lambda record: (int(record["episode"]), int(record["step"])))
        labels = [f'{int(record["episode"])}:{int(record["step"])}' for record in selected]
    else:
        selected.sort(key=lambda record: int(record["episode"]))
        labels = [str(int(record["episode"])) for record in selected]

    def uav_sort_key(value: str) -> tuple[int, int | str]:
        suffix = value.rsplit("-", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, value)

    uav_ids = sorted(
        {uav_id for record in selected for uav_id in record[status_key]},
        key=uav_sort_key,
    )
    remaining: dict[str, np.ndarray] = {}
    levels: dict[str, np.ndarray] = {}
    for uav_id in uav_ids:
        remaining[uav_id] = np.asarray(
            [record[status_key].get(uav_id, {}).get("remaining_energy_j", np.nan) for record in selected],
            dtype=np.float64,
        )
        levels[uav_id] = np.asarray(
            [record[status_key].get(uav_id, {}).get("battery_level", np.nan) for record in selected],
            dtype=np.float64,
        )
    return labels, remaining, levels


def plot_uav_battery(
    metrics_path: Path,
    output_path: Path,
    *,
    level: str = "step",
    uav_ids: list[str] | None = None,
) -> Path:
    labels, remaining, levels = load_battery_series(metrics_path, level=level)
    selected_ids = uav_ids or list(remaining)
    unknown = sorted(set(selected_ids) - set(remaining))
    if unknown:
        raise ValueError(f"Unknown UAV IDs: {', '.join(unknown)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(labels))
    for uav_id in selected_ids:
        axes[0].plot(x, remaining[uav_id], linewidth=1.4, label=uav_id)
        axes[1].plot(x, levels[uav_id] * 100.0, linewidth=1.4, label=uav_id)

    axes[0].set_ylabel("Remaining energy (J)")
    axes[0].set_title(f"UAV battery at each {level}")
    axes[1].set_ylabel("Battery level (%)")
    axes[1].set_xlabel("Episode:step" if level == "step" else "Episode")
    axes[1].set_ylim(0.0, 102.0)
    for axis in axes:
        axis.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
        axis.legend(frameon=False, ncols=min(5, max(1, len(selected_ids))))

    tick_step = max(1, len(x) // 12)
    tick_positions = x[::tick_step]
    axes[1].set_xticks(tick_positions, [labels[index] for index in tick_positions], rotation=45, ha="right")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot per-UAV battery history from training metrics.")
    parser.add_argument("metrics_path", nargs="?", default="outputs/metrics/train_metrics.json")
    parser.add_argument("-o", "--output", default="outputs/figures/uav_battery.png")
    parser.add_argument("--level", choices=("step", "episode"), default="step")
    parser.add_argument("--uavs", nargs="+", help="UAV IDs to plot; default: all UAVs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_uav_battery(
        Path(args.metrics_path),
        Path(args.output),
        level=args.level,
        uav_ids=args.uavs,
    )
    print(saved_path)


if __name__ == "__main__":
    main()
