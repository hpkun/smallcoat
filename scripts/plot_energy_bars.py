from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_episode_energy(metrics_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 episode 级传输、计算和总能耗。"""

    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_records = [
        record for record in records if "episode_total_energy_j" in record
    ]
    if not episode_records:
        raise ValueError(
            "日志中没有 episode_total_energy_j。请使用加入能耗日志后的代码重新训练。"
        )
    episode_records.sort(key=lambda record: int(record.get("episode", 0)))
    transmission = np.asarray(
        [record.get("episode_transmission_energy_j", 0.0) for record in episode_records],
        dtype=np.float64,
    )
    computing = np.asarray(
        [record.get("episode_computing_energy_j", 0.0) for record in episode_records],
        dtype=np.float64,
    )
    total = np.asarray(
        [record["episode_total_energy_j"] for record in episode_records],
        dtype=np.float64,
    )
    return transmission, computing, total


def group_energy(values: np.ndarray, group_size: int) -> np.ndarray:
    """按连续轮次分组求和；group_size=1 时保留逐轮数据。"""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return np.asarray(
        [values[index : index + group_size].sum() for index in range(0, len(values), group_size)],
        dtype=np.float64,
    )


def plot_energy_bars(
    metrics_path: Path,
    output_path: Path,
    *,
    group_size: int = 1,
) -> Path:
    """绘制系统总能耗柱状图和逐轮（或分组轮次）能耗折线图。"""

    transmission, computing, total = load_episode_energy(metrics_path)
    grouped_transmission = group_energy(transmission, group_size)
    grouped_computing = group_energy(computing, group_size)
    grouped_total = group_energy(total, group_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {"transmission": "#2a9d8f", "computing": "#e76f51"}

    total_transmission = float(transmission.sum())
    total_computing = float(computing.sum())
    total_system = float(total.sum())
    computing_bar = axes[0].bar(
        [0],
        [total_computing],
        width=0.55,
        color=colors["computing"],
        label="Computing",
    )
    transmission_bar = axes[0].bar(
        [1],
        [total_transmission],
        width=0.55,
        color=colors["transmission"],
        label="Transmission",
    )
    axes[0].bar_label(computing_bar, labels=[f"{total_computing:,.1f} J"], padding=3)
    axes[0].bar_label(
        transmission_bar,
        labels=[f"{total_transmission:,.1f} J"],
        padding=3,
    )
    axes[0].set_xlim(-0.6, 1.6)
    axes[0].set_ylim(0, max(total_computing, total_transmission, 1.0) * 1.1)
    axes[0].set_xticks([0, 1], ["Computing", "Transmission"])
    axes[0].set_title(f"Total system energy: {total_system:,.1f} J")
    axes[0].set_ylabel("Energy (J)")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    x = np.arange(len(grouped_total))
    axes[1].plot(
        x,
        grouped_transmission,
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        color=colors["transmission"],
        label="Transmission",
    )
    axes[1].plot(
        x,
        grouped_computing,
        marker="s",
        markersize=3.0,
        linewidth=1.2,
        color=colors["computing"],
        label="Computing",
    )
    axes[1].plot(
        x,
        grouped_total,
        linewidth=2.0,
        color="#264653",
        label="Total",
    )
    tick_step = max(1, len(x) // 12)
    tick_positions = x[::tick_step]
    tick_labels = [
        str(int(position * group_size + 1))
        if group_size == 1
        else f"{int(position * group_size + 1)}-{min(int((position + 1) * group_size), len(total))}"
        for position in tick_positions
    ]
    axes[1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[1].set_title(
        "Energy per episode" if group_size == 1 else f"Energy per {group_size} episodes"
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Energy (J)")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    axes[1].legend(frameon=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制系统总能耗柱状图和每轮能耗折线图。")
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default="outputs/metrics/train_metrics.json",
        help="训练指标 JSON 文件。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/figures/energy_bars.png",
        help="输出图片路径。",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="每个折线数据点汇总的 episode 数，默认 1 表示逐轮绘制。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_energy_bars(
        Path(args.metrics_path),
        Path(args.output),
        group_size=args.group_size,
    )
    print(saved_path)


if __name__ == "__main__":
    main()
