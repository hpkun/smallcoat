from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import build_small_scale_env
from train import build_training_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot UAV movement trajectories.")
    parser.add_argument("--steps", type=int, default=200, help="Number of mobility steps to simulate.")
    parser.add_argument(
        "--profile",
        choices=["paper", "small"],
        default="paper",
        help="Environment profile: 'paper' uses 40 UAV + 25 BS; 'small' uses toy scene.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/figures/uav_trajectories.png",
        help="Output image path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = build_training_env().base_env if args.profile == "paper" else build_small_scale_env().base_env
    slot_length_s = env.simulation_config.slot_length_s
    area_side_length_m = env.simulation_config.area.side_length_m

    trajectories = {
        uav.node_id: [(uav.position.x_m, uav.position.y_m)]
        for uav in env.uavs
    }

    for _ in range(args.steps):
        env.advance_system_dynamics(slot_length_s)
        for uav in env.uavs:
            trajectories[uav.node_id].append((uav.position.x_m, uav.position.y_m))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, len(env.uavs))))
    for idx, uav in enumerate(env.uavs):
        points = trajectories[uav.node_id]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = colors[idx % len(colors)]
        plt.plot(xs, ys, linewidth=1.4, color=color)
        plt.scatter(xs[0], ys[0], marker="o", s=24, color=color, alpha=0.8)
        plt.scatter(xs[-1], ys[-1], marker="x", s=30, color=color, alpha=0.9)

    for bs in env.base_stations:
        plt.scatter(bs.position.x_m, bs.position.y_m, marker="s", s=80, color="black")

    plt.xlim(0, area_side_length_m)
    plt.ylim(0, area_side_length_m)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"UAV Trajectories ({args.steps} steps) | UAV={len(env.uavs)} BS={len(env.base_stations)} "
        f"| profile={args.profile}"
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"profile={args.profile} uavs={len(env.uavs)} bss={len(env.base_stations)} output={output_path}")


if __name__ == "__main__":
    main()
