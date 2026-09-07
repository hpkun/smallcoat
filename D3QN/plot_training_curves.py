from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward, TCR, and latency training curves.")
    parser.add_argument("--history", default="outputs/d3qn_seed42/history.json")
    parser.add_argument("--output", default="outputs/d3qn_seed42/convergence.png")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    with Path(args.history).open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    if not history:
        raise ValueError("history is empty")

    episodes = np.asarray([row["episode"] for row in history], dtype=float)
    series = (
        ("Reward", "reward", "Episode reward", None),
        ("Task Completion Rate", "tcr", "TCR (%)", (0, 100)),
        ("Average Latency", "latency_ms", "Latency (ms)", None),
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    figure.suptitle("D3QN Training Convergence (Seed 42)", fontsize=16, fontweight="bold")
    for axis, (title, key, ylabel, limits) in zip(axes, series):
        values = np.asarray([row[key] for row in history], dtype=float)
        smoothed = moving_average(values, args.window)
        axis.plot(episodes, values, color="#7DA0D4", alpha=0.25, linewidth=0.8, label="Per episode")
        axis.plot(episodes, smoothed, color="#2457A6", linewidth=2.0, label=f"{args.window}-episode moving average")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", frameon=False)
        if limits is not None:
            axis.set_ylim(*limits)
    axes[-1].set_xlabel("Episode")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"saved convergence figure to {output.resolve()}")


if __name__ == "__main__":
    main()
