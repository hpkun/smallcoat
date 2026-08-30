from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reproduction summary metrics.")
    parser.add_argument("--input", default="outputs/reproduction/results.json")
    parser.add_argument("--output", default="outputs/reproduction/comparison.png")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    methods = list(payload["methods"])
    metrics = (("tcr", "Task completion rate (%)"), ("latency_ms", "Average latency (ms)"), ("reliability_pct", "System reliability (%)"), ("cvr", "Constraint violation rate (%)"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        means = [payload["methods"][method]["aggregate"][metric]["mean"] for method in methods]
        stds = [payload["methods"][method]["aggregate"][metric]["std"] for method in methods]
        axis.bar(methods, means, yerr=stds, capsize=3, color="#4472C4")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"saved figure to {output.resolve()}")


if __name__ == "__main__":
    main()
