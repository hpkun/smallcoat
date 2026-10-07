from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_agent, build_hierarchical_agent, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot empirical reliability success rate by replica count."
    )
    parser.add_argument("--checkpoint-dir", default="output/experiment_smoke/checkpoints")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Evaluate one checkpoint directly; overrides --checkpoint-dir and --methods.",
    )
    parser.add_argument("--methods", nargs="+", default=["drl-ra"])
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=10_000,
        help="Tasks evaluated per checkpoint (default: 10000).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="output/experiment_smoke/redundancy_success")
    return parser.parse_args()


def load_policy(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if "config" not in metadata:
        raise ValueError(f"checkpoint has no saved config: {checkpoint}")
    config = deepcopy(metadata["config"])
    method = str(metadata.get("method", checkpoint.stem.split("_seed")[0]))
    seed = int(metadata.get("seed", 0))
    probe = SAGINEnv(config, seed=seed)
    if method == "d3qn-ppo":
        agent = build_hierarchical_agent(probe, config, seed, device)
    else:
        agent = build_agent(method, probe, config, seed, device)
    agent.load(checkpoint)
    return agent, config, method, seed


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    probability = successes / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2.0 * total)) / denominator
    margin = z * sqrt(probability * (1.0 - probability) / total + z**2 / (4.0 * total**2)) / denominator
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


def aggregate_counts(counts: dict[int, tuple[int, int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for replicas in sorted(counts):
        successes, total = counts[replicas]
        rate = 100.0 * successes / total
        lower, upper = wilson_interval(successes, total)
        result[str(replicas)] = {
            "tasks": total,
            "successes": successes,
            "success_rate_pct": rate,
            "ci95_low_pct": lower,
            "ci95_high_pct": upper,
        }
    return result


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        replicas = int(record["replicas"])
        counts[replicas][0] += int(record["reliability_success"])
        counts[replicas][1] += 1
    return aggregate_counts(
        {replicas: (values[0], values[1]) for replicas, values in counts.items()}
    )


def evaluate_checkpoint(checkpoint: Path, device: str, evaluation_steps: int | None) -> tuple[str, dict[str, dict[str, float | int]]]:
    agent, config, method, seed = load_policy(checkpoint, device)
    config["environment"]["episode_steps"] = int(
        evaluation_steps
        if evaluation_steps is not None
        else config["training"].get("evaluation_steps", 10_000)
    )
    evaluation_seed = 10_000 + seed
    seed_everything(evaluation_seed)
    env = SAGINEnv(config, seed=evaluation_seed)

    if method == "d3qn-ppo":
        env.reset(seed=evaluation_seed)
        done = False
        while not done:
            ground_action, ntl_action, context, _, _, _ = agent.decide(
                env, epsilon=0.0, deterministic_ntl=True
            )
            _, _, terminated, truncated, _ = env.step_hierarchical(
                ground_action, ntl_action, context=context
            )
            done = terminated or truncated
    else:
        state, info = env.reset(seed=evaluation_seed)
        mask = info["action_mask"]
        done = False
        while not done:
            action = agent.act(state, mask, epsilon=0.0)
            state, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated

    return method, aggregate_records(env.metrics)


def combine_checkpoints(
    checkpoint_results: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for result in checkpoint_results:
        for replicas, values in result.items():
            counts[int(replicas)][0] += int(values["successes"])
            counts[int(replicas)][1] += int(values["tasks"])

    return aggregate_counts(
        {replicas: (values[0], values[1]) for replicas, values in counts.items()}
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        method, grouped = evaluate_checkpoint(
            checkpoint, args.device, args.evaluation_steps
        )
        for replicas, values in grouped.items():
            print(
                f"{method:10s} replicas={replicas} tasks={values['tasks']} "
                f"success={values['success_rate_pct']:.2f}%"
            )
        return {
            "metric": "empirical_reliability_success_rate",
            "checkpoint": str(checkpoint),
            "checkpoint_counts": {method: 1},
            "methods": {method: grouped},
        }

    checkpoint_dir = Path(args.checkpoint_dir)
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    checkpoint_counts: dict[str, int] = {}
    for requested_method in args.methods:
        checkpoints = sorted(checkpoint_dir.glob(f"{requested_method}_seed*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no {requested_method} checkpoints found in {checkpoint_dir}")
        per_checkpoint = []
        for checkpoint in checkpoints:
            method, grouped = evaluate_checkpoint(checkpoint, args.device, args.evaluation_steps)
            if method != requested_method:
                raise ValueError(
                    f"checkpoint method is {method!r}, expected {requested_method!r}: {checkpoint}"
                )
            per_checkpoint.append(grouped)
        results[requested_method] = combine_checkpoints(per_checkpoint)
        checkpoint_counts[requested_method] = len(checkpoints)
        for replicas, values in results[requested_method].items():
            print(
                f"{requested_method:10s} replicas={replicas} tasks={values['tasks']} "
                f"success={values['success_rate_pct']:.2f}%"
            )
    return {
        "metric": "empirical_reliability_success_rate",
        "checkpoint_counts": checkpoint_counts,
        "methods": results,
    }


def plot(payload: dict[str, Any], output: Path) -> None:
    styles = {
        "d3qn": {"color": "#3B6FB6", "marker": "^", "linestyle": "-."},
        "drl-ra": {"color": "#D1495B", "marker": "o", "linestyle": "-"},
        "d3qn-ppo": {"color": "#20866F", "marker": "s", "linestyle": "--"},
    }
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for method, grouped in payload["methods"].items():
        replicas = np.asarray(sorted(int(value) for value in grouped), dtype=int)
        values = [grouped[str(value)] for value in replicas]
        rates = np.asarray([row["success_rate_pct"] for row in values], dtype=float)
        lower = np.asarray([row["ci95_low_pct"] for row in values], dtype=float)
        upper = np.asarray([row["ci95_high_pct"] for row in values], dtype=float)
        errors = np.vstack((rates - lower, upper - rates))
        style = styles.get(method, {"marker": "o", "linestyle": "-"})
        label = method.upper() if method == "d3qn" else method.replace("-", " ").upper()
        axis.errorbar(
            replicas,
            rates,
            yerr=errors,
            label=label,
            linewidth=2,
            markersize=7,
            markerfacecolor="white",
            capsize=4,
            **style,
        )
        for replica_count, rate, row in zip(replicas, rates, values):
            axis.annotate(
                f"n={row['tasks']}",
                (replica_count, rate),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=style.get("color"),
            )

    all_replicas = sorted(
        {int(value) for grouped in payload["methods"].values() for value in grouped}
    )
    all_lower_bounds = [
        float(row["ci95_low_pct"])
        for grouped in payload["methods"].values()
        for row in grouped.values()
    ]
    y_min = max(0.0, 5.0 * np.floor((min(all_lower_bounds) - 2.0) / 5.0))
    axis.set_title("Reliability Success Rate by Replica Count", fontweight="bold")
    axis.set_xlabel("Total Number of Replicas")
    axis.set_ylabel("Empirical Reliability Success Rate (%)")
    axis.set_xticks(all_replicas)
    axis.set_ylim(y_min, 100.5)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = evaluate(args)
    write_json(output_dir / "redundancy_success.json", payload)
    plot(payload, output_dir / "redundancy_success.png")
    print(f"saved redundancy success data and figure to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
