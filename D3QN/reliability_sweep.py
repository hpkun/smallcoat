from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_agent, evaluate_callable, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policies across reliability thresholds.")
    parser.add_argument("--checkpoint-dir", default="output/experiment_paper/checkpoints")
    parser.add_argument("--methods", nargs="+", default=["d3qn", "drl-ra"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95, 0.98])
    parser.add_argument("--evaluation-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="output/experiment_paper/reliability_sweep")
    return parser.parse_args()


def load_policy(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    config = deepcopy(metadata["config"])
    method = str(metadata.get("method", checkpoint.stem.split("_seed")[0]))
    seed = int(metadata.get("seed", 0))
    probe = SAGINEnv(config, seed=seed)
    agent = build_agent(method, probe, config, seed, device)
    agent.load(checkpoint)
    return agent, config, method, seed


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint_dir = Path(args.checkpoint_dir)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for method in args.methods:
        checkpoints = sorted(checkpoint_dir.glob(f"{method}_seed*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no {method} checkpoints found in {checkpoint_dir}")
        results[method] = {}
        loaded = [load_policy(checkpoint, args.device) for checkpoint in checkpoints]
        for threshold in args.thresholds:
            tcr_values: list[float] = []
            cvr_values: list[float] = []
            cost_values: list[float] = []
            for agent, base_config, checkpoint_method, seed in loaded:
                config = deepcopy(base_config)
                config["environment"]["reliability_requirement_override"] = float(threshold)
                config["environment"]["episode_steps"] = int(
                    args.evaluation_steps
                    if args.evaluation_steps is not None
                    else config["training"].get("evaluation_steps", 10_000)
                )

                def policy(state, env, rng, selected=agent):
                    return selected.act(state, [candidate.available for candidate in env.candidates], epsilon=0.0)

                rows, _ = evaluate_callable(config, policy, [10_000 + seed])
                tcr_values.append(rows[0]["tcr"])
                cvr_values.append(rows[0]["cvr"])
                cost_values.append(rows[0]["expected_cost"])
            results[method][f"{threshold:.2f}"] = {
                "tcr": aggregate(tcr_values),
                "cvr": aggregate(cvr_values),
                "expected_cost": aggregate(cost_values),
                "seeds": len(tcr_values),
            }
            print(
                f"{method:10s} threshold={threshold:.2f} "
                f"TCR={np.mean(tcr_values):.2f}% CVR={np.mean(cvr_values):.2f}%"
            )
    return {"thresholds": args.thresholds, "methods": results}


def plot(payload: dict, output: Path) -> None:
    thresholds = np.asarray(payload["thresholds"], dtype=float)
    styles = {
        "d3qn": {"color": "#F28E52", "marker": "^", "linestyle": "-."},
        "drl-ra": {"color": "#E84A5F", "marker": "o", "linestyle": "-"},
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    figure.suptitle("Performance Under Varied Reliability Requirements", fontsize=14, fontweight="bold")
    for method, method_results in payload["methods"].items():
        style = styles.get(method, {"marker": "s", "linestyle": "--"})
        label = method.upper() if method == "d3qn" else method.replace("-", " ").upper()
        for axis, metric in zip(axes, ("tcr", "cvr")):
            means = np.asarray([method_results[f"{value:.2f}"][metric]["mean"] for value in thresholds])
            stds = np.asarray([method_results[f"{value:.2f}"][metric]["std"] for value in thresholds])
            axis.plot(thresholds, means, label=label, linewidth=2, markersize=6, markerfacecolor="white", **style)
            axis.fill_between(thresholds, means - stds, means + stds, color=style.get("color"), alpha=0.13)
    axes[0].set_title("(a) Task Completion Rate", fontweight="bold")
    axes[0].set_ylabel("Task Completion Rate (%)")
    axes[1].set_title("(b) Constraint Violation Rate", fontweight="bold")
    axes[1].set_ylabel("Constraint Violation Rate (%)")
    for axis in axes:
        axis.set_xlabel(r"Minimum Reliability Requirement ($\rho_i^{min}$)")
        axis.set_xticks(thresholds)
        axis.grid(True, alpha=0.25)
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = evaluate(args)
    write_json(output_dir / "reliability_sweep.json", payload)
    plot(payload, output_dir / "reliability_sweep.png")
    print(f"saved sweep data and figure to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
