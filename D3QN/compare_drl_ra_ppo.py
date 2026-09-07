from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from drl_ra.environment import SAGINEnv
from drl_ra.experiment import (
    build_agent,
    build_hierarchical_agent,
    evaluate_callable,
    evaluate_hierarchical_agent,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DRL-RA and D3QN-PPO success and convergence.")
    parser.add_argument("--drl-ra-history", required=True)
    parser.add_argument("--drl-ra-checkpoint", required=True)
    parser.add_argument("--d3qn-ppo-history", required=True)
    parser.add_argument("--d3qn-ppo-checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10_000, 10_010)))
    parser.add_argument("--evaluation-steps", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="9.4/outputs/comparison")
    return parser.parse_args()


def load_history(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    if not history:
        raise ValueError(f"empty history: {path}")
    return history


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    actual_window = min(max(1, int(window)), len(values))
    if actual_window == 1:
        return values.copy()
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    smoothed[actual_window - 1 :] = np.convolve(
        values,
        np.ones(actual_window, dtype=np.float64) / actual_window,
        mode="valid",
    )
    return smoothed


def convergence_stats(history: list[dict[str, Any]], window: int) -> dict[str, float]:
    episodes = np.asarray([row["episode"] for row in history], dtype=np.float64)
    tcr = np.asarray([row["tcr"] for row in history], dtype=np.float64)
    smoothed = moving_average(tcr, window)
    valid = np.flatnonzero(np.isfinite(smoothed))
    final_window = min(max(1, int(window)), len(tcr))
    final_tcr = float(tcr[-final_window:].mean())
    target = 0.95 * final_tcr
    reached = valid[smoothed[valid] >= target]
    convergence_episode = float(episodes[reached[0]]) if len(reached) else float("nan")
    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc = float(trapezoid(tcr / 100.0, episodes) / max(episodes[-1] - episodes[0], 1.0))
    return {
        "final_window_tcr_pct": final_tcr,
        "convergence_target_pct": target,
        "episode_to_95pct_final": convergence_episode,
        "normalized_tcr_auc": auc,
    }


def load_checkpoint(path: str, expected_method: str, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    method = str(metadata.get("method", expected_method))
    if method != expected_method:
        raise ValueError(f"{path} contains method={method!r}, expected {expected_method!r}")
    config = deepcopy(metadata["config"])
    seed = int(metadata.get("seed", 0))
    probe = SAGINEnv(config, seed=seed)
    if method == "d3qn-ppo":
        agent = build_hierarchical_agent(probe, config, seed, device)
    else:
        agent = build_agent(method, probe, config, seed, device)
    agent.load(path)
    return agent, config


def evaluate_models(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    drl_agent, drl_config = load_checkpoint(args.drl_ra_checkpoint, "drl-ra", args.device)
    ppo_agent, ppo_config = load_checkpoint(args.d3qn_ppo_checkpoint, "d3qn-ppo", args.device)
    drl_config["environment"]["episode_steps"] = args.evaluation_steps
    ppo_config["environment"]["episode_steps"] = args.evaluation_steps

    def drl_policy(state, env, rng):
        return drl_agent.act(state, np.asarray([item.available for item in env.candidates]), epsilon=0.0)

    def run_seed_by_seed(method: str, evaluator, config, agent_or_policy):
        rows: list[dict[str, float]] = []
        total = len(args.seeds)
        for index, seed in enumerate(args.seeds, start=1):
            print(
                f"[{method}] seed {index}/{total} ({seed}), tasks={args.evaluation_steps} ...",
                flush=True,
            )
            started = time.perf_counter()
            seed_rows, _ = evaluator(config, agent_or_policy, [seed])
            rows.extend(seed_rows)
            elapsed = time.perf_counter() - started
            print(
                f"[{method}] seed {seed} done in {elapsed:.1f}s, TCR={seed_rows[0]['tcr']:.2f}%",
                flush=True,
            )
        keys = [key for key in rows[0] if key != "seed"]
        aggregate = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for key in keys
        }
        return rows, aggregate

    drl_rows, drl_aggregate = run_seed_by_seed(
        "DRL-RA",
        evaluate_callable,
        drl_config,
        drl_policy,
    )
    ppo_rows, ppo_aggregate = run_seed_by_seed(
        "D3QN-PPO",
        evaluate_hierarchical_agent,
        ppo_config,
        ppo_agent,
    )
    return {
        "drl-ra": {"runs": drl_rows, "aggregate": drl_aggregate},
        "d3qn-ppo": {"runs": ppo_rows, "aggregate": ppo_aggregate},
    }


def plot_comparison(
    histories: dict[str, list[dict[str, Any]]],
    convergence: dict[str, dict[str, float]],
    evaluation: dict[str, dict[str, Any]],
    window: int,
    output: Path,
) -> None:
    colors = {"drl-ra": "#D1495B", "d3qn-ppo": "#197278"}
    labels = {"drl-ra": "DRL-RA", "d3qn-ppo": "D3QN-PPO"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for method, history in histories.items():
        episodes = np.asarray([row["episode"] for row in history], dtype=np.float64)
        tcr = np.asarray([row["tcr"] for row in history], dtype=np.float64)
        smoothed = moving_average(tcr, window)
        stats = convergence[method]
        convergence_episode = stats["episode_to_95pct_final"]
        label = labels[method]
        if np.isfinite(convergence_episode):
            label += f" (95% at ep. {int(convergence_episode)})"
        axes[0].plot(episodes, smoothed, linewidth=2.0, color=colors[method], label=label)
        axes[0].axhline(
            stats["convergence_target_pct"],
            color=colors[method],
            linewidth=0.9,
            linestyle=":",
            alpha=0.7,
        )
    axes[0].set_title(f"Training Convergence ({window}-episode moving average)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Task completion rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    methods = ["drl-ra", "d3qn-ppo"]
    means = [evaluation[item]["aggregate"]["tcr"]["mean"] for item in methods]
    stds = [evaluation[item]["aggregate"]["tcr"]["std"] for item in methods]
    bars = axes[1].bar(
        [labels[item] for item in methods],
        means,
        yerr=stds,
        capsize=5,
        color=[colors[item] for item in methods],
        width=0.55,
    )
    axes[1].bar_label(bars, labels=[f"{value:.2f}%" for value in means], padding=4)
    axes[1].set_title("Independent Evaluation Success Rate")
    axes[1].set_ylabel("Task completion rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = {
        "drl-ra": load_history(args.drl_ra_history),
        "d3qn-ppo": load_history(args.d3qn_ppo_history),
    }
    lengths = {method: len(history) for method, history in histories.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"training histories must have equal episode counts: {lengths}")
    convergence = {method: convergence_stats(history, args.window) for method, history in histories.items()}
    evaluation = evaluate_models(args)
    payload = {
        "evaluation_steps": args.evaluation_steps,
        "seeds": args.seeds,
        "convergence_definition": "first moving-average TCR reaching 95% of its own final-window mean",
        "convergence": convergence,
        "evaluation": evaluation,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "comparison.json", payload)
    plot_comparison(histories, convergence, evaluation, args.window, output_dir / "success_convergence.png")
    for method in ("drl-ra", "d3qn-ppo"):
        stats = convergence[method]
        tcr = evaluation[method]["aggregate"]["tcr"]
        episode = stats["episode_to_95pct_final"]
        episode_text = "not reached" if not np.isfinite(episode) else str(int(episode))
        print(
            f"{method:10s} evaluation TCR={tcr['mean']:.2f}% +/- {tcr['std']:.2f}% "
            f"convergence_episode={episode_text} TCR-AUC={stats['normalized_tcr_auc']:.4f}"
        )
    print(f"saved comparison to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
