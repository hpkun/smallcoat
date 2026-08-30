from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from drl_ra.baselines import POLICIES
from drl_ra.config import apply_overrides, load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import evaluate_callable, train_agent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible paper comparison.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--profile", choices=("smoke", "quick", "paper"), default="quick")
    parser.add_argument("--methods", nargs="+", default=["random", "greedy-nearest", "greedy-reliability", "dqn", "d3qn", "drl-ra"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/reproduction")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def profile_config(config: dict, profile: str) -> tuple[dict, list[int]]:
    result = deepcopy(config)
    if profile == "smoke":
        result["training"]["episodes"] = 1
        result["environment"]["episode_steps"] = 70
        result["training"]["batch_size"] = 16
        return result, [0]
    if profile == "quick":
        result["training"]["episodes"] = 30
        result["environment"]["episode_steps"] = 200
        return result, [0, 1, 2]
    return result, list(range(10))


def main() -> None:
    args = parse_args()
    config, default_seeds = profile_config(load_config(args.config), args.profile)
    config = apply_overrides(config, args.set)
    seeds = args.seeds if args.seeds is not None else default_seeds
    output_dir = Path(args.output_dir)
    all_results: dict[str, dict] = {}
    for method in args.methods:
        method_config = deepcopy(config)
        if method in POLICIES:
            baseline = POLICIES[method]

            def policy(state, env: SAGINEnv, rng, selected=baseline):
                mask = [candidate.available for candidate in env.candidates]
                return selected(mask, env.candidates, env.current_task, rng)

            rows, aggregate = evaluate_callable(method_config, policy, seeds)
        else:
            rows = []
            for seed in seeds:
                train_config = deepcopy(method_config)
                agent, history = train_agent(train_config, method, seed, device=args.device, progress=False)
                checkpoint = output_dir / "checkpoints" / f"{method}_seed{seed}.pt"
                agent.save(checkpoint, metadata={"method": method, "seed": seed, "config": train_config})

                def policy(state, env: SAGINEnv, rng, selected=agent):
                    return selected.act(state, [candidate.available for candidate in env.candidates], epsilon=0.0)

                evaluation, _ = evaluate_callable(train_config, policy, [10_000 + seed])
                rows.extend(evaluation)
            keys = [key for key in rows[0] if key != "seed"]
            import numpy as np
            aggregate = {
                key: {
                    "mean": float(np.mean([row[key] for row in rows])),
                    "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                }
                for key in keys
            }
        all_results[method] = {"runs": rows, "aggregate": aggregate}
        print(f"{method:20s} TCR={aggregate['tcr']['mean']:.2f}% SR={aggregate['reliability_pct']['mean']:.2f}% CVR={aggregate['cvr']['mean']:.2f}%")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "results.json", {"profile": args.profile, "seeds": seeds, "methods": all_results})
    metrics = ("tcr", "latency_ms", "energy_mj", "reliability_pct", "resource_utilization_pct", "decision_latency_ms", "cvr", "expected_cost", "mean_replicas")
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", *[f"{metric}_mean" for metric in metrics], *[f"{metric}_std" for metric in metrics]])
        for method, result in all_results.items():
            aggregate = result["aggregate"]
            writer.writerow([method, *[aggregate[metric]["mean"] for metric in metrics], *[aggregate[metric]["std"] for metric in metrics]])
    print(f"saved reproduction results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
