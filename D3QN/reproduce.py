from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from drl_ra.baselines import POLICIES
from drl_ra.config import apply_overrides, load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import evaluate_callable, evaluate_hierarchical_agent, train_agent, train_hierarchical_agent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible paper comparison.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--profile", choices=("smoke", "quick", "paper"), default="quick")
    parser.add_argument("--methods", nargs="+", default=["random", "greedy-nearest", "greedy-reliability", "dqn", "d3qn", "drl-ra", "d3qn-ppo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/reproduction")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-seed training and evaluation progress.")
    return parser.parse_args()


def profile_config(config: dict, profile: str) -> tuple[dict, list[int]]:
    result = deepcopy(config)
    if profile == "smoke":
        result["training"]["episodes"] = 1
        result["environment"]["episode_steps"] = 70
        result["training"]["batch_size"] = 16
        result["training"]["evaluation_steps"] = 70
        return result, [0]
    if profile == "quick":
        result["training"]["episodes"] = 30
        result["environment"]["episode_steps"] = 200
        result["training"]["evaluation_steps"] = 1000
        return result, [0, 1, 2]
    return result, list(range(10))


def main() -> None:
    args = parse_args()
    config, default_seeds = profile_config(load_config(args.config), args.profile)
    config = apply_overrides(config, args.set)
    seeds = args.seeds if args.seeds is not None else default_seeds
    output_dir = Path(args.output_dir)
    all_results: dict[str, dict] = {}
    for method_index, method in enumerate(args.methods, start=1):
        if not args.quiet:
            print(f"\n[{method_index}/{len(args.methods)}] method={method}", flush=True)
        method_config = deepcopy(config)
        evaluation_config = deepcopy(method_config)
        evaluation_config["environment"]["episode_steps"] = int(method_config["training"]["evaluation_steps"])
        if method in POLICIES:
            baseline = POLICIES[method]

            def policy(state, env: SAGINEnv, rng, selected=baseline):
                mask = [candidate.available for candidate in env.candidates]
                return selected(mask, env.candidates, env.current_task, rng)

            if not args.quiet:
                print(f"  evaluating baseline on seeds={seeds}", flush=True)
            rows, aggregate = evaluate_callable(evaluation_config, policy, seeds)
        else:
            rows = []
            for seed_index, seed in enumerate(seeds, start=1):
                if not args.quiet:
                    print(f"  [{seed_index}/{len(seeds)}] training seed={seed}", flush=True)
                train_config = deepcopy(method_config)
                if method == "d3qn-ppo":
                    agent, history = train_hierarchical_agent(train_config, seed, device=args.device, progress=not args.quiet)
                else:
                    agent, history = train_agent(train_config, method, seed, device=args.device, progress=not args.quiet)
                checkpoint = output_dir / "checkpoints" / f"{method}_seed{seed}.pt"
                agent.save(checkpoint, metadata={"method": method, "seed": seed, "config": train_config})
                if method == "d3qn-ppo":
                    agent.save_components(
                        output_dir / "checkpoints" / f"{method}_seed{seed}",
                        metadata={"method": method, "seed": seed, "config": train_config},
                    )
                if not args.quiet:
                    print(f"  saved checkpoint to {checkpoint}", flush=True)

                evaluation_config = deepcopy(train_config)
                evaluation_config["environment"]["episode_steps"] = int(train_config["training"]["evaluation_steps"])
                evaluation_seed = 10_000 + seed
                if not args.quiet:
                    print(f"  evaluating seed={evaluation_seed}", flush=True)
                if method == "d3qn-ppo":
                    evaluation, _ = evaluate_hierarchical_agent(evaluation_config, agent, [evaluation_seed])
                else:
                    def policy(state, env: SAGINEnv, rng, selected=agent):
                        return selected.act(state, [candidate.available for candidate in env.candidates], epsilon=0.0)

                    evaluation, _ = evaluate_callable(evaluation_config, policy, [evaluation_seed])
                rows.extend(evaluation)
                if not args.quiet:
                    result = evaluation[0]
                    print(
                        f"  evaluation complete: TCR={result['tcr']:.2f}% "
                        f"SR={result['reliability_pct']:.2f}% CVR={result['cvr']:.2f}%",
                        flush=True,
                    )
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
    metrics = ("tcr", "deadline_satisfaction_pct", "latency_ms", "energy_mj", "reliability_pct", "resource_utilization_pct", "decision_latency_ms", "cvr", "expected_cost", "mean_replicas")
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", *[f"{metric}_mean" for metric in metrics], *[f"{metric}_std" for metric in metrics]])
        for method, result in all_results.items():
            aggregate = result["aggregate"]
            writer.writerow([method, *[aggregate[metric]["mean"] for metric in metrics], *[aggregate[metric]["std"] for metric in metrics]])
    print(f"saved reproduction results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
