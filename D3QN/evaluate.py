from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from drl_ra.baselines import POLICIES
from drl_ra.config import apply_overrides, load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_agent, evaluate_callable, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned and heuristic SAGIN offloading policies.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--method", choices=tuple(POLICIES) + ("checkpoint",), default="checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.method == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for method=checkpoint")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        method = metadata.get("method", "drl-ra")
        if "config" in metadata:
            config = deepcopy(metadata["config"])
        config = apply_overrides(config, args.set)
        probe = SAGINEnv(config, seed=args.seeds[0])
        agent = build_agent(method, probe, config, int(metadata.get("seed", 0)), args.device)
        agent.load(args.checkpoint)

        def policy(state, env, rng):
            mask = [candidate.available for candidate in env.candidates]
            return agent.act(state, mask, epsilon=0.0)
    else:
        config = apply_overrides(config, args.set)
        baseline = POLICIES[args.method]

        def policy(state, env, rng):
            mask = [candidate.available for candidate in env.candidates]
            return baseline(mask, env.candidates, env.current_task, rng)
    rows, aggregate = evaluate_callable(config, policy, args.seeds)
    payload = {"method": args.method, "checkpoint": args.checkpoint, "runs": rows, "aggregate": aggregate}
    write_json(args.output, payload)
    print(f"method={args.method}")
    for metric, stats in aggregate.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"saved evaluation to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
