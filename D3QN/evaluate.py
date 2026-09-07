from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from drl_ra.baselines import POLICIES
from drl_ra.config import apply_overrides, load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_agent, build_hierarchical_agent, evaluate_callable, evaluate_hierarchical_agent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned and heuristic SAGIN offloading policies.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--method", choices=tuple(POLICIES) + ("checkpoint",), default="checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ground-checkpoint", help="Standalone Ground D3QN component checkpoint.")
    parser.add_argument("--ppo-checkpoint", help="Standalone NTL PPO component checkpoint.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hierarchical_agent = None
    if args.method == "checkpoint":
        component_mode = bool(args.ground_checkpoint or args.ppo_checkpoint)
        if component_mode and not (args.ground_checkpoint and args.ppo_checkpoint):
            raise SystemExit("--ground-checkpoint and --ppo-checkpoint must be supplied together")
        if not args.checkpoint and not component_mode:
            raise SystemExit("--checkpoint or both component checkpoints are required for method=checkpoint")
        metadata_source = args.ground_checkpoint if component_mode else args.checkpoint
        payload = torch.load(metadata_source, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        method = "d3qn-ppo" if component_mode else metadata.get("method", "drl-ra")
        if "config" in metadata:
            config = deepcopy(metadata["config"])
        config = apply_overrides(config, args.set)
        probe = SAGINEnv(config, seed=args.seeds[0])
        if method == "d3qn-ppo":
            hierarchical_agent = build_hierarchical_agent(probe, config, int(metadata.get("seed", 0)), args.device)
            if component_mode:
                hierarchical_agent.load_components(args.ground_checkpoint, args.ppo_checkpoint)
            else:
                hierarchical_agent.load(args.checkpoint)
        else:
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
    if not any(item.startswith("environment.episode_steps=") for item in args.set):
        config["environment"]["episode_steps"] = int(config["training"].get("evaluation_steps", config["environment"]["episode_steps"]))
    if hierarchical_agent is not None:
        rows, aggregate = evaluate_hierarchical_agent(config, hierarchical_agent, args.seeds)
    else:
        rows, aggregate = evaluate_callable(config, policy, args.seeds)
    payload = {"method": args.method, "checkpoint": args.checkpoint, "ground_checkpoint": args.ground_checkpoint, "ppo_checkpoint": args.ppo_checkpoint, "runs": rows, "aggregate": aggregate}
    write_json(args.output, payload)
    print(f"method={args.method}")
    for metric, stats in aggregate.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"saved evaluation to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
