from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch

from drl_ra.config import apply_overrides, load_config
from drl_ra.experiment import train_agent, train_hierarchical_agent, train_learned_replica_agent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper-aligned DRL-RA/D3QN agent.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--method", choices=("drl-ra", "d3qn", "dqn", "no-dueling", "no-double", "no-redundancy", "d3qn-ppo", "drl-ra-learned-replica"), default="drl-ra")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU; explicit devices such as cuda:0 are also supported.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--torch-threads", type=int, help="Number of PyTorch CPU threads (use 1 for this small network).")
    parser.add_argument("--metrics-only", action="store_true", help="Skip optional feasibility and reward-oracle audits during training.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}", flush=True)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    config = apply_overrides(load_config(args.config), args.set)
    seed = int(config["seed"] if args.seed is None else args.seed)
    if args.method == "d3qn-ppo":
        agent, history = train_hierarchical_agent(deepcopy(config), seed, device=args.device)
    elif args.method == "drl-ra-learned-replica":
        agent, history = train_learned_replica_agent(deepcopy(config), seed, device=args.device, metrics_only=args.metrics_only)
    else:
        agent, history = train_agent(deepcopy(config), args.method, seed, device=args.device, metrics_only=args.metrics_only)
    run_dir = Path(args.output_dir) / f"{args.method}_seed{seed}"
    clean_history = [{key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()} for row in history]
    agent.save(run_dir / "model.pt", metadata={"method": args.method, "seed": seed, "config": config})
    if args.method == "d3qn-ppo":
        agent.save_components(run_dir, metadata={"method": args.method, "seed": seed, "config": config})
    write_json(run_dir / "history.json", clean_history)
    print(f"saved checkpoint and history to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
