from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

from drl_ra.config import apply_overrides, load_config
from drl_ra.experiment import train_agent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper-aligned DRL-RA/D3QN agent.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--method", choices=("drl-ra", "d3qn", "dqn", "no-dueling", "no-double", "no-redundancy"), default="drl-ra")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    seed = int(config["seed"] if args.seed is None else args.seed)
    agent, history = train_agent(deepcopy(config), args.method, seed, device=args.device)
    run_dir = Path(args.output_dir) / f"{args.method}_seed{seed}"
    clean_history = [{key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()} for row in history]
    agent.save(run_dir / "model.pt", metadata={"method": args.method, "seed": seed, "config": config})
    write_json(run_dir / "history.json", clean_history)
    print(f"saved checkpoint and history to {run_dir.resolve()}")


if __name__ == "__main__":
    main()

