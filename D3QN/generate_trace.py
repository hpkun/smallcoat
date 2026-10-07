from __future__ import annotations

import argparse
import json
from copy import deepcopy

from drl_ra.config import apply_overrides, load_config
from drl_ra.trace import generate_trace, input_sequence_digest, save_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save fixed tasks, arrivals, channels, and node outcomes.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", default=None)
    source.add_argument("--checkpoint", help="Use this checkpoint's environment and reward configuration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", help="Defaults to outputs/traces/tasks{steps}_seed{seed}.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or f"outputs/traces/tasks{args.steps}_seed{args.seed}.json"
    if args.checkpoint:
        from audit_replica import load_checkpoint

        _, metadata = load_checkpoint(args.checkpoint, "cpu")
        config = deepcopy(metadata["config"])
    else:
        config = load_config(args.config or "configs/paper.yaml")
    trace = generate_trace(apply_overrides(config, args.set), args.seed, args.steps)
    save_trace(trace, output, overwrite=args.overwrite)
    print(json.dumps({
        "output": output, "steps": trace["steps"], "seed": trace["seed"],
        "trace_sha256": trace["sha256"], "input_sequence_sha256": input_sequence_digest(trace["entries"]),
    }, indent=2))


if __name__ == "__main__":
    main()
