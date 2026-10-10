from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from drl_ra.agent import ReplicaD3QNAgent
from drl_ra.config import apply_overrides, load_config


def load_teacher_data(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1 or data.get("teacher_method") != "drl-ra":
        raise ValueError("unsupported teacher data or teacher is not original DRL-RA")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("teacher data contains no samples")
    states = np.asarray([row["state"] for row in samples], dtype=np.float32)
    masks = np.asarray([row["action_mask"] for row in samples], dtype=bool)
    actions = np.asarray([row["teacher_action"] for row in samples], dtype=np.int64)
    if states.ndim != 2 or masks.shape != (len(samples), int(data["action_dim"])):
        raise ValueError("teacher data dimensions are inconsistent")
    if states.shape[1] != int(data["state_dim"]):
        raise ValueError("teacher data state dimension is inconsistent")
    return {"metadata": data, "states": states, "masks": masks, "actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the Replica D3QN from DRL-RA teacher demonstrations.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(args.torch_threads)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another output or use --overwrite")

    data = load_teacher_data(args.data)
    config = apply_overrides(load_config(args.config), args.set)
    config = deepcopy(config)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate must be positive")
        config["learned_replica"]["training"]["learning_rate"] = args.learning_rate
    replica = ReplicaD3QNAgent(
        int(data["metadata"]["state_dim"]),
        int(data["metadata"]["action_dim"]),
        config,
        args.seed,
        device=args.device,
    )
    losses = replica.behavior_clone(
        data["states"], data["masks"], data["actions"], args.epochs, args.batch_size
    )
    metadata = {
        "artifact": "replica_behavior_cloning",
        "teacher_data": str(args.data),
        "teacher_data_trace_sha256": data["metadata"].get("trace_sha256"),
        "teacher_data_sha256": data["metadata"].get("teacher_checkpoint_sha256"),
        "state_dim": replica.state_dim,
        "action_dim": replica.action_dim,
        "samples": int(len(data["states"])),
        "epochs": int(args.epochs),
        "config": config,
    }
    replica.save(output, metadata=metadata)
    print(json.dumps({
        "output": str(output),
        "samples": len(data["states"]),
        "epochs": args.epochs,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }, indent=2))


if __name__ == "__main__":
    main()
