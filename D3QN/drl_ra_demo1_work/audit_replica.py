from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl_ra.config import apply_overrides, load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import (
    build_agent,
    build_learned_replica_agent,
    seed_everything,
)


AUDIT_FIELDS = (
    "num_feasible_candidates",
    "num_feasible_ground",
    "num_feasible_uav",
    "num_feasible_leo",
    "max_reliability_all",
    "min_replicas_to_requirement",
    "intrinsically_infeasible",
    "max_reliability_given_primary",
    "primary_bottleneck",
    "replica_policy_failure",
    "best_reward_n1",
    "best_reward_n2",
    "best_reward_n3",
    "reward_optimal_replica_count",
    "active_stop",
    "max_replica_stop",
    "no_feasible_candidate_stop",
    "capacity_block",
    "visibility_block",
    "battery_block",
    "coverage_block",
)


def reliability_bucket(required: float) -> str:
    if required < 0.90:
        return "<0.90"
    if required < 0.95:
        return "0.90-0.95"
    if required < 0.98:
        return "0.95-0.98"
    return ">=0.98"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = max(len(rows), 1)
    result = {
        "tasks": float(len(rows)),
        "tcr": 100.0 * sum(float(row["completed"]) for row in rows) / count,
        "cvr": 100.0 * sum(float(row["violation"]) for row in rows) / count,
        "mean_replicas": sum(float(row["replicas"]) for row in rows) / count,
        "reliability_shortfall": sum(float(row["reliability_shortfall"]) for row in rows) / count,
        "reliability_excess": sum(float(row["reliability_excess"]) for row in rows) / count,
    }
    for field in AUDIT_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            result[field] = float(np.mean(values))
    return result


def load_checkpoint(path: str, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if "config" not in metadata:
        raise ValueError("checkpoint does not contain its training configuration")
    return payload, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feasibility, primary bottlenecks, and reward oracle behavior.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/replica_audit.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    _, metadata = load_checkpoint(args.checkpoint, args.device)
    config = apply_overrides(deepcopy(metadata["config"]), args.set)
    config["environment"]["episode_steps"] = int(args.steps)
    method = str(metadata.get("method", "drl-ra"))
    seed_everything(args.seed)
    env = SAGINEnv(config, seed=args.seed)
    if method == "drl-ra-learned-replica":
        policy = build_learned_replica_agent(env, config, int(metadata.get("seed", 0)), args.device)
        policy.load(args.checkpoint)
        learned = True
    else:
        policy = build_agent(method, env, config, int(metadata.get("seed", 0)), args.device)
        policy.load(args.checkpoint)
        learned = False

    state, reset_info = env.reset(seed=args.seed)
    mask = reset_info["action_mask"]
    records: list[dict[str, Any]] = []
    while True:
        if learned:
            _, selected, _ = policy.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
            state, _, terminated, truncated, info = env.step_with_replicas(
                selected, stop_reason=policy.last_stop_reason
            )
        else:
            action = policy.act(state, mask, epsilon=0.0)
            state, _, terminated, truncated, info = env.step(action)
        records.append({key: info.get(key) for key in (*AUDIT_FIELDS, "task_kind", "required_reliability", "replicas", "reliability", "completed", "violation", "reliability_shortfall", "reliability_excess")})
        mask = info["action_mask"]
        if terminated or truncated:
            break

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[f"reliability:{reliability_bucket(float(row['required_reliability']))}"].append(row)
        groups[f"task_kind:{row['task_kind']}"] .append(row)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "method": method,
        "seed": args.seed,
        "steps": args.steps,
        "summary": {**env.summary(), "audit": aggregate(records)},
        "groups": {name: aggregate(rows) for name, rows in sorted(groups.items())},
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
