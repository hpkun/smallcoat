from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from drl_ra.config import apply_overrides
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_learned_replica_agent, seed_everything


def bucket(required: float) -> str:
    if required < 0.90:
        return "<0.90"
    if required < 0.95:
        return "0.90-0.95"
    if required < 0.98:
        return "0.95-0.98"
    return ">=0.98"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose learned-replica behavior by reliability and task kind.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/behavior_diagnostics.json")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    config = apply_overrides(metadata["config"], args.set)
    config["environment"]["episode_steps"] = int(args.steps)
    seed_everything(args.seed)
    env = SAGINEnv(config, seed=args.seed)
    agent = build_learned_replica_agent(env, config, int(metadata.get("seed", args.seed)), "cpu")
    agent.load(args.checkpoint)
    env.reset(seed=args.seed)

    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    active_stops = max_replica_stops = no_feasible_candidate_stops = 0
    while env.step_count < args.steps:
        required = float(env.current_task.reliability_required)
        kind = str(env.current_task.kind)
        _, selected, transitions = agent.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
        active_stops += int(agent.last_stop_reason == "active_stop")
        max_replica_stops += int(agent.last_stop_reason == "max_replica_stop")
        no_feasible_candidate_stops += int(agent.last_stop_reason == "no_feasible_candidate_stop")
        _, _, terminated, truncated, info = env.step_with_replicas(
            selected, stop_reason=agent.last_stop_reason
        )
        row = {
            "replicas": float(info["replicas"]),
            "reliability": float(info["reliability"]),
            "required_reliability": required,
            "cvr": float(info["violation"]),
            "reliability_shortfall": float(info["reliability_shortfall"]),
            "reliability_excess": float(info["reliability_excess"]),
            "intrinsically_infeasible": float(info["intrinsically_infeasible"]),
            "primary_bottleneck": float(info["primary_bottleneck"]),
            "replica_policy_failure": float(info["replica_policy_failure"]),
        }
        groups[f"reliability:{bucket(required)}"].append(row)
        groups[f"task_kind:{kind}"].append(row)
        if terminated or truncated:
            break

    def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        count = max(len(rows), 1)
        return {
            "tasks": float(len(rows)),
            "mean_replicas": sum(row["replicas"] for row in rows) / count,
            "cvr": 100.0 * sum(row["cvr"] for row in rows) / count,
            "mean_reliability": sum(row["reliability"] for row in rows) / count,
            "reliability_shortfall": sum(row["reliability_shortfall"] for row in rows) / count,
            "reliability_excess": sum(row["reliability_excess"] for row in rows) / count,
        }

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "seed": args.seed,
        "steps": args.steps,
        "active_stop_rate": 100.0 * active_stops / max(env.step_count, 1),
        "max_replica_stop_rate": 100.0 * max_replica_stops / max(env.step_count, 1),
        "no_feasible_candidate_stop_rate": 100.0 * no_feasible_candidate_stops / max(env.step_count, 1),
        "groups": {name: aggregate(rows) for name, rows in sorted(groups.items())},
        "summary": env.summary(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
