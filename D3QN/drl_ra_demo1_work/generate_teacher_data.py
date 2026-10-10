from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_replica import load_checkpoint
from drl_ra.experiment import build_agent, method_options, seed_everything
from drl_ra.trace import TraceEnv, digest, evaluation_config, input_sequence_digest, load_trace


SCHEMA_VERSION = 1


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def generate_teacher_data(
    trace: dict[str, Any],
    checkpoint: str | Path,
    device: str = "cpu",
    progress: bool = False,
) -> dict[str, Any]:
    """Expand the original DRL-RA replica heuristic into BC state/action rows."""
    _, metadata = load_checkpoint(str(checkpoint), device)
    if metadata.get("method") != "drl-ra":
        raise ValueError("teacher checkpoint must have method='drl-ra'")
    training_config = deepcopy(metadata["config"])
    config = evaluation_config(training_config, trace)
    config["environment"]["enable_redundancy"] = method_options("drl-ra")["redundancy"]
    seed_everything(int(trace["seed"]))
    env = TraceEnv(config, trace)
    teacher = build_agent("drl-ra", env, config, int(metadata.get("seed", trace["seed"])), device)
    teacher.load(checkpoint)

    state, reset_info = env.reset()
    records: list[dict[str, Any]] = []
    skipped_stop_labels = 0
    max_replicas = int(env.env_cfg["max_replicas"])
    stop_action = env.action_dim
    with torch.inference_mode():
        for task_id, entry in enumerate(trace["entries"]):
            if env.step_count != task_id or env.current_time_s != entry["arrival_time_s"]:
                raise RuntimeError(f"shared trace input mismatch at task {task_id}")
            primary_state = env._state_from_candidates(env.current_task, env.candidates)
            primary_mask = np.asarray([candidate.available for candidate in env.candidates], dtype=bool)
            primary_action = teacher.act(primary_state, primary_mask, epsilon=0.0)
            primary = env.candidates[primary_action]
            if not primary.available:
                primary = env.candidates[0]
                primary_action = primary.action
            plan = env._replica_plan(primary)
            plan_actions = [int(candidate.action) for candidate in plan]
            if not plan_actions or plan_actions[0] != primary_action:
                raise RuntimeError(f"teacher replica plan has an invalid primary at task {task_id}")

            selected = [primary_action]
            task_samples: list[dict[str, Any]] = []
            for backup_action in plan_actions[1:]:
                replica_state = env.learned_redundancy_observation(selected)
                replica_mask = env.learned_redundancy_action_mask(selected)
                if not replica_mask[backup_action]:
                    raise RuntimeError(
                        f"teacher backup action {backup_action} is masked at task {task_id}; "
                        "the teacher plan is incompatible with the current STOP/mask rules"
                    )
                task_samples.append({
                    "task_id": task_id,
                    "input_sha256": digest(entry),
                    "primary_action": primary_action,
                    "selected_actions": list(selected),
                    "state": replica_state.tolist(),
                    "action_mask": replica_mask.tolist(),
                    "teacher_action": int(backup_action),
                })
                selected.append(int(backup_action))

            # A heuristic plan shorter than the cap ends with STOP when the
            # current V2 legality mask permits it. At three replicas the
            # environment submits automatically and has no STOP decision.
            if len(selected) < max_replicas:
                replica_state = env.learned_redundancy_observation(selected)
                replica_mask = env.learned_redundancy_action_mask(selected)
                if replica_mask[stop_action]:
                    task_samples.append({
                        "task_id": task_id,
                        "input_sha256": digest(entry),
                        "primary_action": primary_action,
                        "selected_actions": list(selected),
                        "state": replica_state.tolist(),
                        "action_mask": replica_mask.tolist(),
                        "teacher_action": stop_action,
                    })
                else:
                    # The old heuristic can return a one/two-replica plan
                    # whose STOP is illegal under the current reliability
                    # mask. Such an action cannot be used as CE supervision.
                    skipped_stop_labels += 1
            records.extend(task_samples)
            state, _, terminated, truncated, _ = env.step(primary_action)
            if bool(terminated or truncated) != (task_id == trace["steps"] - 1):
                raise RuntimeError("teacher replay terminated at the wrong task")
            if progress and (task_id + 1) % 500 == 0:
                print(f"teacher: {task_id + 1}/{trace['steps']} tasks", flush=True)

    if not records:
        raise RuntimeError("teacher trace produced no replica BC samples")
    first_state = np.asarray(records[0]["state"], dtype=np.float32)
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher_method": "drl-ra",
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": file_digest(checkpoint),
        "trace_sha256": trace["sha256"],
        "input_sequence_sha256": input_sequence_digest(trace["entries"]),
        "trace_seed": trace["seed"],
        "trace_steps": trace["steps"],
        "state_dim": int(first_state.shape[0]),
        "action_dim": int(len(records[0]["action_mask"])),
        "skipped_illegal_stop_labels": skipped_stop_labels,
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Replica D3QN BC demonstrations from the original DRL-RA teacher.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another output or use --overwrite")
    data = generate_teacher_data(
        load_trace(args.trace), args.teacher_checkpoint, args.device, progress=args.progress
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: data[key] for key in (
        "output", "teacher_method", "trace_sha256", "trace_steps", "state_dim", "action_dim"
    ) if key in data} | {"output": str(output), "samples": len(data["samples"])}, indent=2))


if __name__ == "__main__":
    main()
