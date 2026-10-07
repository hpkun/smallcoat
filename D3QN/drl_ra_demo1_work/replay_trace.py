from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from audit_replica import AUDIT_FIELDS, aggregate, load_checkpoint, reliability_bucket
from drl_ra.experiment import build_agent, build_learned_replica_agent, method_options, seed_everything
from drl_ra.trace import TraceEnv, digest, evaluation_config, input_sequence_digest, load_trace


METHODS = ("d3qn", "drl-ra", "drl-ra-learned-replica")
STOP_FIELDS = ("active_stop", "max_replica_stop", "no_feasible_candidate_stop")
FAILURE_FIELDS = ("intrinsically_infeasible", "primary_bottleneck", "replica_policy_failure")
AUDIT_NAMES = {
    "d3qn": "audit_d3qn.json",
    "drl-ra": "audit_drl_ra.json",
    "drl-ra-learned-replica": "audit_learned_replica.json",
}


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize_records(rows: list[dict[str, Any]], learned: bool) -> dict[str, Any]:
    result: dict[str, Any] = aggregate(rows)
    count = len(rows)
    for field in FAILURE_FIELDS:
        result[f"{field}_pct"] = 100.0 * sum(row[field] for row in rows) / count
    result["violation_causes_counts"] = {field: sum(row[field] for row in rows) for field in FAILURE_FIELDS}
    result["min_replicas_to_requirement_counts"] = {
        str(n): sum(row["min_replicas_to_requirement"] == n for row in rows) for n in (-1, 1, 2, 3)
    }
    result["reward_optimal_replica_counts"] = {
        str(n): sum(row["reward_optimal_replica_count"] == n for row in rows) for n in (1, 2, 3)
    }
    result["mean_reward"] = sum(row["reward"] for row in rows) / count
    result["mean_snapshot_reward_regret"] = sum(row["snapshot_reward_regret"] for row in rows) / count
    # Flat baselines do not have the learned replica STOP action.
    result["stop_reasons"] = {
        "applicable": learned,
        "counts": {field: sum(row[field] for row in rows) if learned else None for field in STOP_FIELDS},
        "rates_pct": {field: 100.0 * sum(row[field] for row in rows) / count if learned else None for field in STOP_FIELDS},
    }
    return result


def run_replay(trace: dict[str, Any], checkpoint: str | Path, device: str = "cpu", expected_method: str | None = None, progress: bool = False) -> dict[str, Any]:
    _, metadata = load_checkpoint(str(checkpoint), device)
    method = str(metadata.get("method", ""))
    if method not in METHODS or (expected_method is not None and method != expected_method):
        raise ValueError(f"wrong checkpoint method: expected {expected_method or METHODS}, got {method!r}")
    training_config = deepcopy(metadata["config"])
    config = evaluation_config(training_config, trace)
    config["environment"]["enable_redundancy"] = method_options(method)["redundancy"]
    overrides = {
        section: {
            key: {"training": training_config[section].get(key), "evaluation": value}
            for key, value in config[section].items() if training_config[section].get(key) != value
        } for section in ("environment", "reward")
    }
    seed_everything(int(trace["seed"]))
    env = TraceEnv(config, trace)
    learned = method == "drl-ra-learned-replica"
    policy_seed = int(metadata.get("seed", trace["seed"]))
    policy = (build_learned_replica_agent(env, config, policy_seed, device) if learned
              else build_agent(method, env, config, policy_seed, device))
    policy.load(checkpoint)
    state, reset_info = env.reset()
    mask = reset_info["action_mask"]
    records = []
    with torch.inference_mode():
        for task_id, entry in enumerate(trace["entries"]):
            if env.step_count != task_id or asdict(env.current_task) != entry["task"] or env.current_time_s != entry["arrival_time_s"]:
                raise RuntimeError(f"shared trace input mismatch at task {task_id}")
            if learned:
                _, actions, _ = policy.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
                stop_reason = policy.last_stop_reason
            else:
                primary_action = policy.act(state, mask, epsilon=0.0)
                primary = env.candidates[primary_action]
                if not primary.available:
                    primary = env.candidates[0]
                actions = [item.action for item in env._replica_plan(primary)]
                stop_reason = "not_applicable"
            selected_reward = env._replica_set_reward([env.candidates[action] for action in actions])
            if learned:
                state, reward, terminated, truncated, info = env.step_with_replicas(actions, stop_reason=stop_reason)
            else:
                state, reward, terminated, truncated, info = env.step(primary_action)
            if info["replica_actions"] != actions:
                raise RuntimeError(f"executed replicas differ from audited replicas at task {task_id}")
            row = {key: value for key, value in info.items() if key != "action_mask"}
            row.update(
                task_id=task_id, arrival_time_s=entry["arrival_time_s"], task=entry["task"],
                input_sha256=digest(entry), reward=reward, stop_reason=stop_reason,
                selected_snapshot_reward=selected_reward,
                snapshot_reward_regret=max(row[field] for field in ("best_reward_n1", "best_reward_n2", "best_reward_n3") if row[field] is not None) - selected_reward,
            )
            if any(field not in row for field in AUDIT_FIELDS):
                raise RuntimeError("missing Step 2-5 audit fields")
            if sum(row[field] for field in FAILURE_FIELDS) != row["violation"]:
                raise RuntimeError("violation causes do not partition reliability violations")
            if learned and sum(row[field] for field in STOP_FIELDS) != 1:
                raise RuntimeError("learned replica stop reasons do not partition tasks")
            if bool(terminated or truncated) != (task_id == trace["steps"] - 1):
                raise RuntimeError("trace terminated at the wrong task")
            records.append(row)
            mask = info["action_mask"]
            if progress and (task_id + 1) % 500 == 0:
                print(f"{method}: {task_id + 1}/{trace['steps']} tasks", flush=True)
    actual_sequence = digest([row["input_sha256"] for row in records])
    if actual_sequence != input_sequence_digest(trace["entries"]):
        raise RuntimeError("replay input sequence checksum differs from trace")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[f"reliability:{reliability_bucket(row['required_reliability'])}"].append(row)
        groups[f"task_kind:{row['task_kind']}"].append(row)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
        "method": method, "seed": trace["seed"], "steps": trace["steps"],
        "trace_sha256": trace["sha256"], "input_sequence_sha256": actual_sequence,
        "evaluation_config": trace["evaluation_config"], "evaluation_overrides": overrides,
        "policy_enable_redundancy": config["environment"]["enable_redundancy"],
        "oracle_convention": "Global feasible sets of 1-3 replicas; deterministic snapshot reward uses minimum candidate delay, not expected sampled reward.",
        "summary": {**env.summary(), "audit": summarize_records(records, learned)},
        "groups": {name: summarize_records(rows, learned) for name, rows in sorted(groups.items())},
        "records": records,
    }


def verify_shared_inputs(results: list[dict[str, Any]], trace: dict[str, Any]) -> None:
    expected_hashes = [digest(entry) for entry in trace["entries"]]
    if {result["method"] for result in results} != set(METHODS) or len(results) != len(METHODS):
        raise RuntimeError("comparison requires all three distinct algorithms")
    for result in results:
        if result["trace_sha256"] != trace["sha256"] or result["steps"] != trace["steps"]:
            raise RuntimeError("comparison uses different traces")
        if [row["input_sha256"] for row in result["records"]] != expected_hashes:
            raise RuntimeError("comparison task inputs differ")


def comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# 固定 trace：Step 2–5", "",
        f"三个算法使用同一份 {comparison['steps']} 任务 trace（seed={comparison['seed']}）。",
        f"逐任务输入核对：{comparison['shared_inputs_verified']}。", "",
        f"Trace SHA-256：`{comparison['trace_sha256']}`", "",
        "TCR 为按时且抽样成功的比例；CVR 为组合可靠性未达要求的比例。所有百分比以任务数为分母。", "",
        "| 算法 | TCR % | CVR % | 平均副本数 | 内在不可行 % | Primary 瓶颈 % | 副本策略失败 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in comparison["algorithms"].items():
        summary, audit = result["summary"], result["summary"]["audit"]
        lines.append(f"| {method} | {summary['tcr']:.2f} | {summary['cvr']:.2f} | {summary['mean_replicas']:.3f} | {audit['intrinsically_infeasible_pct']:.2f} | {audit['primary_bottleneck_pct']:.2f} | {audit['replica_policy_failure_pct']:.2f} |")
    lines += ["", "后三项互斥，合计等于 CVR；内在不可行按最多 3 个可行副本计算。", "",
              "| 算法 | 平均可行候选 | Ground | UAV | LEO |", "|---|---:|---:|---:|---:|"]
    for method, result in comparison["algorithms"].items():
        audit = result["summary"]["audit"]
        values = [audit[field] for field in ("num_feasible_candidates", "num_feasible_ground", "num_feasible_uav", "num_feasible_leo")]
        lines.append(f"| {method} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "| 算法 | Reward 最优 n=1 任务数 | n=2 | n=3 | 平均 snapshot reward regret |",
              "|---|---:|---:|---:|---:|"]
    for method, result in comparison["algorithms"].items():
        audit = result["summary"]["audit"]
        counts = audit["reward_optimal_replica_counts"]
        lines.append(f"| {method} | {counts['1']} | {counts['2']} | {counts['3']} | {audit['mean_snapshot_reward_regret']:.6f} |")
    lines += ["", "Reward oracle 沿用现有快照口径：枚举全局可行候选的 1/2/3 副本集合，时延取集合中的最小时延。它不是对成功抽样求期望后的 reward。", "",
              "| 算法 | 主动 STOP % | 达到副本上限 % | 无可行候选 % |", "|---|---:|---:|---:|"]
    for method, result in comparison["algorithms"].items():
        stop = result["summary"]["audit"]["stop_reasons"]
        values = [f"{stop['rates_pct'][field]:.2f}" if stop["applicable"] else "不适用" for field in STOP_FIELDS]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines += ["", "容量、可见窗口、电池和覆盖阻塞分别统计不可用候选，可能同时发生，不作为互斥的任务停止原因。", "",
              "固定的是任务、到达时间、初始拓扑、信道随机量和按任务/节点索引的成功随机量。资源与队列由各算法动作分别演化，因此候选可行性及 oracle 可以不同。", "",
              "逐任务 Step 2–5 字段、分组结果及检查点哈希见三个 audit JSON。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay all three algorithms on one saved trace and audit Steps 2-5.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--d3qn-checkpoint", default="outputs/screen_seed42/d3qn_seed42/model.pt")
    parser.add_argument("--drl-ra-checkpoint", default="outputs/screen_seed42/drl-ra_seed42/model.pt")
    parser.add_argument("--learned-replica-checkpoint", default="outputs/screen_seed42/drl-ra-learned-replica_seed42/model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", help="Defaults to outputs/fixed_trace_seed{trace seed}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    trace = load_trace(args.trace)
    output = Path(args.output_dir or f"outputs/fixed_trace_seed{trace['seed']}")
    filenames = [*AUDIT_NAMES.values(), "comparison.json", "comparison.md", "behavior_diagnostics.json"]
    for name in filenames:
        if (output / name).exists() and not args.overwrite:
            raise FileExistsError(f"{output / name} exists; choose another output directory or --overwrite")
    checkpoints = (args.d3qn_checkpoint, args.drl_ra_checkpoint, args.learned_replica_checkpoint)
    # Validate every checkpoint before starting a potentially lengthy replay.
    for method, checkpoint in zip(METHODS, checkpoints):
        _, metadata = load_checkpoint(checkpoint, args.device)
        if metadata.get("method") != method:
            raise ValueError(f"{checkpoint}: expected method {method}")
        evaluation_config(metadata["config"], trace)
    results = [run_replay(trace, checkpoint, args.device, method, progress=True) for method, checkpoint in zip(METHODS, checkpoints)]
    verify_shared_inputs(results, trace)
    comparison = {
        "trace": str(Path(args.trace)), "trace_sha256": trace["sha256"],
        "input_sequence_sha256": input_sequence_digest(trace["entries"]),
        "seed": trace["seed"], "steps": trace["steps"], "shared_inputs_verified": True,
        "algorithms": {result["method"]: {key: value for key, value in result.items() if key != "records"} for result in results},
    }
    output.mkdir(parents=True, exist_ok=True)

    def write_json(name: str, value: Any) -> None:
        with (output / name).open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

    for result in results:
        write_json(AUDIT_NAMES[result["method"]], result)
    write_json("comparison.json", comparison)
    learned = results[-1]
    stop_rates = learned["summary"]["audit"]["stop_reasons"]["rates_pct"]
    write_json("behavior_diagnostics.json", {
        **{key: value for key, value in learned.items() if key != "records"},
        "active_stop_rate": stop_rates["active_stop"],
        "max_replica_stop_rate": stop_rates["max_replica_stop"],
        "no_feasible_candidate_stop_rate": stop_rates["no_feasible_candidate_stop"],
    })
    with (output / "comparison.md").open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        stream.write(comparison_markdown(comparison))
    print(json.dumps({
        "output_dir": str(output), "shared_inputs_verified": True,
        "summary": {result["method"]: result["summary"] for result in results},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
