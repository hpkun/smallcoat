from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import BaseStation
from src import LEOSatellite
from src import UAV
from src.debug_tools import summarize_execution_record
from train import build_medium_env
from train import build_small_scale_env
from train import build_training_env


def _build_env(env_name: str):
    if env_name == "small":
        return build_small_scale_env()
    if env_name == "medium":
        return build_medium_env()
    if env_name == "training":
        return build_training_env()
    raise ValueError(f"Unsupported env: {env_name}")


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0 (0.00%)"
    return f"{numerator}/{denominator} ({100.0 * float(numerator) / float(denominator):.2f}%)"


def _target_bucket(target_node_type: str) -> str:
    if target_node_type == "uav":
        return "uav_local"
    if target_node_type == "bs":
        return "bs"
    if target_node_type == "leo":
        return "leo"
    return target_node_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize task lifecycle statistics.")
    parser.add_argument(
        "--env",
        choices=["small", "medium", "training"],
        default="training",
        help="Which environment preset to inspect.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="How many environment slots to roll forward for summary statistics.",
    )
    parser.add_argument(
        "--move-uavs",
        action="store_true",
        help="Whether to move UAVs before each slot step.",
    )
    args = parser.parse_args()

    env = _build_env(args.env)
    slot_length_s = env.base_env.simulation_config.slot_length_s
    delay_lambda = env.base_env.task_generator.task_model_config.delay_sensitivity_lambda

    all_records = []
    current_time_s = 0.0

    for _ in range(max(args.steps, 1)):
        records = env.base_env.step(
            slot_length_s=slot_length_s,
            current_time_s=current_time_s,
            delay_sensitivity_lambda=delay_lambda,
            move_uavs=args.move_uavs,
        )
        all_records.extend(records)
        current_time_s += slot_length_s

    total_records = len(all_records)
    completed_records = [record for record in all_records if record.completed_before_deadline]
    timeout_records = [record for record in all_records if not record.completed_before_deadline]

    by_target_total = Counter(_target_bucket(record.target_node_type) for record in all_records)
    by_target_completed = Counter(_target_bucket(record.target_node_type) for record in completed_records)
    timeout_bottlenecks = Counter(summarize_execution_record(record).dominant_stage for record in timeout_records)

    deadline_ok_count = sum(
        1 for record in all_records if record.constraint_check is not None and record.constraint_check.satisfies_deadline
    )
    capacity_ok_count = sum(
        1 for record in all_records if record.constraint_check is not None and record.constraint_check.satisfies_capacity
    )
    feasible_count = sum(
        1 for record in all_records if record.constraint_check is not None and record.constraint_check.feasible
    )
    print("task_lifecycle_summary")
    print(f"env={args.env}")
    print(f"steps={args.steps}")
    print(f"move_uavs={args.move_uavs}")
    print(f"total_tasks={total_records}")
    print(f"overall_completion={_format_ratio(len(completed_records), total_records)}")
    print(f"overall_timeout={_format_ratio(len(timeout_records), total_records)}")
    print(f"deadline_ok={_format_ratio(deadline_ok_count, total_records)}")
    print(f"capacity_ok={_format_ratio(capacity_ok_count, total_records)}")
    print(f"feasible={_format_ratio(feasible_count, total_records)}")
    print("")
    print("completion_by_target")
    for bucket in ["uav_local", "bs", "leo"]:
        total = by_target_total.get(bucket, 0)
        completed = by_target_completed.get(bucket, 0)
        print(f"{bucket}={_format_ratio(completed, total)}")
    print("")
    print("timeout_bottleneck_share")
    for bucket in ["transmission", "queue", "compute"]:
        count = timeout_bottlenecks.get(bucket, 0)
        print(f"{bucket}={_format_ratio(count, len(timeout_records))}")


if __name__ == "__main__":
    main()
