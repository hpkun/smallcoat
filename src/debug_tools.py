from __future__ import annotations

from dataclasses import dataclass

from .entities import ExecutionRecord


@dataclass(frozen=True)
class TaskLifecycleSummary:
    """任务生命周期调试摘要。"""

    task_id: str
    ingress_uav_id: str
    decision_uav_id: str
    target_node_id: str
    target_node_type: str
    transmission_delay_s: float
    queue_delay_s: float
    compute_delay_s: float
    communication_delay_s: float
    total_delay_s: float
    actual_finish_delay_s: float
    completed_before_deadline: bool
    dominant_stage: str


def summarize_execution_record(record: ExecutionRecord) -> TaskLifecycleSummary:
    """
    将执行记录压缩成便于排障的生命周期摘要。

    这里把通信总时延定义为：
    - 接入传输时延
    - 接入传播时延
    - 回传传输时延
    - 回传传播时延
    """

    transmission_delay_s = (
        record.ingress_transmission_delay_s
        + record.ingress_propagation_delay_s
        + record.backhaul_transmission_delay_s
        + record.backhaul_propagation_delay_s
    )

    dominant_candidates = {
        "transmission": transmission_delay_s,
        "queue": record.queue_delay_s,
        "compute": record.compute_delay_s,
    }
    dominant_stage = max(dominant_candidates, key=dominant_candidates.get)

    return TaskLifecycleSummary(
        task_id=record.task_id,
        ingress_uav_id=record.ingress_uav_id,
        decision_uav_id=record.decision_uav_id,
        target_node_id=record.target_node_id,
        target_node_type=record.target_node_type,
        transmission_delay_s=float(transmission_delay_s),
        queue_delay_s=float(record.queue_delay_s),
        compute_delay_s=float(record.compute_delay_s),
        communication_delay_s=float(record.communication_delay_s),
        total_delay_s=float(record.total_delay_s),
        actual_finish_delay_s=float(record.actual_finish_delay_s),
        completed_before_deadline=bool(record.completed_before_deadline),
        dominant_stage=dominant_stage,
    )


def format_execution_record_debug(record: ExecutionRecord) -> str:
    """格式化单个任务的生命周期调试输出。"""

    summary = summarize_execution_record(record)
    status = "on-time" if summary.completed_before_deadline else "timeout"
    return (
        f"{summary.task_id} | ingress={summary.ingress_uav_id} | "
        f"decision={summary.decision_uav_id} | target={summary.target_node_id}({summary.target_node_type}) | "
        f"tx={summary.transmission_delay_s:.6f}s | "
        f"queue={summary.queue_delay_s:.6f}s | "
        f"compute={summary.compute_delay_s:.6f}s | "
        f"comm={summary.communication_delay_s:.6f}s | "
        f"formula7_total={summary.total_delay_s:.6f}s | "
        f"actual_finish={summary.actual_finish_delay_s:.6f}s | "
        f"status={status} | bottleneck={summary.dominant_stage}"
    )


def format_records_debug_report(records: list[ExecutionRecord]) -> str:
    """格式化一批任务的生命周期调试报告。"""

    if not records:
        return "no tasks generated in this step"

    lines = ["task_lifecycle_debug_report"]
    for record in records:
        lines.append(format_execution_record_debug(record))
    return "\n".join(lines)
