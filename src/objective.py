from __future__ import annotations

from dataclasses import dataclass

from .entities import ExecutionRecord


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """长期平均收益目标的统计分解。"""

    total_profit: float
    average_profit_per_record: float
    completion_rate: float
    average_formula7_delay_s: float
    total_energy_j: float
    average_energy_per_record_j: float


def compute_equation_8_objective(records: list[ExecutionRecord]) -> ObjectiveBreakdown:
    """
    近似统计论文式 (8) 的目标量。

    论文目标是长期平均收益：
        max lim_{T->inf} 1/T * sum alpha_{k,j}(t) * G_k(t)

    当前实现按一批执行记录做经验统计近似。
    """

    if not records:
        return ObjectiveBreakdown(
            total_profit=0.0,
            average_profit_per_record=0.0,
            completion_rate=0.0,
            average_formula7_delay_s=0.0,
            total_energy_j=0.0,
            average_energy_per_record_j=0.0,
        )

    total_profit = float(sum(record.realized_profit for record in records))
    average_profit_per_record = total_profit / len(records)
    completion_rate = sum(1.0 if record.completed_before_deadline else 0.0 for record in records) / len(records)
    average_formula7_delay_s = sum(record.total_delay_s for record in records) / len(records)
    total_energy_j = float(sum(record.total_energy_j for record in records))
    average_energy_per_record_j = total_energy_j / len(records)
    return ObjectiveBreakdown(
        total_profit=total_profit,
        average_profit_per_record=float(average_profit_per_record),
        completion_rate=float(completion_rate),
        average_formula7_delay_s=float(average_formula7_delay_s),
        total_energy_j=total_energy_j,
        average_energy_per_record_j=float(average_energy_per_record_j),
    )
