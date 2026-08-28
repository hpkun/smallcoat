from __future__ import annotations

from dataclasses import dataclass

from .entities import TaskInstance


@dataclass(frozen=True)
class ConstraintCheckResult:
    """单个候选卸载动作的约束检查结果。"""

    satisfies_unique_offload: bool
    satisfies_deadline: bool
    satisfies_binary_action: bool
    satisfies_capacity: bool

    @property
    def feasible(self) -> bool:
        """是否同时满足公式 (9)-(12) 的全部约束。"""
        return (
            self.satisfies_unique_offload
            and self.satisfies_deadline
            and self.satisfies_binary_action
            and self.satisfies_capacity
        )


@dataclass(frozen=True)
class CapacitySnapshot:
    """
    公式 (12) 在当前时隙 t、目标节点 j 上的容量快照。

    论文原式：
        sum alpha_{k,j}(t) * phi_k * gamma_k <= C_j(t)

    因此这里直接保存：
    - assigned_load_cycles: 当前时隙内分配到节点 j 的总卸载计算负载
    - capacity_limit_cycles: 节点 j 在当前时隙 t 的计算容量上限
    """

    target_node_id: str
    assigned_load_cycles: float
    capacity_limit_cycles: float

    @property
    def remaining_capacity_cycles(self) -> float:
        """当前时隙剩余可用容量。"""
        return self.capacity_limit_cycles - self.assigned_load_cycles


def check_equation_9_unique_offload(num_selected_targets: int) -> bool:
    """公式 (9): 每个任务只能卸载到一个目标。"""
    return num_selected_targets == 1


def check_equation_10_deadline(total_delay_s: float, task_instance: TaskInstance) -> bool:
    """公式 (10): 总时延不超过任务容忍时延。"""
    return total_delay_s <= task_instance.task.tolerable_latency_s


def check_equation_11_binary_action(target_selected: bool) -> bool:
    """公式 (11): 卸载变量 alpha_{k,j} 是 0-1 变量。"""
    return bool(target_selected)


def check_equation_12_capacity(capacity_snapshot: CapacitySnapshot) -> bool:
    """
    公式 (12): 当前时隙内，目标节点 j 的总卸载负载不能超过其容量。

    也就是直接检查：
        assigned_load_cycles <= capacity_limit_cycles
    """

    return capacity_snapshot.assigned_load_cycles <= capacity_snapshot.capacity_limit_cycles
