from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING
import numpy as np
from .queue_manager import QueueSnapshot
from .queue_manager import TaskQueueManager
from .task_model import Task
from .task_model import compute_computing_delay
from .task_model import compute_task_profit

if TYPE_CHECKING:
    from .constraints import ConstraintCheckResult


@dataclass(frozen=True)
class Position:
    x_m: float
    y_m: float
    altitude_m: float = 0.0

    def distance_to(self, other: "Position") -> float:
        dx = self.x_m - other.x_m
        dy = self.y_m - other.y_m
        dz = self.altitude_m - other.altitude_m
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    ingress_uav_id: str
    created_at_s: float
    task: Task
    workflow_id: str | None = None
    owner_ch_id: str | None = None
    owner_agent_id: str | None = None
    workflow_arrival_time_s: float | None = None
    workflow_deadline_s: float | None = None
    workflow_task_count: int = 1
    workflow_step_index: int = 0
    predecessor_task_ids: tuple[str, ...] = ()
    successor_task_ids: tuple[str, ...] = ()
    workflow_embedding: tuple[float, ...] | None = None

    @property
    def deadline_at_s(self) -> float:
        return self.created_at_s + self.task.tolerable_latency_s


@dataclass(frozen=True)
class ExecutionRecord:
    task_id: str
    ingress_uav_id: str
    decision_uav_id: str
    target_node_id: str
    target_node_type: str
    compute_priority_eta: float
    created_at_s: float
    arrival_at_uav_s: float
    arrival_at_target_s: float
    start_compute_s: float
    finish_time_s: float
    ingress_transmission_delay_s: float
    ingress_propagation_delay_s: float
    backhaul_transmission_delay_s: float
    backhaul_propagation_delay_s: float
    queue_delay_s: float
    compute_delay_s: float
    communication_delay_s: float
    total_delay_s: float
    actual_finish_delay_s: float
    completed_before_deadline: bool
    realized_profit: float
    transmission_energy_j: float = 0.0
    computing_energy_j: float = 0.0
    total_energy_j: float = 0.0
    primary_replica_energy_j: float = 0.0
    backup_replica_energy_j: float = 0.0
    cancellation_time_s: float | None = None
    cancelled_replica_count: int = 0
    cancellation_energy_saved_j: float = 0.0
    replica_cancelled: bool = False
    cancellation_stage: str | None = None
    replica_index: int = 0
    requested_replica_count: int = 1
    admitted_replica_count: int = 1
    capacity_rejected_replica_count: int = 0
    replica_target_node_ids: tuple[str, ...] = ()
    winner_replica_index: int | None = None
    capacity_rejected: bool = False
    energy_rejected: bool = False
    redundancy_requested: bool = False
    is_redundant_task: bool = False
    replica_role: str = "primary"
    primary_target_node_id: str | None = None
    backup_target_node_id: str | None = None
    backup_succeeded: bool = False
    redundancy_succeeded: bool = False
    selected_replica_role: str = "primary"
    execution_reliability: float = 1.0
    transmission_reliability: float = 1.0
    end_to_end_reliability: float = 1.0
    expected_reliability: float = 0.0
    satisfies_reliability: bool = True
    execution_failure_rate: float = 0.0
    transmission_failure_rate: float = 0.0
    execution_failed: bool = False
    transmission_failed: bool = False
    failed_due_to_reliability: bool = False
    workflow_id: str | None = None
    owner_ch_id: str | None = None
    workflow_task_count: int = 1
    workflow_step_index: int = 0
    predecessor_task_ids: tuple[str, ...] = ()
    successor_task_ids: tuple[str, ...] = ()
    workflow_completed: bool = False
    workflow_completion_delay_s: float = 0.0
    workflow_sla_violated: bool = False
    constraint_check: "ConstraintCheckResult | None" = None


@dataclass
class ComputeNode:
    node_id: str
    position: Position
    compute_capacity_cycles_per_s: float
    queue_manager: TaskQueueManager = field(default_factory=TaskQueueManager)
    execution_failure_rate: float = 0.0
    restart_time_s: float = 0.0

    @property
    def node_type(self) -> str:
        return "compute-node"

    def estimate_compute_delay(self, task: Task) -> float:
        return compute_computing_delay(
            total_compute_cycles=task.total_compute_cycles,
            device_compute_rate_cycles_per_s=self.compute_capacity_cycles_per_s,
            parallel_efficiency=task.parallel_efficiency,
        )

    def estimate_finish_time(
        self,
        task: Task,
        arrival_time_s: float,
        priority_eta: float,
        current_time_s: float,
        task_id: str,
    ) -> tuple[float, float, float]:
        compute_delay_s = self.estimate_compute_delay(task)
        queue_entry = self.queue_manager.estimate(
            task_id=task_id,
            arrival_time_s=arrival_time_s,
            service_time_s=compute_delay_s,
            priority_eta=priority_eta,
            current_time_s=current_time_s,
        )
        return queue_entry.start_time_s, queue_entry.finish_time_s, queue_entry.queue_delay_s

    def commit_task(
        self,
        task: Task,
        arrival_time_s: float,
        priority_eta: float,
        current_time_s: float,
        task_id: str,
    ) -> tuple[float, float, float, float]:
        compute_delay_s = self.estimate_compute_delay(task)
        queue_entry = self.queue_manager.commit(
            task_id=task_id,
            arrival_time_s=arrival_time_s,
            service_time_s=compute_delay_s,
            priority_eta=priority_eta,
            current_time_s=current_time_s,
        )
        return (
            queue_entry.start_time_s,
            queue_entry.finish_time_s,
            queue_entry.queue_delay_s,
            compute_delay_s,
        )

    def queue_snapshot(self, current_time_s: float) -> QueueSnapshot:
        return self.queue_manager.snapshot(current_time_s)

    def cancel_task(self, task_id: str, current_time_s: float) -> bool:
        return self.queue_manager.cancel(task_id, current_time_s)


@dataclass
class BaseStation(ComputeNode):
    @property
    def node_type(self) -> str:
        return "bs"


@dataclass
class UAV(ComputeNode):
    speed_m_per_s: float = 0.0
    heading_rad: float = 0.0
    cluster_id: int | None = None
    head_uav_id: str | None = None
    is_cluster_head: bool = False
    is_isolated: bool = False
    battery_capacity_j: float = 150_000.0
    remaining_energy_j: float = 150_000.0
    safe_energy_ratio: float = 0.15
    episode_energy_consumed_j: float = 0.0

    @property
    def safe_energy_j(self) -> float:
        return float(self.safe_energy_ratio * self.battery_capacity_j)

    @property
    def battery_level(self) -> float:
        return float(self.remaining_energy_j / self.battery_capacity_j)

    @property
    def can_serve(self) -> bool:
        return self.remaining_energy_j > self.safe_energy_j

    def reset_battery(self, initial_energy_j: float | None = None) -> None:
        energy = self.battery_capacity_j if initial_energy_j is None else initial_energy_j
        self.remaining_energy_j = float(min(max(energy, 0.0), self.battery_capacity_j))
        self.episode_energy_consumed_j = 0.0

    def consume_energy(self, energy_j: float) -> float:
        if energy_j < 0:
            raise ValueError("energy_j must be non-negative.")
        consumed = min(float(energy_j), self.remaining_energy_j)
        self.remaining_energy_j -= consumed
        self.episode_energy_consumed_j += consumed
        return consumed

    def restore_energy(self, energy_j: float) -> None:
        if energy_j < 0:
            raise ValueError("energy_j must be non-negative.")
        restored = min(float(energy_j), self.battery_capacity_j - self.remaining_energy_j)
        self.remaining_energy_j += restored
        self.episode_energy_consumed_j = max(0.0, self.episode_energy_consumed_j - restored)

    @property
    def node_type(self) -> str:
        return "uav"

    def move(
        self,
        slot_length_s: float,
        rng: np.random.Generator,
        *,
        max_turn_angle_rad: float,
        area_side_length_m: float,
    ) -> None:
        turn_delta = rng.uniform(-max_turn_angle_rad, max_turn_angle_rad)
        self.heading_rad += float(turn_delta)
        dx = self.speed_m_per_s * slot_length_s * math.cos(self.heading_rad)
        dy = self.speed_m_per_s * slot_length_s * math.sin(self.heading_rad)
        new_x = self.position.x_m + dx
        new_y = self.position.y_m + dy

        if new_x < 0.0 or new_x > area_side_length_m:
            self.heading_rad = math.pi - self.heading_rad
            new_x = min(max(new_x, 0.0), area_side_length_m)
        if new_y < 0.0 or new_y > area_side_length_m:
            self.heading_rad = -self.heading_rad
            new_y = min(max(new_y, 0.0), area_side_length_m)

        self.position = Position(new_x, new_y, self.position.altitude_m)


@dataclass
class LEOSatellite(ComputeNode):
    controlled_uav_ids: list[str] = field(default_factory=list)

    @property
    def node_type(self) -> str:
        return "leo"

    def register_uavs(self, uavs: list[UAV]) -> None:
        self.controlled_uav_ids = [uav.node_id for uav in uavs]

    def evaluate_profit(self, task: Task, delay_sensitivity_lambda: float | None) -> float:
        if delay_sensitivity_lambda is None:
            return float(task.profit or 0.0)

        return compute_task_profit(
            input_size_bits=task.input_size_bits,
            cycles_per_bit=task.cycles_per_bit,
            tolerable_latency_s=task.tolerable_latency_s,
            delay_sensitivity_lambda=delay_sensitivity_lambda,
        )
