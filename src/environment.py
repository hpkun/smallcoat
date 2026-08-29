from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Iterable

import numpy as np

from .clustering import KMDUCManager
from .communication import CommunicationModel
from .communication import NetworkProfiles
from .config import SimulationConfig
from .constraints import ConstraintCheckResult
from .constraints import check_equation_10_deadline
from .constraints import check_equation_11_binary_action
from .constraints import check_equation_9_unique_offload
from .entities import BaseStation
from .entities import ExecutionRecord
from .entities import LEOSatellite
from .entities import TaskInstance
from .entities import UAV
from .energy import EnergyModel
from .task_generator import TaskGenerator


@dataclass(frozen=True)
class CandidateExecutionPlan:
    decision_uav_id: str
    target_node_id: str
    target_node_type: str
    compute_priority_eta: float
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
    execution_reliability: float
    transmission_reliability: float
    end_to_end_reliability: float
    expected_reliability: float
    satisfies_reliability: bool
    execution_failure_rate: float
    transmission_failure_rate: float
    completed_before_deadline: bool
    constraint_check: ConstraintCheckResult
    realized_profit: float


@dataclass(frozen=True)
class OffloadingAction:
    target_node_id: str | None = None
    priority_eta: float = 0.5
    redundancy_eta: float = 0.0
    backup_target_node_id: str | None = None
    replica_count: int | None = None
    replica_target_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = self.replica_count
        if count is not None and not 1 <= count <= 3:
            raise ValueError("replica_count must be in [1, 3].")
        if len(set(self.replica_target_node_ids)) != len(self.replica_target_node_ids):
            raise ValueError("replica_target_node_ids must be distinct.")


@dataclass(frozen=True)
class PlannedTaskAssignment:
    task_instance: TaskInstance
    ingress_uav: UAV
    decision_uav: UAV
    target_node: UAV | BaseStation | LEOSatellite
    plan: CandidateExecutionPlan
    redundancy_requested: bool = False
    replica_role: str = "primary"
    primary_target_node_id: str | None = None
    backup_target_node_id: str | None = None
    replica_index: int = 0
    requested_replica_count: int = 1
    replica_target_node_ids: tuple[str, ...] = ()


class SAGINEnvironment:
    def __init__(
        self,
        uavs: list[UAV],
        base_stations: list[BaseStation],
        leo_satellite: LEOSatellite,
        communication_model: CommunicationModel,
        network_profiles: NetworkProfiles,
        task_generator: TaskGenerator,
        simulation_config: SimulationConfig | None = None,
        clustering_manager: KMDUCManager | None = None,
        rng: np.random.Generator | None = None,
        redundancy_priority_threshold: float = 0.65,
        backup_capacity_reserve_ratio: float = 0.15,
        enable_redundancy: bool = True,
    ) -> None:
        self.uavs = uavs
        self.base_stations = base_stations
        self.leo_satellite = leo_satellite
        self.communication_model = communication_model
        self.network_profiles = network_profiles
        self.task_generator = task_generator
        self.simulation_config = simulation_config
        self.clustering_manager = clustering_manager
        self.rng = rng or np.random.default_rng()
        self.current_slot_index = 0
        if not 0.0 <= redundancy_priority_threshold <= 1.0:
            raise ValueError("redundancy_priority_threshold must be in [0, 1].")
        if not 0.0 <= backup_capacity_reserve_ratio < 1.0:
            raise ValueError("backup_capacity_reserve_ratio must be in [0, 1).")
        self.redundancy_priority_threshold = redundancy_priority_threshold
        self.backup_capacity_reserve_ratio = backup_capacity_reserve_ratio
        self.enable_redundancy = bool(enable_redundancy)
        self.energy_model = EnergyModel(
            simulation_config.energy if simulation_config is not None else None
        )
        energy_config = self.energy_model.config
        for uav in self.uavs:
            uav.battery_capacity_j = energy_config.uav_battery_capacity_j
            uav.safe_energy_ratio = energy_config.uav_safe_energy_ratio
            uav.reset_battery()

        self.leo_satellite.register_uavs(self.uavs)
        if self.clustering_manager is not None:
            self.clustering_manager.centralized_clustering(self.uavs, self.rng)

    def get_uav_by_id(self, uav_id: str) -> UAV:
        for uav in self.uavs:
            if uav.node_id == uav_id:
                return uav
        raise ValueError(f"Unknown ingress_uav_id: {uav_id}")

    def get_compute_node_by_id(self, node_id: str) -> UAV | BaseStation | LEOSatellite:
        """按节点编号查找计算节点，用于副本取消和资源释放。"""

        for node in [*self.uavs, *self.base_stations, self.leo_satellite]:
            if node.node_id == node_id:
                return node
        raise ValueError(f"Unknown compute node id: {node_id}")

    def link_profile_for_target(
        self,
        ingress_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
    ):
        if target_node.node_id == ingress_uav.node_id:
            return None
        if isinstance(target_node, LEOSatellite):
            return self.network_profiles.uav_to_leo
        if isinstance(target_node, UAV):
            return self.network_profiles.peer_uav_profile()
        return self.network_profiles.uav_to_bs

    def reset_batteries(self) -> None:
        for uav in self.uavs:
            uav.reset_battery()

    def battery_status(self) -> dict[str, dict[str, float | bool]]:
        return {
            uav.node_id: {
                "remaining_energy_j": uav.remaining_energy_j,
                "battery_level": uav.battery_level,
                "safe_energy_j": uav.safe_energy_j,
                "episode_energy_consumed_j": uav.episode_energy_consumed_j,
                "can_serve": uav.can_serve,
            }
            for uav in self.uavs
        }

    def cancel_later_replicas(
        self,
        replica_records: list[ExecutionRecord],
        successful_records: list[ExecutionRecord],
    ) -> tuple[
        list[ExecutionRecord],
        ExecutionRecord | None,
        list[tuple[float, str, str]],
    ]:
        """由最早成功完成的副本触发取消事件，并截断其他副本的能耗。"""

        if not successful_records:
            return replica_records, None, []

        winner = min(successful_records, key=lambda record: record.finish_time_s)
        cancellation_time_s = winner.finish_time_s
        updated_records: list[ExecutionRecord] = []
        cancellation_events: list[tuple[float, str, str]] = []
        for record in replica_records:
            # 已经完成（成功或失败）的副本不受后续取消事件影响。
            if record is winner or record.finish_time_s <= cancellation_time_s:
                updated_records.append(record)
                continue

            transmission_start_s = (
                record.arrival_at_target_s
                - record.backhaul_propagation_delay_s
                - record.backhaul_transmission_delay_s
            )
            used_transmission_s = min(
                record.backhaul_transmission_delay_s,
                max(0.0, cancellation_time_s - transmission_start_s),
            )
            transmission_fraction = (
                used_transmission_s / record.backhaul_transmission_delay_s
                if record.backhaul_transmission_delay_s > 0
                else 0.0
            )
            used_computing_s = min(
                record.compute_delay_s,
                max(0.0, cancellation_time_s - record.start_compute_s),
            )
            computing_fraction = (
                used_computing_s / record.compute_delay_s
                if record.compute_delay_s > 0
                else 0.0
            )
            actual_transmission_energy_j = (
                record.transmission_energy_j * transmission_fraction
            )
            actual_computing_energy_j = record.computing_energy_j * computing_fraction
            actual_total_energy_j = actual_transmission_energy_j + actual_computing_energy_j
            ingress_uav = self.get_uav_by_id(record.ingress_uav_id)
            ingress_uav.restore_energy(
                record.transmission_energy_j - actual_transmission_energy_j
            )
            target_node = self.get_compute_node_by_id(record.target_node_id)
            if isinstance(target_node, UAV):
                target_node.restore_energy(
                    record.computing_energy_j - actual_computing_energy_j
                )

            transmission_end_s = (
                transmission_start_s + record.backhaul_transmission_delay_s
            )
            if cancellation_time_s < transmission_end_s:
                cancellation_stage = "transmitting"
            elif cancellation_time_s < record.arrival_at_target_s:
                cancellation_stage = "propagating"
            elif cancellation_time_s < record.start_compute_s:
                cancellation_stage = "queued"
            else:
                cancellation_stage = "computing"

            cancellation_events.append(
                (cancellation_time_s, record.target_node_id, record.task_id)
            )
            updated_records.append(
                replace(
                    record,
                    completed_before_deadline=False,
                    realized_profit=0.0,
                    transmission_energy_j=actual_transmission_energy_j,
                    computing_energy_j=actual_computing_energy_j,
                    total_energy_j=actual_total_energy_j,
                    primary_replica_energy_j=(
                        actual_total_energy_j if record.replica_role == "primary" else 0.0
                    ),
                    backup_replica_energy_j=(
                        actual_total_energy_j
                        if record.replica_role.startswith("backup")
                        else 0.0
                    ),
                    cancellation_time_s=cancellation_time_s,
                    cancelled_replica_count=1,
                    cancellation_energy_saved_j=max(
                        0.0,
                        record.total_energy_j - actual_total_energy_j,
                    ),
                    replica_cancelled=True,
                    cancellation_stage=cancellation_stage,
                )
            )
        return updated_records, winner, cancellation_events

    def generate_tasks(
        self,
        slot_length_s: float,
        current_time_s: float,
        *,
        delay_sensitivity_lambda: float | None = None,
    ) -> list[TaskInstance]:
        return self.task_generator.generate_tasks(
            uavs=self.uavs,
            slot_length_s=slot_length_s,
            current_time_s=current_time_s,
            rng=self.rng,
            delay_sensitivity_lambda=delay_sensitivity_lambda,
        )

    def build_candidate_plan(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
        current_time_s: float,
        priority_eta: float | None = None,
        delay_sensitivity_lambda: float | None = None,
    ) -> CandidateExecutionPlan:
        task = task_instance.task
        compute_priority_eta = (
            self.derive_priority_eta(task)
            if priority_eta is None
            else float(min(max(priority_eta, 0.0), 1.0))
        )

        ingress_tx_s = 0.0
        ingress_prop_s = 0.0
        arrival_at_uav_s = current_time_s

        if target_node.node_id == ingress_uav.node_id:
            backhaul_tx_s = 0.0
            backhaul_prop_s = 0.0
            transmission_failure_rate = 0.0
            arrival_at_target_s = arrival_at_uav_s
        else:
            backhaul_profile = self.link_profile_for_target(ingress_uav, target_node)
            if backhaul_profile is None:
                raise RuntimeError("A remote target requires a link profile.")
            transmission_failure_rate = backhaul_profile.transmission_failure_rate
            backhaul_tx_s, backhaul_prop_s = self.communication_model.total_link_delay_s(
                data_size_bits=task.input_size_bits,
                sender=ingress_uav.position,
                receiver=target_node.position,
                profile=backhaul_profile,
            )
            arrival_at_target_s = arrival_at_uav_s + backhaul_tx_s + backhaul_prop_s

        start_compute_s, finish_time_s, queue_delay_s = target_node.estimate_finish_time(
            task=task,
            arrival_time_s=arrival_at_target_s,
            priority_eta=compute_priority_eta,
            current_time_s=current_time_s,
            task_id=task_instance.task_id,
        )
        compute_delay_s = finish_time_s - start_compute_s
        communication_delay_s = ingress_tx_s + ingress_prop_s + backhaul_tx_s + backhaul_prop_s
        total_delay_s = communication_delay_s + compute_delay_s
        actual_finish_delay_s = finish_time_s - task_instance.created_at_s
        execution_reliability = float(
            np.exp(-target_node.execution_failure_rate * max(compute_delay_s, 0.0))
        )
        transmission_reliability = float(
            np.exp(-transmission_failure_rate * max(backhaul_tx_s + backhaul_prop_s, 0.0))
        )
        end_to_end_reliability = float(execution_reliability * transmission_reliability)
        expected_reliability = float(task.expected_reliability)
        satisfies_reliability = end_to_end_reliability >= expected_reliability

        constraint_check = ConstraintCheckResult(
            satisfies_unique_offload=check_equation_9_unique_offload(1),
            satisfies_deadline=check_equation_10_deadline(actual_finish_delay_s, task_instance),
            satisfies_binary_action=check_equation_11_binary_action(True),
            # Compute capacity is enforced as queue service rate. Tasks may span
            # multiple slots, so admission does not require the full workload to
            # fit inside one slot.
            satisfies_capacity=True,
        )
        realized_profit = (
            self.leo_satellite.evaluate_profit(task, delay_sensitivity_lambda)
            if constraint_check.feasible
            else 0.0
        )
        return CandidateExecutionPlan(
            decision_uav_id=decision_uav.node_id,
            target_node_id=target_node.node_id,
            target_node_type=target_node.node_type,
            compute_priority_eta=compute_priority_eta,
            arrival_at_target_s=arrival_at_target_s,
            start_compute_s=start_compute_s,
            finish_time_s=finish_time_s,
            ingress_transmission_delay_s=ingress_tx_s,
            ingress_propagation_delay_s=ingress_prop_s,
            backhaul_transmission_delay_s=backhaul_tx_s,
            backhaul_propagation_delay_s=backhaul_prop_s,
            queue_delay_s=queue_delay_s,
            compute_delay_s=compute_delay_s,
            communication_delay_s=communication_delay_s,
            total_delay_s=total_delay_s,
            actual_finish_delay_s=actual_finish_delay_s,
            execution_reliability=execution_reliability,
            transmission_reliability=transmission_reliability,
            end_to_end_reliability=end_to_end_reliability,
            expected_reliability=expected_reliability,
            satisfies_reliability=satisfies_reliability,
            execution_failure_rate=target_node.execution_failure_rate,
            transmission_failure_rate=transmission_failure_rate,
            completed_before_deadline=constraint_check.satisfies_deadline,
            constraint_check=constraint_check,
            realized_profit=realized_profit,
        )

    def select_best_plan(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
        current_time_s: float,
        delay_sensitivity_lambda: float | None = None,
    ) -> tuple[CandidateExecutionPlan, UAV | BaseStation | LEOSatellite]:
        candidates: list[tuple[CandidateExecutionPlan, UAV | BaseStation | LEOSatellite]] = []
        for target_node in self.iter_compute_targets(decision_uav, ingress_uav):
            plan = self.build_candidate_plan(
                task_instance=task_instance,
                ingress_uav=ingress_uav,
                decision_uav=decision_uav,
                target_node=target_node,
                current_time_s=current_time_s,
                delay_sensitivity_lambda=delay_sensitivity_lambda,
            )
            candidates.append((plan, target_node))

        feasible = [item for item in candidates if item[0].constraint_check.feasible]
        if feasible:
            return max(feasible, key=lambda item: (item[0].realized_profit, -item[0].total_delay_s))
        return min(candidates, key=lambda item: item[0].total_delay_s)

    def get_target_by_id(
        self,
        decision_uav: UAV,
        ingress_uav: UAV,
        target_node_id: str,
    ) -> UAV | BaseStation | LEOSatellite:
        if ingress_uav.node_id == target_node_id:
            return ingress_uav
        for uav in self.uavs:
            if uav.node_id == target_node_id:
                return uav
        for bs in self.base_stations:
            if bs.node_id == target_node_id:
                return bs
        if self.leo_satellite.node_id == target_node_id:
            return self.leo_satellite
        raise ValueError(f"Unknown or unavailable target_node_id: {target_node_id}")

    def select_plan_from_action(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
        current_time_s: float,
        action: OffloadingAction,
        delay_sensitivity_lambda: float | None = None,
    ) -> tuple[CandidateExecutionPlan, UAV | BaseStation | LEOSatellite]:
        target_node = self.get_target_by_id(
            decision_uav=decision_uav,
            ingress_uav=ingress_uav,
            target_node_id=action.target_node_id,
        )
        plan = self.build_candidate_plan(
            task_instance=task_instance,
            ingress_uav=ingress_uav,
            decision_uav=decision_uav,
            target_node=target_node,
            current_time_s=current_time_s,
            priority_eta=action.priority_eta,
            delay_sensitivity_lambda=delay_sensitivity_lambda,
        )
        return plan, target_node

    def should_redundantly_offload(
        self,
        action: OffloadingAction,
        primary_plan: CandidateExecutionPlan,
    ) -> bool:
        """仅在策略请求冗余且主方案可靠性不足时生成一个备份副本。"""

        if not self.enable_redundancy:
            return False
        requests_redundancy = action.redundancy_eta >= self.redundancy_priority_threshold
        primary_has_high_failure_risk = (
            primary_plan.end_to_end_reliability < primary_plan.expected_reliability
        )
        return requests_redundancy and primary_has_high_failure_risk

    def sample_poisson_failure(self, failure_rate: float, exposure_time_s: float) -> bool:
        """
        Sample whether at least one failure occurs during an exposure interval.

        HRTO models execution and transmission failures as Poisson processes.
        The probability of no failure is exp(-rate * time), so a task instance
        fails when at least one event occurs in the interval.
        """

        if failure_rate <= 0.0 or exposure_time_s <= 0.0:
            return False
        mean_failures = float(failure_rate) * float(exposure_time_s)
        return bool(self.rng.poisson(mean_failures) > 0)

    def select_backup_target(
        self,
        primary_target_node: UAV | BaseStation | LEOSatellite,
        ingress_uav: UAV,
    ) -> BaseStation | LEOSatellite | None:
        if isinstance(primary_target_node, UAV | LEOSatellite):
            reachable_base_stations = list(self.base_stations)
            if self.clustering_manager is not None:
                radius_m = self.clustering_manager.config.communication_radius_m
                reachable_base_stations = [
                    bs
                    for bs in self.base_stations
                    if bs.position.distance_to(ingress_uav.position) <= radius_m
                ]
            if reachable_base_stations:
                return min(
                    reachable_base_stations,
                    key=lambda bs: bs.position.distance_to(ingress_uav.position),
                )
            if isinstance(primary_target_node, UAV):
                return self.leo_satellite
            return None
        if isinstance(primary_target_node, BaseStation):
            return self.leo_satellite
        return None

    def is_valid_backup_target(
        self,
        primary_target_node: UAV | BaseStation | LEOSatellite,
        backup_target_node: UAV | BaseStation | LEOSatellite,
    ) -> bool:
        if backup_target_node.node_id == primary_target_node.node_id:
            return False
        if isinstance(primary_target_node, UAV):
            return isinstance(backup_target_node, BaseStation | LEOSatellite)
        if isinstance(primary_target_node, BaseStation):
            return isinstance(backup_target_node, LEOSatellite)
        if isinstance(primary_target_node, LEOSatellite):
            return isinstance(backup_target_node, BaseStation)
        return False

    def resolve_backup_target(
        self,
        action: OffloadingAction,
        primary_target_node: UAV | BaseStation | LEOSatellite,
        decision_uav: UAV,
        ingress_uav: UAV,
    ) -> BaseStation | LEOSatellite | None:
        if action.backup_target_node_id is not None:
            backup_node = self.get_target_by_id(
                decision_uav=decision_uav,
                ingress_uav=ingress_uav,
                target_node_id=action.backup_target_node_id,
            )
            if self.is_valid_backup_target(primary_target_node, backup_node):
                return backup_node
        return self.select_backup_target(primary_target_node, ingress_uav)

    def iter_compute_targets(
        self,
        decision_uav: UAV,
        ingress_uav: UAV,
    ) -> Iterable[UAV | BaseStation | LEOSatellite]:
        if ingress_uav.can_serve:
            yield ingress_uav
        radius_m = None
        if self.clustering_manager is not None:
            radius_m = self.clustering_manager.config.communication_radius_m
        elif self.simulation_config is not None:
            radius_m = self.simulation_config.clustering.communication_radius_m
        for peer_uav in self.uavs:
            if peer_uav.node_id == ingress_uav.node_id or not peer_uav.can_serve:
                continue
            if radius_m is not None and peer_uav.position.distance_to(ingress_uav.position) > radius_m:
                continue
            yield peer_uav
        if self.clustering_manager is None:
            for bs in self.base_stations:
                yield bs
        else:
            radius_m = self.clustering_manager.config.communication_radius_m
            for bs in self.base_stations:
                if bs.position.distance_to(ingress_uav.position) <= radius_m:
                    yield bs
        yield self.leo_satellite

    def commit_plan(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
        plan: CandidateExecutionPlan,
        current_time_s: float,
        delay_sensitivity_lambda: float | None = None,
        redundancy_requested: bool = False,
        replica_role: str = "primary",
        primary_target_node_id: str | None = None,
        backup_target_node_id: str | None = None,
        replica_index: int = 0,
        requested_replica_count: int = 1,
        replica_target_node_ids: tuple[str, ...] = (),
    ) -> ExecutionRecord:
        start_compute_s, finish_time_s, queue_delay_s, compute_delay_s = target_node.commit_task(
            task=task_instance.task,
            arrival_time_s=plan.arrival_at_target_s,
            priority_eta=plan.compute_priority_eta,
            current_time_s=current_time_s,
            task_id=task_instance.task_id,
        )
        communication_delay_s = (
            plan.ingress_transmission_delay_s
            + plan.ingress_propagation_delay_s
            + plan.backhaul_transmission_delay_s
            + plan.backhaul_propagation_delay_s
        )
        total_delay_s = communication_delay_s + compute_delay_s
        actual_finish_delay_s = finish_time_s - task_instance.created_at_s
        execution_reliability = float(
            np.exp(-target_node.execution_failure_rate * max(compute_delay_s, 0.0))
        )
        transmission_reliability = plan.transmission_reliability
        end_to_end_reliability = float(execution_reliability * transmission_reliability)
        expected_reliability = float(task_instance.task.expected_reliability)
        satisfies_reliability = end_to_end_reliability >= expected_reliability
        transmission_exposure_s = (
            plan.backhaul_transmission_delay_s + plan.backhaul_propagation_delay_s
        )
        execution_failed = self.sample_poisson_failure(
            target_node.execution_failure_rate,
            compute_delay_s,
        )
        transmission_failed = self.sample_poisson_failure(
            plan.transmission_failure_rate,
            transmission_exposure_s,
        )
        failed_due_to_reliability = execution_failed or transmission_failed
        constraint_check = ConstraintCheckResult(
            satisfies_unique_offload=plan.constraint_check.satisfies_unique_offload,
            satisfies_deadline=check_equation_10_deadline(actual_finish_delay_s, task_instance),
            satisfies_binary_action=plan.constraint_check.satisfies_binary_action,
            satisfies_capacity=plan.constraint_check.satisfies_capacity,
        )
        completed_before_deadline = (
            constraint_check.satisfies_deadline
            and not failed_due_to_reliability
        )
        realized_profit = (
            self.leo_satellite.evaluate_profit(
                task_instance.task,
                delay_sensitivity_lambda,
            )
            if constraint_check.feasible and not failed_due_to_reliability
            else 0.0
        )
        backhaul_profile = self.link_profile_for_target(ingress_uav, target_node)
        # 能耗只统计真正提交执行的任务；传播时延不产生发射能耗。
        energy = self.energy_model.compute(
            task=task_instance.task,
            node_type=target_node.node_type,
            backhaul_transmission_delay_s=plan.backhaul_transmission_delay_s,
            backhaul_profile=backhaul_profile,
        )
        ingress_uav.consume_energy(energy.transmission_energy_j)
        if isinstance(target_node, UAV):
            target_node.consume_energy(energy.computing_energy_j)

        return ExecutionRecord(
            task_id=task_instance.task_id,
            ingress_uav_id=ingress_uav.node_id,
            decision_uav_id=decision_uav.node_id,
            target_node_id=target_node.node_id,
            target_node_type=target_node.node_type,
            compute_priority_eta=plan.compute_priority_eta,
            created_at_s=task_instance.created_at_s,
            arrival_at_uav_s=task_instance.created_at_s,
            arrival_at_target_s=plan.arrival_at_target_s,
            start_compute_s=start_compute_s,
            finish_time_s=finish_time_s,
            ingress_transmission_delay_s=plan.ingress_transmission_delay_s,
            ingress_propagation_delay_s=plan.ingress_propagation_delay_s,
            backhaul_transmission_delay_s=plan.backhaul_transmission_delay_s,
            backhaul_propagation_delay_s=plan.backhaul_propagation_delay_s,
            queue_delay_s=queue_delay_s,
            compute_delay_s=compute_delay_s,
            communication_delay_s=communication_delay_s,
            total_delay_s=total_delay_s,
            actual_finish_delay_s=actual_finish_delay_s,
            completed_before_deadline=completed_before_deadline,
            realized_profit=realized_profit,
            transmission_energy_j=energy.transmission_energy_j,
            computing_energy_j=energy.computing_energy_j,
            total_energy_j=energy.total_energy_j,
            primary_replica_energy_j=(
                energy.total_energy_j if replica_role == "primary" else 0.0
            ),
            backup_replica_energy_j=(
                energy.total_energy_j if replica_role.startswith("backup") else 0.0
            ),
            replica_index=replica_index,
            requested_replica_count=requested_replica_count,
            admitted_replica_count=1,
            replica_target_node_ids=replica_target_node_ids,
            redundancy_requested=redundancy_requested,
            is_redundant_task=requested_replica_count > 1,
            replica_role=replica_role,
            primary_target_node_id=primary_target_node_id,
            backup_target_node_id=backup_target_node_id,
            execution_reliability=execution_reliability,
            transmission_reliability=transmission_reliability,
            end_to_end_reliability=end_to_end_reliability,
            expected_reliability=expected_reliability,
            satisfies_reliability=satisfies_reliability,
            execution_failure_rate=target_node.execution_failure_rate,
            transmission_failure_rate=plan.transmission_failure_rate,
            execution_failed=execution_failed,
            transmission_failed=transmission_failed,
            failed_due_to_reliability=failed_due_to_reliability,
            workflow_id=task_instance.workflow_id,
            owner_ch_id=task_instance.owner_ch_id,
            workflow_task_count=task_instance.workflow_task_count,
            workflow_step_index=task_instance.workflow_step_index,
            predecessor_task_ids=task_instance.predecessor_task_ids,
            successor_task_ids=task_instance.successor_task_ids,
            constraint_check=constraint_check,
        )

    def build_capacity_rejection_record(
        self,
        assignment: PlannedTaskAssignment,
    ) -> ExecutionRecord:
        """Build a record for a task replica rejected by a full compute queue."""

        plan = assignment.plan
        rejected_constraint = ConstraintCheckResult(
            satisfies_unique_offload=plan.constraint_check.satisfies_unique_offload,
            # A replica that never entered the queue is a capacity drop, not a
            # deadline failure.
            satisfies_deadline=True,
            satisfies_binary_action=plan.constraint_check.satisfies_binary_action,
            satisfies_capacity=False,
        )
        return ExecutionRecord(
            task_id=assignment.task_instance.task_id,
            ingress_uav_id=assignment.ingress_uav.node_id,
            decision_uav_id=assignment.decision_uav.node_id,
            target_node_id=assignment.target_node.node_id,
            target_node_type=assignment.target_node.node_type,
            compute_priority_eta=plan.compute_priority_eta,
            created_at_s=assignment.task_instance.created_at_s,
            arrival_at_uav_s=assignment.task_instance.created_at_s,
            arrival_at_target_s=plan.arrival_at_target_s,
            start_compute_s=plan.start_compute_s,
            finish_time_s=plan.finish_time_s,
            ingress_transmission_delay_s=plan.ingress_transmission_delay_s,
            ingress_propagation_delay_s=plan.ingress_propagation_delay_s,
            backhaul_transmission_delay_s=plan.backhaul_transmission_delay_s,
            backhaul_propagation_delay_s=plan.backhaul_propagation_delay_s,
            queue_delay_s=plan.queue_delay_s,
            compute_delay_s=plan.compute_delay_s,
            communication_delay_s=plan.communication_delay_s,
            total_delay_s=plan.total_delay_s,
            actual_finish_delay_s=plan.actual_finish_delay_s,
            completed_before_deadline=False,
            realized_profit=0.0,
            replica_index=assignment.replica_index,
            requested_replica_count=assignment.requested_replica_count,
            admitted_replica_count=0,
            capacity_rejected_replica_count=1,
            replica_target_node_ids=assignment.replica_target_node_ids,
            capacity_rejected=True,
            redundancy_requested=assignment.redundancy_requested,
            is_redundant_task=assignment.requested_replica_count > 1,
            replica_role=assignment.replica_role,
            primary_target_node_id=assignment.primary_target_node_id,
            backup_target_node_id=assignment.backup_target_node_id,
            execution_reliability=plan.execution_reliability,
            transmission_reliability=plan.transmission_reliability,
            end_to_end_reliability=plan.end_to_end_reliability,
            expected_reliability=plan.expected_reliability,
            satisfies_reliability=plan.satisfies_reliability,
            execution_failure_rate=plan.execution_failure_rate,
            transmission_failure_rate=plan.transmission_failure_rate,
            execution_failed=False,
            transmission_failed=False,
            failed_due_to_reliability=False,
            workflow_id=assignment.task_instance.workflow_id,
            owner_ch_id=assignment.task_instance.owner_ch_id,
            workflow_task_count=assignment.task_instance.workflow_task_count,
            workflow_step_index=assignment.task_instance.workflow_step_index,
            predecessor_task_ids=assignment.task_instance.predecessor_task_ids,
            successor_task_ids=assignment.task_instance.successor_task_ids,
            constraint_check=rejected_constraint,
        )

    def build_energy_rejection_record(
        self,
        assignment: PlannedTaskAssignment,
    ) -> ExecutionRecord:
        record = self.build_capacity_rejection_record(assignment)
        if record.constraint_check is None:
            raise RuntimeError("An admission rejection must include constraint results.")
        return replace(
            record,
            capacity_rejected=False,
            energy_rejected=True,
            capacity_rejected_replica_count=0,
            constraint_check=replace(
                record.constraint_check,
                satisfies_capacity=True,
                satisfies_energy=False,
            ),
        )

    def assignment_has_safe_energy(self, assignment: PlannedTaskAssignment) -> bool:
        """Reject a UAV placement before it crosses any battery safety floor."""

        profile = self.link_profile_for_target(
            assignment.ingress_uav,
            assignment.target_node,
        )
        energy = self.energy_model.compute(
            task=assignment.task_instance.task,
            node_type=assignment.target_node.node_type,
            backhaul_transmission_delay_s=assignment.plan.backhaul_transmission_delay_s,
            backhaul_profile=profile,
        )
        required_by_uav: dict[str, float] = {
            assignment.ingress_uav.node_id: energy.transmission_energy_j
        }
        if isinstance(assignment.target_node, UAV):
            required_by_uav[assignment.target_node.node_id] = (
                required_by_uav.get(assignment.target_node.node_id, 0.0)
                + energy.computing_energy_j
            )
        return all(
            self.get_uav_by_id(uav_id).remaining_energy_j - required_energy_j
            >= self.get_uav_by_id(uav_id).safe_energy_j
            for uav_id, required_energy_j in required_by_uav.items()
        )

    def queue_has_capacity(
        self,
        assignment: PlannedTaskAssignment,
        current_time_s: float,
    ) -> bool:
        """检查目标节点的有限缓冲队列是否还能接纳当前任务副本。"""

        # 未提供仿真配置时不启用有限缓冲约束，保持基础环境的兼容行为。
        if self.simulation_config is None:
            return True

        # 当前工作量包含节点正在执行任务的剩余服务时间，以及缓冲队列中
        # 所有待执行任务的服务时间，单位为秒。
        queued_workload_s = assignment.target_node.queue_manager.workload_s(current_time_s)

        # 新任务的服务时间由计算量、目标节点算力和任务并行效率共同决定；
        # 任务允许跨越多个时隙执行。
        service_time_s = assignment.target_node.estimate_compute_delay(
            assignment.task_instance.task
        )

        # 根据目标节点所属层取得对应阈值：UAV、BS 和 LEO 分别使用独立配置。
        workload_limit_s = self.simulation_config.queue_capacity.limit_for(
            assignment.target_node.node_type
        )
        # 只有加入新任务后的总工作量不超过该层有限缓冲阈值时才允许入队；
        # 超过阈值的任务副本会被标记为容量丢弃。
        return queued_workload_s + service_time_s <= workload_limit_s

    def select_admissible_backup_assignment(
        self,
        assignment: PlannedTaskAssignment,
        current_time_s: float,
        delay_sensitivity_lambda: float | None,
    ) -> PlannedTaskAssignment | None:
        """Use the requested backup when feasible, otherwise try another layer."""

        primary_target = self.get_compute_node_by_id(
            assignment.primary_target_node_id or assignment.target_node.node_id
        )
        candidates = [
            assignment.target_node,
            *self.iter_compute_targets(assignment.decision_uav, assignment.ingress_uav),
        ]
        candidate_assignments: list[PlannedTaskAssignment] = []
        seen_node_ids: set[str] = set()
        for target_node in candidates:
            if target_node.node_id in seen_node_ids:
                continue
            seen_node_ids.add(target_node.node_id)
            if not self.is_valid_backup_target(primary_target, target_node):
                continue
            plan = self.build_candidate_plan(
                task_instance=assignment.task_instance,
                ingress_uav=assignment.ingress_uav,
                decision_uav=assignment.decision_uav,
                target_node=target_node,
                current_time_s=current_time_s,
                priority_eta=assignment.plan.compute_priority_eta,
                delay_sensitivity_lambda=delay_sensitivity_lambda,
            )
            candidate = replace(
                assignment,
                target_node=target_node,
                plan=plan,
                backup_target_node_id=target_node.node_id,
            )
            if not plan.constraint_check.satisfies_deadline:
                continue
            if not self.queue_has_capacity(candidate, current_time_s):
                continue
            candidate_assignments.append(candidate)

        if not candidate_assignments:
            return None
        if candidate_assignments[0].target_node.node_id == assignment.target_node.node_id:
            return candidate_assignments[0]
        return min(
            candidate_assignments,
            key=lambda candidate: (
                candidate.plan.actual_finish_delay_s,
                -candidate.plan.end_to_end_reliability,
            ),
        )

    def derive_priority_eta(self, task) -> float:
        delta_max = self.task_generator.task_model_config.tolerable_latency_s.high
        if delta_max <= 0:
            return 0.5
        eta = 1.0 - min(max(task.tolerable_latency_s / delta_max, 0.0), 1.0)
        return float(min(max(eta, 0.0), 1.0))

    def get_decision_uav(self, ingress_uav: UAV) -> UAV:
        if self.clustering_manager is None:
            return ingress_uav
        cluster = self.clustering_manager.get_cluster_info_for_uav(ingress_uav.node_id)
        if cluster is None:
            return ingress_uav
        head_uav_id = self.clustering_manager.resolve_serviceable_head(
            cluster.cluster_id,
            self.uavs,
        )
        if head_uav_id is None:
            return ingress_uav
        for uav in self.uavs:
            if uav.node_id == head_uav_id:
                return uav
        return ingress_uav

    def update_clusters(self) -> None:
        if self.clustering_manager is None:
            return
        period = self.clustering_manager.config.clustering_period_slots
        if period > 0 and self.current_slot_index % period == 0:
            self.clustering_manager.centralized_clustering(self.uavs, self.rng)
        else:
            self.clustering_manager.maintain_clusters(self.uavs)

    def advance_system_dynamics(self, slot_length_s: float) -> None:
        if self.simulation_config is None:
            raise ValueError("simulation_config is required to advance system dynamics.")
        for uav in self.uavs:
            if uav.remaining_energy_j > 0.0:
                uav.consume_energy(
                    self.energy_model.config.uav_propulsion_power_w * slot_length_s
                )
            uav.move(
                slot_length_s,
                self.rng,
                max_turn_angle_rad=self.simulation_config.mobility.max_turn_angle_rad,
                area_side_length_m=self.simulation_config.area.side_length_m,
            )
        self.current_slot_index += 1
        self.update_clusters()

    def step(
        self,
        slot_length_s: float,
        current_time_s: float,
        *,
        delay_sensitivity_lambda: float | None = None,
        move_uavs: bool = False,
        actions_by_task_id: dict[str, OffloadingAction] | None = None,
        external_tasks: list[TaskInstance] | None = None,
        apply_pre_step_dynamics: bool = True,
    ) -> list[ExecutionRecord]:
        if apply_pre_step_dynamics and move_uavs:
            if self.simulation_config is None:
                raise ValueError("simulation_config is required when move_uavs=True.")
            for uav in self.uavs:
                uav.move(
                    slot_length_s,
                    self.rng,
                    max_turn_angle_rad=self.simulation_config.mobility.max_turn_angle_rad,
                    area_side_length_m=self.simulation_config.area.side_length_m,
                )

        if apply_pre_step_dynamics:
            self.update_clusters()

        tasks = (
            external_tasks
            if external_tasks is not None
            else self.generate_tasks(
                slot_length_s=slot_length_s,
                current_time_s=current_time_s,
                delay_sensitivity_lambda=delay_sensitivity_lambda,
            )
        )

        assignments: list[PlannedTaskAssignment] = []
        for task_instance in tasks:
            ingress_uav = self.get_uav_by_id(task_instance.ingress_uav_id)
            decision_uav = self.get_decision_uav(ingress_uav)
            action = actions_by_task_id.get(task_instance.task_id) if actions_by_task_id else None

            if action is None:
                primary_plan, primary_target = self.select_best_plan(
                    task_instance=task_instance,
                    ingress_uav=ingress_uav,
                    decision_uav=decision_uav,
                    current_time_s=current_time_s,
                    delay_sensitivity_lambda=delay_sensitivity_lambda,
                )
                target_nodes = [primary_target]
                plans = [primary_plan]
                requested_replica_count = 1
            elif action.replica_target_node_ids:
                requested_replica_count = action.replica_count or len(action.replica_target_node_ids)
                target_nodes = [
                    self.get_target_by_id(decision_uav, ingress_uav, target_node_id)
                    for target_node_id in action.replica_target_node_ids[:3]
                ]
                plans = [
                    self.build_candidate_plan(
                        task_instance=task_instance,
                        ingress_uav=ingress_uav,
                        decision_uav=decision_uav,
                        target_node=target_node,
                        current_time_s=current_time_s,
                        priority_eta=action.priority_eta,
                        delay_sensitivity_lambda=delay_sensitivity_lambda,
                    )
                    for target_node in target_nodes
                ]
            else:
                primary_plan, primary_target = self.select_plan_from_action(
                    task_instance=task_instance,
                    ingress_uav=ingress_uav,
                    decision_uav=decision_uav,
                    current_time_s=current_time_s,
                    action=action,
                    delay_sensitivity_lambda=delay_sensitivity_lambda,
                )
                target_nodes = [primary_target]
                plans = [primary_plan]
                if self.should_redundantly_offload(action, primary_plan):
                    legacy_backup = self.resolve_backup_target(
                        action,
                        primary_target,
                        decision_uav,
                        ingress_uav,
                    )
                    if legacy_backup is not None:
                        target_nodes.append(legacy_backup)
                        plans.append(
                            self.build_candidate_plan(
                                task_instance=task_instance,
                                ingress_uav=ingress_uav,
                                decision_uav=decision_uav,
                                target_node=legacy_backup,
                                current_time_s=current_time_s,
                                priority_eta=action.priority_eta,
                                delay_sensitivity_lambda=delay_sensitivity_lambda,
                            )
                        )
                requested_replica_count = len(target_nodes)

            target_ids = tuple(target_node.node_id for target_node in target_nodes)
            if len(set(target_ids)) != len(target_ids):
                raise ValueError("A task action selected the same node for multiple replicas.")
            primary_target_node_id = target_ids[0]
            legacy_backup_target_node_id = target_ids[1] if len(target_ids) > 1 else None
            for replica_index, (target_node, plan) in enumerate(zip(target_nodes, plans)):
                legacy_action = action is not None and not action.replica_target_node_ids
                assignments.append(
                    PlannedTaskAssignment(
                        task_instance=task_instance,
                        ingress_uav=ingress_uav,
                        decision_uav=decision_uav,
                        target_node=target_node,
                        plan=plan,
                        redundancy_requested=requested_replica_count > 1,
                        replica_role=(
                            "primary"
                            if replica_index == 0
                            else ("backup" if legacy_action else f"backup-{replica_index}")
                        ),
                        primary_target_node_id=primary_target_node_id,
                        backup_target_node_id=legacy_backup_target_node_id,
                        replica_index=replica_index,
                        requested_replica_count=requested_replica_count,
                        replica_target_node_ids=target_ids,
                    )
                )

        ordered_assignments = sorted(
            assignments,
            key=lambda assignment: (
                assignment.replica_index,
                -assignment.plan.compute_priority_eta,
                assignment.task_instance.created_at_s,
                assignment.task_instance.task_id,
            ),
        )
        records_by_task_id: dict[str, list[ExecutionRecord]] = {}
        for assignment in ordered_assignments:
            if assignment.replica_role == "backup":
                legacy_assignment = self.select_admissible_backup_assignment(
                    assignment,
                    current_time_s,
                    delay_sensitivity_lambda,
                )
                if legacy_assignment is None:
                    continue
                assignment = legacy_assignment
            if not self.assignment_has_safe_energy(assignment):
                record = self.build_energy_rejection_record(assignment)
            elif not self.queue_has_capacity(assignment, current_time_s):
                record = self.build_capacity_rejection_record(assignment)
            else:
                record = self.commit_plan(
                    task_instance=assignment.task_instance,
                    ingress_uav=assignment.ingress_uav,
                    decision_uav=assignment.decision_uav,
                    target_node=assignment.target_node,
                    plan=assignment.plan,
                    current_time_s=current_time_s,
                    delay_sensitivity_lambda=delay_sensitivity_lambda,
                    redundancy_requested=assignment.redundancy_requested,
                    replica_role=assignment.replica_role,
                    primary_target_node_id=assignment.primary_target_node_id,
                    backup_target_node_id=assignment.backup_target_node_id,
                    replica_index=assignment.replica_index,
                    requested_replica_count=assignment.requested_replica_count,
                    replica_target_node_ids=assignment.replica_target_node_ids,
                )
            records_by_task_id.setdefault(record.task_id, []).append(record)

        records: list[ExecutionRecord] = []
        cancellation_events: list[tuple[float, str, str]] = []
        for task_instance in tasks:
            replica_records = records_by_task_id.get(task_instance.task_id, [])
            if not replica_records:
                continue
            admitted_records = [
                record
                for record in replica_records
                if record.constraint_check is not None
                and record.constraint_check.satisfies_capacity
                and record.constraint_check.satisfies_energy
            ]
            rejected_records = [record for record in replica_records if record not in admitted_records]
            successful_records = [
                record
                for record in admitted_records
                if record.completed_before_deadline and not record.failed_due_to_reliability
            ]
            admitted_records, winner, task_cancellation_events = self.cancel_later_replicas(
                admitted_records,
                successful_records,
            )
            cancellation_events.extend(task_cancellation_events)
            replica_records = sorted(
                [*admitted_records, *rejected_records],
                key=lambda record: record.replica_index,
            )
            admitted_replica_count = len(admitted_records)
            capacity_rejected_replica_count = sum(
                int(record.capacity_rejected) for record in replica_records
            )
            requested_replica_count = replica_records[0].requested_replica_count
            winner_replica_index = winner.replica_index if winner is not None else None
            selected_record = (
                next(
                    record
                    for record in admitted_records
                    if record.replica_index == winner_replica_index
                )
                if winner_replica_index is not None
                else min(replica_records, key=lambda record: record.actual_finish_delay_s)
            )
            backup_succeeded = any(
                record.replica_role.startswith("backup")
                and record.completed_before_deadline
                and not record.replica_cancelled
                and not record.failed_due_to_reliability
                for record in admitted_records
            )
            actual_backup_target_node_id = next(
                (
                    record.target_node_id
                    for record in admitted_records
                    if record.replica_index == 1
                ),
                None,
            )
            failure_probability_product = 1.0
            for record in admitted_records:
                reliability = min(max(record.end_to_end_reliability, 0.0), 1.0)
                failure_probability_product *= 1.0 - reliability
            combined_reliability = (
                float(1.0 - failure_probability_product) if admitted_records else 0.0
            )
            expected_reliability = float(task_instance.task.expected_reliability)
            satisfies_reliability = combined_reliability >= expected_reliability
            is_redundant_task = admitted_replica_count > 1
            task_failed_due_to_reliability = not bool(successful_records) and any(
                record.failed_due_to_reliability for record in admitted_records
            )
            selected_constraint = selected_record.constraint_check
            if selected_constraint is not None:
                selected_constraint = replace(
                    selected_constraint,
                    satisfies_capacity=admitted_replica_count > 0,
                    satisfies_energy=admitted_replica_count > 0,
                )
            records.append(
                replace(
                    selected_record,
                    realized_profit=(selected_record.realized_profit if winner is not None else 0.0),
                    completed_before_deadline=bool(successful_records),
                    requested_replica_count=requested_replica_count,
                    admitted_replica_count=admitted_replica_count,
                    capacity_rejected_replica_count=capacity_rejected_replica_count,
                    replica_target_node_ids=replica_records[0].replica_target_node_ids,
                    backup_target_node_id=actual_backup_target_node_id,
                    winner_replica_index=winner_replica_index,
                    capacity_rejected=capacity_rejected_replica_count > 0,
                    energy_rejected=any(record.energy_rejected for record in replica_records),
                    redundancy_requested=requested_replica_count > 1,
                    is_redundant_task=is_redundant_task,
                    backup_succeeded=backup_succeeded,
                    redundancy_succeeded=is_redundant_task and bool(successful_records),
                    selected_replica_role=selected_record.replica_role,
                    end_to_end_reliability=combined_reliability,
                    expected_reliability=expected_reliability,
                    satisfies_reliability=satisfies_reliability,
                    execution_failed=any(record.execution_failed for record in admitted_records),
                    transmission_failed=any(record.transmission_failed for record in admitted_records),
                    failed_due_to_reliability=task_failed_due_to_reliability,
                    transmission_energy_j=sum(record.transmission_energy_j for record in replica_records),
                    computing_energy_j=sum(record.computing_energy_j for record in replica_records),
                    total_energy_j=sum(record.total_energy_j for record in replica_records),
                    primary_replica_energy_j=sum(
                        record.total_energy_j
                        for record in replica_records
                        if record.replica_role == "primary"
                    ),
                    backup_replica_energy_j=sum(
                        record.total_energy_j
                        for record in replica_records
                        if record.replica_role.startswith("backup")
                    ),
                    cancellation_time_s=(
                        winner.finish_time_s
                        if winner is not None
                        and any(record.replica_cancelled for record in admitted_records)
                        else None
                    ),
                    cancelled_replica_count=sum(
                        int(record.replica_cancelled) for record in admitted_records
                    ),
                    cancellation_energy_saved_j=sum(
                        record.cancellation_energy_saved_j for record in admitted_records
                    ),
                    constraint_check=selected_constraint,
                )
            )

        # 取消属于未来事件，必须按物理时间顺序处理，不能依赖任务输入顺序。
        for cancellation_time_s, target_node_id, task_id in sorted(cancellation_events):
            target_node = self.get_compute_node_by_id(target_node_id)
            target_node.cancel_task(task_id, cancellation_time_s)

        if apply_pre_step_dynamics:
            self.current_slot_index += 1
        return records
