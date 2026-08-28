from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from src.communication import CommunicationModel
from src.config import SimulationConfig
from src.config import build_default_network_profiles
from src.entities import BaseStation
from src.entities import LEOSatellite
from src.entities import Position
from src.entities import TaskInstance
from src.entities import UAV
from src.environment import OffloadingAction
from src.environment import SAGINEnvironment
from src.task_generator import TaskGenerator
from src.task_model import Task
from src.task_model import TaskModelConfig


def _env(*, enable_redundancy: bool = True) -> SAGINEnvironment:
    config = SimulationConfig(slot_length_s=1.0)
    return SAGINEnvironment(
        uavs=[
            UAV(
                "uav-0",
                Position(0.0, 0.0, 100.0),
                compute_capacity_cycles_per_s=10e9,
                execution_failure_rate=0.1,
            ),
        ],
        base_stations=[
            BaseStation(
                "bs-0",
                Position(100.0, 0.0, 0.0),
                compute_capacity_cycles_per_s=20e9,
                execution_failure_rate=0.05,
            ),
        ],
        leo_satellite=LEOSatellite(
            "leo-0",
            Position(500.0, 500.0, 550_000.0),
            compute_capacity_cycles_per_s=50e9,
            execution_failure_rate=0.02,
        ),
        communication_model=CommunicationModel(),
        network_profiles=build_default_network_profiles(),
        task_generator=TaskGenerator(TaskModelConfig(delay_sensitivity_lambda=1.0)),
        simulation_config=config,
        rng=np.random.default_rng(1),
        redundancy_priority_threshold=0.7,
        enable_redundancy=enable_redundancy,
    )


def _task(task_id: str) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        ingress_uav_id="uav-0",
        created_at_s=0.0,
        task=Task(
            input_size_bits=1_000_000.0,
            total_compute_cycles=10_000_000.0,
            tolerable_latency_s=2.0,
            parallel_efficiency=1.0,
            profit=10.0,
            expected_reliability=0.99999,
        ),
    )


def test_important_uav_primary_gets_base_station_backup() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].is_redundant_task
    assert records[0].redundancy_requested
    assert records[0].primary_target_node_id == "uav-0"
    assert records[0].backup_target_node_id == "bs-0"
    assert not records[0].backup_succeeded
    assert records[0].redundancy_succeeded


def test_important_base_station_primary_gets_leo_backup() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="bs-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].is_redundant_task
    assert records[0].redundancy_requested
    assert records[0].primary_target_node_id == "bs-0"
    assert records[0].backup_target_node_id == "leo-0"
    assert records[0].redundancy_succeeded == (
        records[0].is_redundant_task and records[0].completed_before_deadline
    )


def test_important_leo_primary_gets_base_station_backup() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="leo-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].is_redundant_task
    assert records[0].redundancy_requested
    assert records[0].primary_target_node_id == "leo-0"
    assert records[0].backup_target_node_id == "bs-0"


def test_redundant_energy_includes_primary_and_backup_replicas() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    record = records[0]
    assert record.primary_replica_energy_j > 0.0
    assert record.backup_replica_energy_j > 0.0
    assert math.isclose(
        record.total_energy_j,
        record.primary_replica_energy_j + record.backup_replica_energy_j,
    )


def test_bs_reachability_uses_ingress_uav_3d_position() -> None:
    env = _env()
    ingress_uav = env.uavs[0]
    decision_uav = UAV(
        "uav-ch",
        Position(1_000.0, 0.0, 100.0),
        compute_capacity_cycles_per_s=10e9,
    )
    env.base_stations.append(
        BaseStation(
            "bs-near-ch-only",
            Position(950.0, 0.0, 0.0),
            compute_capacity_cycles_per_s=20e9,
        )
    )
    env.clustering_manager = SimpleNamespace(
        config=SimpleNamespace(communication_radius_m=300.0)
    )

    target_ids = {
        node.node_id
        for node in env.iter_compute_targets(decision_uav, ingress_uav)
    }

    # bs-0 is within 3D range of the task source, but outside the CH's range.
    assert "bs-0" in target_ids
    assert "bs-near-ch-only" not in target_ids
    assert env.select_backup_target(ingress_uav, ingress_uav).node_id == "bs-0"


def test_first_success_cancels_later_backup_and_saves_energy() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    record = records[0]
    assert record.selected_replica_role == "primary"
    assert record.cancelled_replica_count == 1
    assert record.cancellation_time_s == record.finish_time_s
    assert record.cancellation_energy_saved_j > 0.0
    assert record.backup_replica_energy_j > 0.0
    assert record.computing_energy_j == record.primary_replica_energy_j


def test_low_redundancy_action_keeps_single_target_baseline() -> None:
    env = _env()

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.9,
                redundancy_eta=0.2,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert not records[0].is_redundant_task
    assert records[0].backup_target_node_id is None


def test_high_redundancy_action_does_not_copy_low_risk_primary() -> None:
    env = _env()
    env.uavs[0].execution_failure_rate = 0.0

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert not records[0].is_redundant_task
    assert records[0].backup_target_node_id is None


def test_redundancy_can_be_disabled_for_plain_offloading_baseline() -> None:
    env = _env(enable_redundancy=False)

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert not records[0].is_redundant_task
    assert records[0].backup_target_node_id is None


def test_redundant_task_uses_combined_reliability() -> None:
    env = _env()

    redundant_records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.2,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )
    single_env = _env()
    single_records = single_env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.9,
                redundancy_eta=0.2,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert redundant_records[0].end_to_end_reliability >= single_records[0].end_to_end_reliability
    assert redundant_records[0].expected_reliability == 0.99999


def test_poisson_execution_failure_can_make_on_time_task_fail() -> None:
    env = _env()
    env.uavs[0].execution_failure_rate = 1_000_000.0

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.9,
                redundancy_eta=0.2,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].execution_failed
    assert records[0].failed_due_to_reliability
    assert not records[0].completed_before_deadline
    assert records[0].realized_profit == 0.0


def test_redundancy_succeeds_when_backup_survives_primary_failure() -> None:
    env = _env()
    env.uavs[0].execution_failure_rate = 1_000_000.0
    env.base_stations[0].execution_failure_rate = 0.0

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                priority_eta=0.9,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].is_redundant_task
    assert records[0].redundancy_requested
    assert records[0].execution_failed
    assert not records[0].failed_due_to_reliability
    assert records[0].completed_before_deadline
    assert records[0].selected_replica_role == "backup"
    assert records[0].backup_succeeded
    assert records[0].redundancy_succeeded


def test_congested_requested_backup_falls_back_to_available_layer() -> None:
    env = _env()
    env.uavs[0].execution_failure_rate = 1_000_000.0
    env.base_stations[0].queue_manager.commit(
        task_id="busy-bs",
        arrival_time_s=0.0,
        service_time_s=0.13,
        priority_eta=1.0,
        current_time_s=0.0,
    )

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                backup_target_node_id="bs-0",
                priority_eta=0.9,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert records[0].is_redundant_task
    assert records[0].redundancy_requested
    assert records[0].backup_target_node_id == "leo-0"
    assert records[0].selected_replica_role == "backup"
    assert records[0].completed_before_deadline
    assert records[0].backup_succeeded


def test_unavailable_backup_capacity_keeps_primary_as_single_copy() -> None:
    env = _env()
    env.base_stations[0].queue_manager.commit(
        task_id="busy-bs",
        arrival_time_s=0.0,
        service_time_s=0.15,
        priority_eta=1.0,
        current_time_s=0.0,
    )
    env.leo_satellite.queue_manager.commit(
        task_id="busy-leo",
        arrival_time_s=0.0,
        service_time_s=0.08,
        priority_eta=1.0,
        current_time_s=0.0,
    )

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            "task-0": OffloadingAction(
                target_node_id="uav-0",
                backup_target_node_id="bs-0",
                priority_eta=0.9,
                redundancy_eta=0.9,
            ),
        },
        external_tasks=[_task("task-0")],
        apply_pre_step_dynamics=False,
    )

    assert records[0].completed_before_deadline
    assert records[0].constraint_check is not None
    assert records[0].constraint_check.satisfies_capacity
    assert records[0].redundancy_requested
    assert not records[0].is_redundant_task
    assert records[0].backup_target_node_id is None
    assert not records[0].backup_succeeded
    assert not records[0].redundancy_succeeded
