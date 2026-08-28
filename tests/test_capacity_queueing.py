from __future__ import annotations

import numpy as np

from src.communication import CommunicationModel
from src.config import SimulationConfig
from src.config import QueueCapacityConfig
from src.config import build_default_network_profiles
from src.entities import LEOSatellite
from src.entities import Position
from src.entities import TaskInstance
from src.entities import UAV
from src.environment import OffloadingAction
from src.environment import SAGINEnvironment
from src.task_generator import TaskGenerator
from src.task_model import Task
from src.task_model import TaskModelConfig


def _env(*, uav_max_workload_s: float = 0.4) -> SAGINEnvironment:
    return SAGINEnvironment(
        uavs=[
            UAV(
                "uav-0",
                Position(0.0, 0.0, 100.0),
                compute_capacity_cycles_per_s=10e9,
                execution_failure_rate=0.0,
            )
        ],
        base_stations=[],
        leo_satellite=LEOSatellite(
            "leo-0",
            Position(500.0, 500.0, 550_000.0),
            compute_capacity_cycles_per_s=50e9,
            execution_failure_rate=0.0,
        ),
        communication_model=CommunicationModel(),
        network_profiles=build_default_network_profiles(),
        task_generator=TaskGenerator(TaskModelConfig(delay_sensitivity_lambda=1.0)),
        simulation_config=SimulationConfig(
            slot_length_s=0.1,
            queue_capacity=QueueCapacityConfig(
                uav_max_workload_s=uav_max_workload_s,
            ),
        ),
        rng=np.random.default_rng(1),
        enable_redundancy=False,
    )


def _task(task_id: str, *, cycles: float, deadline_s: float) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        ingress_uav_id="uav-0",
        created_at_s=0.0,
        task=Task(
            input_size_bits=1_000_000.0,
            total_compute_cycles=cycles,
            tolerable_latency_s=deadline_s,
            parallel_efficiency=1.0,
            profit=10.0,
            expected_reliability=0.9,
        ),
    )


def _local_actions(*task_ids: str) -> dict[str, OffloadingAction]:
    return {
        task_id: OffloadingAction(
            target_node_id="uav-0",
            priority_eta=0.5,
        )
        for task_id in task_ids
    }


def test_task_can_span_multiple_slots_when_it_meets_deadline() -> None:
    env = _env()
    task = _task("task-0", cycles=2e9, deadline_s=0.5)

    records = env.step(
        slot_length_s=0.1,
        current_time_s=0.0,
        external_tasks=[task],
        actions_by_task_id=_local_actions(task.task_id),
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].compute_delay_s == 0.2
    assert records[0].constraint_check is not None
    assert records[0].constraint_check.satisfies_capacity
    assert records[0].completed_before_deadline


def test_full_queue_rejects_excess_workload_as_capacity_drop() -> None:
    env = _env(uav_max_workload_s=0.5)
    tasks = [
        _task("task-0", cycles=3e9, deadline_s=0.5),
        _task("task-1", cycles=3e9, deadline_s=0.5),
    ]

    records = env.step(
        slot_length_s=0.1,
        current_time_s=0.0,
        external_tasks=tasks,
        actions_by_task_id=_local_actions(*(task.task_id for task in tasks)),
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 2
    assert sum(record.completed_before_deadline for record in records) == 1
    assert sum(
        record.constraint_check is not None
        and not record.constraint_check.satisfies_capacity
        for record in records
    ) == 1
    assert all(
        record.constraint_check is not None
        and record.constraint_check.satisfies_deadline
        for record in records
    )


def test_admitted_queue_congestion_can_become_deadline_failure() -> None:
    env = _env(uav_max_workload_s=0.7)
    tasks = [
        _task("task-0", cycles=3e9, deadline_s=0.5),
        _task("task-1", cycles=3e9, deadline_s=0.5),
    ]

    records = env.step(
        slot_length_s=0.1,
        current_time_s=0.0,
        external_tasks=tasks,
        actions_by_task_id=_local_actions(*(task.task_id for task in tasks)),
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 2
    assert all(
        record.constraint_check is not None
        and record.constraint_check.satisfies_capacity
        for record in records
    )
    assert sum(record.completed_before_deadline for record in records) == 1
    assert sum(
        record.constraint_check is not None
        and not record.constraint_check.satisfies_deadline
        for record in records
    ) == 1


def test_queue_capacity_limits_are_selected_by_compute_layer() -> None:
    config = QueueCapacityConfig()

    assert config.limit_for("uav") == 0.40
    assert config.limit_for("bs") == 0.20
    assert config.limit_for("leo") == 0.10
