from __future__ import annotations

import numpy as np

from src.communication import CommunicationModel
from src.config import QueueCapacityConfig, SimulationConfig, build_default_network_profiles
from src.entities import BaseStation, LEOSatellite, Position, TaskInstance, UAV
from src.environment import OffloadingAction, SAGINEnvironment
from src.observation_builder import CANDIDATE_FEATURE_DIM, ObservationBuilder
from src.reward import SharedRewardCalculator
from src.rl_env import CMADDPGEnv
from src.task_generator import TaskGenerator
from src.task_model import Task, TaskModelConfig


def _task(task_id: str, *, cycles: float = 10_000_000.0) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        ingress_uav_id="uav-0",
        created_at_s=0.0,
        task=Task(
            input_size_bits=1_000_000.0,
            total_compute_cycles=cycles,
            tolerable_latency_s=2.0,
            parallel_efficiency=1.0,
            profit=10.0,
            expected_reliability=0.99,
        ),
    )


def _base_env(*, queue_capacity: QueueCapacityConfig | None = None) -> SAGINEnvironment:
    config = SimulationConfig(
        slot_length_s=1.0,
        queue_capacity=queue_capacity or QueueCapacityConfig(),
    )
    uavs = [
        UAV(
            f"uav-{index}",
            Position(float(index * 100), 0.0, 100.0),
            compute_capacity_cycles_per_s=10e9,
            execution_failure_rate=0.01 * (index + 1),
        )
        for index in range(4)
    ]
    base_stations = [
        BaseStation(
            f"bs-{index}",
            Position(float(index * 100), 300.0, 0.0),
            compute_capacity_cycles_per_s=20e9,
            execution_failure_rate=0.02,
        )
        for index in range(7)
    ]
    return SAGINEnvironment(
        uavs=uavs,
        base_stations=base_stations,
        leo_satellite=LEOSatellite(
            "leo-0",
            Position(500.0, 500.0, 550_000.0),
            compute_capacity_cycles_per_s=50e9,
            execution_failure_rate=0.01,
        ),
        communication_model=CommunicationModel(),
        network_profiles=config.network_profiles,
        task_generator=TaskGenerator(TaskModelConfig(delay_sensitivity_lambda=1.0)),
        simulation_config=config,
        rng=np.random.default_rng(4),
    )


def test_candidate_slots_include_peers_bs_and_leo_with_full_features() -> None:
    base_env = _base_env()
    builder = ObservationBuilder(
        base_env.communication_model,
        base_env.network_profiles,
        area_side_length_m=5_000.0,
        energy_config=base_env.simulation_config.energy,
        queue_capacity=base_env.simulation_config.queue_capacity,
    )
    rl_env = CMADDPGEnv(base_env, builder, SharedRewardCalculator())
    task = _task("task-0")

    nodes = rl_env._build_slot_target_nodes(base_env.uavs[0], [task], [base_env.uavs[0]])[0]
    observation = builder.build_observation(
        decision_uav=base_env.uavs[0],
        ingress_uav=base_env.uavs[0],
        member_uavs=base_env.uavs,
        base_stations=base_env.base_stations,
        leo_satellite=base_env.leo_satellite,
        task_instance=task,
        candidate_nodes=[node for node in nodes if node is not None],
        target_node_order=nodes,
        current_time_s=0.0,
    )

    assert len(nodes) == 11
    assert [node.node_type for node in nodes[:4]] == ["uav", "uav", "uav", "uav"]
    assert all(node is None or node.node_type == "bs" for node in nodes[4:10])
    assert nodes[10].node_type == "leo"
    assert observation.candidate_feature_matrix.shape == (11, CANDIDATE_FEATURE_DIM)
    assert CANDIDATE_FEATURE_DIM == 14
    assert observation.candidate_feature_matrix[1, 8] > 0.0  # UAV-UAV failure intensity
    assert observation.candidate_feature_matrix[1, 10] > 0.0  # estimated reliability


def test_three_replica_action_is_executed_without_environment_gate() -> None:
    env = _base_env()
    task = _task("task-0")
    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            task.task_id: OffloadingAction(
                priority_eta=0.8,
                replica_count=3,
                replica_target_node_ids=("uav-0", "bs-0", "leo-0"),
            )
        },
        external_tasks=[task],
        apply_pre_step_dynamics=False,
    )

    assert len(records) == 1
    assert records[0].requested_replica_count == 3
    assert records[0].admitted_replica_count == 3
    assert records[0].replica_target_node_ids == ("uav-0", "bs-0", "leo-0")
    assert records[0].winner_replica_index in {0, 1, 2}
    assert records[0].end_to_end_reliability >= 0.99


def test_same_step_replicas_compete_for_real_queue_capacity() -> None:
    env = _base_env(
        queue_capacity=QueueCapacityConfig(
            uav_max_workload_s=0.4,
            bs_max_workload_s=0.15,
            leo_max_workload_s=0.1,
        )
    )
    tasks = [_task("task-0", cycles=2e9), _task("task-1", cycles=2e9)]
    actions = {
        task.task_id: OffloadingAction(
            priority_eta=0.8,
            replica_count=1,
            replica_target_node_ids=("bs-0",),
        )
        for task in tasks
    }

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id=actions,
        external_tasks=tasks,
        apply_pre_step_dynamics=False,
    )

    assert sum(record.admitted_replica_count for record in records) == 1
    assert sum(record.capacity_rejected_replica_count for record in records) == 1


def test_uav_energy_floor_is_checked_before_admission() -> None:
    env = _base_env()
    uav = env.uavs[0]
    uav.remaining_energy_j = uav.safe_energy_j + 1e-6
    task = _task("task-0")

    records = env.step(
        slot_length_s=1.0,
        current_time_s=0.0,
        actions_by_task_id={
            task.task_id: OffloadingAction(
                replica_count=1,
                replica_target_node_ids=("uav-0",),
            )
        },
        external_tasks=[task],
        apply_pre_step_dynamics=False,
    )

    assert records[0].admitted_replica_count == 0
    assert records[0].energy_rejected
    assert uav.remaining_energy_j >= uav.safe_energy_j
