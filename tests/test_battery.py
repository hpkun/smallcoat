from __future__ import annotations

import math
import numpy as np

from src.entities import Position
from src.entities import UAV
from src.task_generator import TaskGenerator
from src.task_model import TaskModelConfig


def _uav() -> UAV:
    return UAV(
        node_id="uav-0",
        position=Position(0.0, 0.0, 100.0),
        compute_capacity_cycles_per_s=1e9,
        battery_capacity_j=100.0,
        remaining_energy_j=100.0,
        safe_energy_ratio=0.2,
    )


def test_battery_consumption_is_clamped_and_reset_per_episode() -> None:
    uav = _uav()

    uav.consume_energy(85.0)
    assert math.isclose(uav.remaining_energy_j, 15.0)
    assert not uav.can_serve

    uav.reset_battery()
    assert math.isclose(uav.remaining_energy_j, 100.0)
    assert math.isclose(uav.episode_energy_consumed_j, 0.0)
    assert uav.can_serve


def test_restoring_cancelled_work_reduces_episode_consumption() -> None:
    uav = _uav()
    uav.consume_energy(30.0)

    uav.restore_energy(12.0)

    assert math.isclose(uav.remaining_energy_j, 82.0)
    assert math.isclose(uav.episode_energy_consumed_j, 18.0)


def test_depleted_uav_does_not_reduce_ground_task_arrivals() -> None:
    uav = _uav()
    config = TaskModelConfig(arrival_rate_tasks_per_s=10_000.0)
    powered_tasks = TaskGenerator(config).generate_tasks(
        uavs=[uav],
        slot_length_s=1.0,
        current_time_s=0.0,
        rng=np.random.default_rng(7),
    )

    uav.consume_energy(100.0)
    depleted_tasks = TaskGenerator(config).generate_tasks(
        uavs=[uav],
        slot_length_s=1.0,
        current_time_s=0.0,
        rng=np.random.default_rng(7),
    )

    assert len(depleted_tasks) == len(powered_tasks) > 0
    assert all(task.ingress_uav_id == uav.node_id for task in depleted_tasks)
