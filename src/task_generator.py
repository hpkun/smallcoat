from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entities import TaskInstance
from .entities import UAV
from .task_model import TaskModelConfig
from .task_model import sample_num_arrivals
from .task_model import sample_task


@dataclass
class TaskGenerator:
    """Generate per-slot system tasks and assign each to an ingress UAV."""

    task_model_config: TaskModelConfig
    next_task_index: int = 0

    def generate_tasks(
        self,
        *,
        uavs: list[UAV],
        slot_length_s: float,
        current_time_s: float,
        rng: np.random.Generator,
        delay_sensitivity_lambda: float | None = None,
    ) -> list[TaskInstance]:
        tasks: list[TaskInstance] = []
        if not uavs:
            return tasks

        count = sample_num_arrivals(self.task_model_config, slot_length_s, rng)
        # Tasks originate from the ground workload. Battery state only affects
        # UAV association and must not thin the exogenous arrival process.
        serviceable_uavs = [uav for uav in uavs if uav.can_serve]
        association_pool = serviceable_uavs or uavs
        for _ in range(count):
            ingress_uav = association_pool[int(rng.integers(0, len(association_pool)))]
            task = sample_task(
                self.task_model_config,
                rng,
                delay_sensitivity_lambda=delay_sensitivity_lambda,
            )
            task_id = f"{ingress_uav.node_id}-task-{self.next_task_index}"
            self.next_task_index += 1
            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    ingress_uav_id=ingress_uav.node_id,
                    created_at_s=current_time_s,
                    task=task,
                )
            )
        return tasks
