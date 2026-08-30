from __future__ import annotations

import numpy as np

from .entities import BaseStation
from .entities import LEOSatellite
from .entities import TaskInstance
from .entities import UAV
from .observation_builder import CANDIDATE_FEATURE_DIM
from .observation_builder import ObservationBuilder


class BaselineObservationBuilder(ObservationBuilder):
    """Original-CMADDPG observation without Proposed reliability/energy state."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("enable_resource_awareness", None)
        kwargs.pop("observation_profile", None)
        super().__init__(*args, observation_profile="baseline", **kwargs)

    def build_node_load_vector(
        self,
        decision_uav: UAV,
        current_time_s: float,
    ) -> np.ndarray:
        vector = super().build_node_load_vector(decision_uav, current_time_s)
        vector[4] = 0.0  # execution failure rate
        vector[5] = 0.0  # UAV battery level
        return vector

    def build_task_vector(self, task_instance: TaskInstance | None) -> np.ndarray:
        vector = super().build_task_vector(task_instance)
        vector[4] = 0.0  # expected reliability
        return vector

    def _build_candidate_vector(
        self,
        *,
        ingress_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
        task_instance: TaskInstance | None,
        current_time_s: float,
    ) -> np.ndarray:
        if task_instance is None:
            return np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
        return super()._build_candidate_vector(
            ingress_uav=ingress_uav,
            target_node=target_node,
            task_instance=task_instance,
            current_time_s=current_time_s,
        )
