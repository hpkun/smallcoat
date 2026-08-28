from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .communication import CommunicationModel
from .communication import NetworkProfiles
from .entities import BaseStation
from .entities import LEOSatellite
from .entities import TaskInstance
from .entities import UAV


NODE_LOAD_DIM = 6
TASK_FEATURE_DIM = 6
LINK_FEATURE_DIM = 3
# 每个任务固定为 ingress UAV、Top-9 BS 和 LEO，共 11 个目标槽。
MAX_NEIGHBOR_LINKS = 11
OBSERVATION_INPUT_DIM = NODE_LOAD_DIM + TASK_FEATURE_DIM + LINK_FEATURE_DIM * MAX_NEIGHBOR_LINKS


@dataclass(frozen=True)
class ObservationComponents:
    """Attention 输入前的结构化观测。"""

    node_load_vector: np.ndarray
    task_vector: np.ndarray
    link_feature_matrix: np.ndarray

    def flatten(self) -> np.ndarray:
        return np.concatenate(
            [
                self.node_load_vector.astype(np.float32, copy=False),
                self.task_vector.astype(np.float32, copy=False),
                self.link_feature_matrix.astype(np.float32, copy=False).reshape(-1),
            ]
        )


class ObservationBuilder:
    """
    观测构造器。
    观测输入由三部分组成：
    - 当前节点负载特征 6 维
    - 任务基础特征 6 维
    - 11 个候选目标的链路特征，每个目标 3 维
    """

    def __init__(
        self,
        communication_model: CommunicationModel,
        network_profiles: NetworkProfiles,
        area_side_length_m: float,
        enable_resource_awareness: bool = False,
    ) -> None:
        self.communication_model = communication_model
        self.network_profiles = network_profiles
        self.area_side_length_m = area_side_length_m
        self.enable_resource_awareness = bool(enable_resource_awareness)

    def build_node_load_vector(
        self,
        decision_uav: UAV,
        current_time_s: float,
    ) -> np.ndarray:
        queue_snapshot = decision_uav.queue_snapshot(current_time_s)
        return np.array(
            [
                float(decision_uav.compute_capacity_cycles_per_s / 1e11),
                float(queue_snapshot.executing_queue_length / 10.0),
                float(queue_snapshot.buffer_queue_length / 20.0),
                float(queue_snapshot.expected_total_wait_s / 10.0),
                float(decision_uav.execution_failure_rate),
                float(decision_uav.battery_level),
            ],
            dtype=np.float32,
        )

    def build_task_vector(self, task_instance: TaskInstance | None) -> np.ndarray:
        if task_instance is None:
            return np.zeros(TASK_FEATURE_DIM, dtype=np.float32)

        task = task_instance.task
        return np.array(
            [
                float(task.input_size_bits / 1e8),
                float(task.total_compute_cycles / 1e10),
                float(task.cycles_per_bit / 1e3),
                float(task.tolerable_latency_s / 10.0),
                float(task.expected_reliability),
                float(task.forward_count / 10.0),
            ],
            dtype=np.float32,
        )

    def _link_profile_for(self, target_node: UAV | BaseStation | LEOSatellite):
        return (
            self.network_profiles.uav_to_leo
            if isinstance(target_node, LEOSatellite)
            else self.network_profiles.uav_to_bs
        )

    def _build_link_feature_triplet(
        self,
        *,
        ingress_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
    ) -> np.ndarray:
        profile = self._link_profile_for(target_node)

        if target_node.node_id == ingress_uav.node_id:
            return np.zeros(LINK_FEATURE_DIM, dtype=np.float32)

        rate_bps = self.communication_model.link_rate_bps(
            sender=ingress_uav.position,
            receiver=target_node.position,
            profile=profile,
        )
        distance_norm = ingress_uav.position.distance_to(target_node.position) / max(
            self.area_side_length_m,
            1.0,
        )

        return np.array(
            [
                float(rate_bps / 1e9),
                float(distance_norm),
                float(profile.transmission_failure_rate),
            ],
            dtype=np.float32,
        )

    def _build_resource_aware_triplet(
        self,
        *,
        ingress_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
        task_instance: TaskInstance | None,
        current_time_s: float,
    ) -> np.ndarray:
        """Compress target capacity, queue pressure, and deadline feasibility."""
        queue = target_node.queue_snapshot(current_time_s)
        compute_delay_s = (
            target_node.estimate_compute_delay(task_instance.task)
            if task_instance is not None
            else 0.0
        )
        if target_node.node_id == ingress_uav.node_id or task_instance is None:
            communication_delay_s = 0.0
        else:
            transmission_s, propagation_s = self.communication_model.total_link_delay_s(
                data_size_bits=task_instance.task.input_size_bits,
                sender=ingress_uav.position,
                receiver=target_node.position,
                profile=self._link_profile_for(target_node),
            )
            communication_delay_s = transmission_s + propagation_s

        estimated_finish_delay_s = (
            communication_delay_s + queue.expected_total_wait_s + compute_delay_s
        )
        deadline_s = (
            max(task_instance.task.tolerable_latency_s, 1e-6)
            if task_instance is not None
            else 1.0
        )
        deadline_slack_ratio = np.clip(
            (deadline_s - estimated_finish_delay_s) / deadline_s,
            -5.0,
            1.0,
        )
        return np.array(
            [
                float(target_node.compute_capacity_cycles_per_s / 1e11),
                float(np.clip(queue.expected_total_wait_s / deadline_s, 0.0, 5.0)),
                float(deadline_slack_ratio),
            ],
            dtype=np.float32,
        )

    def build_link_feature_matrix(
        self,
        *,
        ingress_uav: UAV,
        candidate_nodes: list[UAV | BaseStation | LEOSatellite],
        target_node_order: list[UAV | BaseStation | LEOSatellite | None] | None = None,
    ) -> np.ndarray:
        if target_node_order is not None:
            if len(target_node_order) > MAX_NEIGHBOR_LINKS:
                raise ValueError(
                    "target_node_order exceeds MAX_NEIGHBOR_LINKS: "
                    f"{len(target_node_order)} > {MAX_NEIGHBOR_LINKS}"
                )
            candidate_ids = {node.node_id for node in candidate_nodes}
            rows = [
                self._build_link_feature_triplet(
                    ingress_uav=ingress_uav,
                    target_node=target_node,
                )
                if target_node is not None and target_node.node_id in candidate_ids
                else np.zeros(LINK_FEATURE_DIM, dtype=np.float32)
                for target_node in target_node_order
            ]
            matrix = (
                np.stack(rows, axis=0).astype(np.float32, copy=False)
                if rows
                else np.zeros((0, LINK_FEATURE_DIM), dtype=np.float32)
            )
        elif not candidate_nodes:
            return np.zeros((MAX_NEIGHBOR_LINKS, LINK_FEATURE_DIM), dtype=np.float32)
        else:
            matrix = np.stack(
                [
                    self._build_link_feature_triplet(
                        ingress_uav=ingress_uav,
                        target_node=target_node,
                    )
                    for target_node in candidate_nodes
                ],
                axis=0,
            ).astype(np.float32, copy=False)

        if matrix.shape[0] >= MAX_NEIGHBOR_LINKS:
            return matrix[:MAX_NEIGHBOR_LINKS]

        padding = np.zeros((MAX_NEIGHBOR_LINKS - matrix.shape[0], LINK_FEATURE_DIM), dtype=np.float32)
        return np.concatenate([matrix, padding], axis=0)

    def _build_aligned_resource_matrix(
        self,
        *,
        ingress_uav: UAV,
        task_instance: TaskInstance | None,
        current_time_s: float,
        candidate_nodes: list[UAV | BaseStation | LEOSatellite],
        target_node_order: list[UAV | BaseStation | LEOSatellite | None] | None = None,
    ) -> np.ndarray:
        ordered_nodes = target_node_order if target_node_order is not None else candidate_nodes
        if len(ordered_nodes) > MAX_NEIGHBOR_LINKS:
            raise ValueError(
                "target_node_order exceeds MAX_NEIGHBOR_LINKS: "
                f"{len(ordered_nodes)} > {MAX_NEIGHBOR_LINKS}"
            )
        candidate_ids = {node.node_id for node in candidate_nodes}
        rows = [
            self._build_resource_aware_triplet(
                ingress_uav=ingress_uav,
                target_node=target_node,
                task_instance=task_instance,
                current_time_s=current_time_s,
            )
            if target_node is not None and target_node.node_id in candidate_ids
            else np.zeros(LINK_FEATURE_DIM, dtype=np.float32)
            for target_node in ordered_nodes
        ]
        matrix = (
            np.stack(rows, axis=0).astype(np.float32, copy=False)
            if rows
            else np.zeros((0, LINK_FEATURE_DIM), dtype=np.float32)
        )
        if matrix.shape[0] >= MAX_NEIGHBOR_LINKS:
            return matrix[:MAX_NEIGHBOR_LINKS]
        padding = np.zeros(
            (MAX_NEIGHBOR_LINKS - matrix.shape[0], LINK_FEATURE_DIM),
            dtype=np.float32,
        )
        return np.concatenate([matrix, padding], axis=0)

    def build_observation(
        self,
        *,
        decision_uav: UAV,
        ingress_uav: UAV,
        member_uavs: list[UAV],
        base_stations: list[BaseStation],
        leo_satellite: LEOSatellite,
        task_instance: TaskInstance | None,
        candidate_nodes: list[UAV | BaseStation | LEOSatellite],
        target_node_order: list[UAV | BaseStation | LEOSatellite | None] | None = None,
        current_time_s: float,
        cluster_radius_m: float | None = None,
    ) -> ObservationComponents:
        return ObservationComponents(
            node_load_vector=self.build_node_load_vector(
                decision_uav=decision_uav,
                current_time_s=current_time_s,
            ),
            task_vector=self.build_task_vector(task_instance),
            link_feature_matrix=(
                self._build_aligned_resource_matrix(
                    ingress_uav=ingress_uav,
                    task_instance=task_instance,
                    current_time_s=current_time_s,
                    candidate_nodes=candidate_nodes,
                    target_node_order=target_node_order,
                )
                if self.enable_resource_awareness
                else self.build_link_feature_matrix(
                    ingress_uav=ingress_uav,
                    candidate_nodes=candidate_nodes,
                    target_node_order=target_node_order,
                )
            ),
        )

    def build_agent_state(self, **kwargs) -> np.ndarray:
        return self.build_observation(**kwargs).flatten()
