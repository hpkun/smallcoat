from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .communication import CommunicationModel, NetworkProfiles
from .config import QueueCapacityConfig
from .energy import EnergyConfig, EnergyModel
from .entities import BaseStation, LEOSatellite, TaskInstance, UAV


NODE_LOAD_DIM = 6
TASK_FEATURE_DIM = 6
CANDIDATE_FEATURE_DIM = 14
# Compatibility name retained for network and analysis code.
LINK_FEATURE_DIM = CANDIDATE_FEATURE_DIM
MAX_NEIGHBOR_LINKS = 11
OBSERVATION_INPUT_DIM = (
    NODE_LOAD_DIM + TASK_FEATURE_DIM + CANDIDATE_FEATURE_DIM * MAX_NEIGHBOR_LINKS
)


@dataclass(frozen=True)
class ObservationComponents:
    """Structured per-task observation before attention encoding."""

    node_load_vector: np.ndarray
    task_vector: np.ndarray
    link_feature_matrix: np.ndarray

    @property
    def candidate_feature_matrix(self) -> np.ndarray:
        return self.link_feature_matrix

    def flatten(self) -> np.ndarray:
        return np.concatenate(
            [
                self.node_load_vector.astype(np.float32, copy=False),
                self.task_vector.astype(np.float32, copy=False),
                self.link_feature_matrix.astype(np.float32, copy=False).reshape(-1),
            ]
        )


class ObservationBuilder:
    """Build the unified Proposed reliability/resource/energy observation."""

    def __init__(
        self,
        communication_model: CommunicationModel,
        network_profiles: NetworkProfiles,
        area_side_length_m: float,
        enable_resource_awareness: bool | None = None,
        *,
        observation_profile: str = "proposed",
        energy_config: EnergyConfig | None = None,
        queue_capacity: QueueCapacityConfig | None = None,
    ) -> None:
        if observation_profile not in {"baseline", "proposed"}:
            raise ValueError("observation_profile must be 'baseline' or 'proposed'.")
        # The former flag can still opt into Proposed, but Proposed no longer
        # requires a command-line switch to expose resource state.
        if enable_resource_awareness:
            observation_profile = "proposed"
        self.communication_model = communication_model
        self.network_profiles = network_profiles
        self.area_side_length_m = area_side_length_m
        self.observation_profile = observation_profile
        self.enable_resource_awareness = observation_profile == "proposed"
        self.energy_model = EnergyModel(energy_config)
        self.queue_capacity = queue_capacity or QueueCapacityConfig()

    def build_node_load_vector(self, decision_uav: UAV, current_time_s: float) -> np.ndarray:
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

    def _link_profile_for(self, ingress_uav: UAV, target_node: UAV | BaseStation | LEOSatellite):
        if target_node.node_id == ingress_uav.node_id:
            return None
        if isinstance(target_node, LEOSatellite):
            return self.network_profiles.uav_to_leo
        if isinstance(target_node, UAV):
            return self.network_profiles.peer_uav_profile()
        return self.network_profiles.uav_to_bs

    @staticmethod
    def _node_type_one_hot(target_node: UAV | BaseStation | LEOSatellite) -> tuple[float, ...]:
        return (
            float(isinstance(target_node, UAV)),
            float(isinstance(target_node, BaseStation)),
            float(isinstance(target_node, LEOSatellite)),
        )

    def _build_candidate_vector(
        self,
        *,
        ingress_uav: UAV,
        target_node: UAV | BaseStation | LEOSatellite,
        task_instance: TaskInstance | None,
        current_time_s: float,
    ) -> np.ndarray:
        queue = target_node.queue_snapshot(current_time_s)
        workload_limit_s = self.queue_capacity.limit_for(target_node.node_type)
        queued_workload_s = target_node.queue_manager.workload_s(current_time_s)
        remaining_capacity_ratio = np.clip(
            (workload_limit_s - queued_workload_s) / workload_limit_s,
            0.0,
            1.0,
        )
        battery_ratio = target_node.battery_level if isinstance(target_node, UAV) else 1.0
        if task_instance is None:
            return np.array(
                [
                    *self._node_type_one_hot(target_node),
                    target_node.compute_capacity_cycles_per_s / 1e11,
                    remaining_capacity_ratio,
                    0.0,
                    *np.zeros(7, dtype=np.float32),
                    battery_ratio,
                ],
                dtype=np.float32,
            )

        task = task_instance.task
        deadline_s = max(task.tolerable_latency_s, 1e-6)
        compute_delay_s = target_node.estimate_compute_delay(task)
        profile = self._link_profile_for(ingress_uav, target_node)
        if profile is None:
            rate_bps = 0.0
            transmission_s = 0.0
            propagation_s = 0.0
            transmission_failure_rate = 0.0
        else:
            rate_bps = self.communication_model.link_rate_bps(
                ingress_uav.position,
                target_node.position,
                profile,
            )
            transmission_s, propagation_s = self.communication_model.total_link_delay_s(
                task.input_size_bits,
                ingress_uav.position,
                target_node.position,
                profile,
            )
            transmission_failure_rate = profile.transmission_failure_rate
        communication_delay_s = transmission_s + propagation_s
        estimated_finish_delay_s = (
            communication_delay_s + queue.expected_total_wait_s + compute_delay_s
        )
        transmission_reliability = np.exp(
            -transmission_failure_rate * communication_delay_s
        )
        execution_reliability = np.exp(
            -target_node.execution_failure_rate * compute_delay_s
        )
        estimated_reliability = transmission_reliability * execution_reliability
        deadline_slack = np.clip(
            (deadline_s - estimated_finish_delay_s) / deadline_s,
            -5.0,
            1.0,
        )
        energy = self.energy_model.compute(
            task,
            target_node.node_type,
            transmission_s,
            profile,
        )
        proposed = np.array(
            [
                *self._node_type_one_hot(target_node),
                target_node.compute_capacity_cycles_per_s / 1e11,
                remaining_capacity_ratio,
                np.clip(queue.expected_total_wait_s / deadline_s, 0.0, 5.0),
                rate_bps / 1e9,
                communication_delay_s / deadline_s,
                transmission_failure_rate,
                target_node.execution_failure_rate,
                estimated_reliability,
                deadline_slack,
                energy.total_energy_j / 1_000.0,
                battery_ratio,
            ],
            dtype=np.float32,
        )
        if self.observation_profile == "proposed":
            return proposed

        baseline = np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
        baseline[:3] = (
            rate_bps / 1e9,
            ingress_uav.position.distance_to(target_node.position)
            / max(self.area_side_length_m, 1.0),
            transmission_failure_rate,
        )
        return baseline

    def build_link_feature_matrix(
        self,
        *,
        ingress_uav: UAV,
        candidate_nodes: list[UAV | BaseStation | LEOSatellite],
        target_node_order: list[UAV | BaseStation | LEOSatellite | None] | None = None,
        task_instance: TaskInstance | None = None,
        current_time_s: float = 0.0,
    ) -> np.ndarray:
        ordered_nodes = target_node_order if target_node_order is not None else candidate_nodes
        if len(ordered_nodes) > MAX_NEIGHBOR_LINKS:
            raise ValueError(
                f"target_node_order exceeds MAX_NEIGHBOR_LINKS: {len(ordered_nodes)}"
            )
        candidate_ids = {node.node_id for node in candidate_nodes}
        rows = [
            self._build_candidate_vector(
                ingress_uav=ingress_uav,
                target_node=target_node,
                task_instance=task_instance,
                current_time_s=current_time_s,
            )
            if target_node is not None and target_node.node_id in candidate_ids
            else np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
            for target_node in ordered_nodes
        ]
        matrix = (
            np.stack(rows, axis=0).astype(np.float32, copy=False)
            if rows
            else np.zeros((0, CANDIDATE_FEATURE_DIM), dtype=np.float32)
        )
        padding = np.zeros(
            (MAX_NEIGHBOR_LINKS - matrix.shape[0], CANDIDATE_FEATURE_DIM),
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
        del member_uavs, base_stations, leo_satellite, cluster_radius_m
        return ObservationComponents(
            node_load_vector=self.build_node_load_vector(decision_uav, current_time_s),
            task_vector=self.build_task_vector(task_instance),
            link_feature_matrix=self.build_link_feature_matrix(
                ingress_uav=ingress_uav,
                candidate_nodes=candidate_nodes,
                target_node_order=target_node_order,
                task_instance=task_instance,
                current_time_s=current_time_s,
            ),
        )

    def build_agent_state(self, **kwargs) -> np.ndarray:
        return self.build_observation(**kwargs).flatten()
