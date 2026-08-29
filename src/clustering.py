from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import AreaConfig
from .config import ClusteringConfig
from .entities import Position
from .entities import UAV


@dataclass(frozen=True)
class ClusterInfo:
    """单个簇的静态视图。"""

    cluster_id: int
    logical_agent_id: str
    head_uav_id: str
    member_uav_ids: list[str]
    centroid: Position


class KMDUCManager:
    """
    KMDUC 动态聚类管理器。

    对应论文公式：
    - (13) CH 覆盖概率
    - (14) 基于覆盖概率的簇规模期望
    - (15) 理论簇规模
    - (16) 最大覆盖概率
    - (17) 最优簇数
    - (18) 基于位置的 K-Means 目标
    """

    def __init__(
        self,
        clustering_config: ClusteringConfig,
        area_config: AreaConfig,
    ) -> None:
        self.config = clustering_config
        self.area_config = area_config
        self.cluster_infos: dict[int, ClusterInfo] = {}
        self.ch_not_center_counters: dict[int, int] = {}
        self._next_logical_agent_index = 0

    def _new_logical_agent_id(self) -> str:
        logical_agent_id = f"ch-agent-{self._next_logical_agent_index}"
        self._next_logical_agent_index += 1
        return logical_agent_id

    @staticmethod
    def _jaccard_similarity(first: ClusterInfo, second: ClusterInfo) -> float:
        first_members = set(first.member_uav_ids)
        second_members = set(second.member_uav_ids)
        union = first_members | second_members
        return float(len(first_members & second_members) / len(union)) if union else 0.0

    def _assign_logical_agent_ids(
        self,
        old_clusters: dict[int, ClusterInfo],
        new_clusters: dict[int, ClusterInfo],
    ) -> dict[int, ClusterInfo]:
        if not old_clusters:
            assigned: dict[int, ClusterInfo] = {}
            for cluster in sorted(
                new_clusters.values(),
                key=lambda item: (
                    item.centroid.x_m,
                    item.centroid.y_m,
                    item.cluster_id,
                ),
            ):
                assigned[cluster.cluster_id] = ClusterInfo(
                    cluster_id=cluster.cluster_id,
                    logical_agent_id=self._new_logical_agent_id(),
                    head_uav_id=cluster.head_uav_id,
                    member_uav_ids=cluster.member_uav_ids,
                    centroid=cluster.centroid,
                )
            return assigned

        candidates: list[tuple[float, float, str, int, int]] = []
        for old_cluster in old_clusters.values():
            for new_cluster in new_clusters.values():
                candidates.append(
                    (
                        -self._jaccard_similarity(old_cluster, new_cluster),
                        old_cluster.centroid.distance_to(new_cluster.centroid),
                        old_cluster.logical_agent_id,
                        old_cluster.cluster_id,
                        new_cluster.cluster_id,
                    )
                )
        matched_old_ids: set[int] = set()
        matched_new_ids: set[int] = set()
        logical_ids_by_new_cluster: dict[int, str] = {}
        for _, _, logical_agent_id, old_cluster_id, new_cluster_id in sorted(candidates):
            if old_cluster_id in matched_old_ids or new_cluster_id in matched_new_ids:
                continue
            matched_old_ids.add(old_cluster_id)
            matched_new_ids.add(new_cluster_id)
            logical_ids_by_new_cluster[new_cluster_id] = logical_agent_id

        assigned = {}
        for cluster_id, cluster in new_clusters.items():
            logical_agent_id = logical_ids_by_new_cluster.get(cluster_id)
            if logical_agent_id is None:
                logical_agent_id = self._new_logical_agent_id()
            assigned[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                logical_agent_id=logical_agent_id,
                head_uav_id=cluster.head_uav_id,
                member_uav_ids=cluster.member_uav_ids,
                centroid=cluster.centroid,
            )
        return assigned

    def optimal_cluster_count(self, num_uavs: int) -> int:
        """按公式 (17) 计算最优簇数 c_n^*。"""

        if num_uavs <= 1:
            return 1

        kappa = num_uavs / self.area_config.area_m2
        coverage_fraction = kappa * math.pi * (self.config.communication_radius_m ** 2) / num_uavs
        coverage_fraction = min(max(coverage_fraction, 1e-9), 1.0 - 1e-9)
        threshold = min(max(self.config.coverage_threshold_pmax, 1e-9), 1.0 - 1e-9)

        raw_clusters = math.log(1.0 - threshold) / math.log(1.0 - coverage_fraction)
        return max(1, min(num_uavs, math.ceil(raw_clusters)))

    def coverage_probability(self, distance_m: float) -> float:
        """
        按论文公式 (13) 的字面表达计算覆盖概率：
            P_il = 1 / (1 + e^(-zeta * (d_il - R)))

        注意：论文正文说“距离越近，概率越大”，与该式字面符号冲突。
        这里严格按公式字面实现，以复现公式本身。
        """

        exponent = -self.config.logistic_zeta * (distance_m - self.config.communication_radius_m)
        exponent = min(max(exponent, -60.0), 60.0)
        return 1.0 / (1.0 + math.exp(exponent))

    def expected_cluster_size(self, num_uavs: int) -> float:
        """按公式 (15) 计算理论簇规模 E[N_l] = kappa * pi * R^2。"""

        if num_uavs <= 0:
            return 0.0
        kappa = num_uavs / self.area_config.area_m2
        return kappa * math.pi * (self.config.communication_radius_m ** 2)

    def expected_cluster_size_from_heads(
        self,
        uav: UAV,
        head_uavs: list[UAV],
    ) -> float:
        """按公式 (14) 计算 E[N_l] = sum_i P_il 的覆盖概率求和形式。"""

        return float(
            sum(
                self.coverage_probability(uav.position.distance_to(head_uav.position))
                for head_uav in head_uavs
            )
        )

    def maximum_coverage_probability(
        self,
        uav: UAV,
        head_uavs: list[UAV],
    ) -> float:
        """按公式 (16) 计算 P_max = 1 - prod_l (1 - P_il)。"""

        if not head_uavs:
            return 0.0

        product_term = 1.0
        for head_uav in head_uavs:
            pil = self.coverage_probability(uav.position.distance_to(head_uav.position))
            product_term *= (1.0 - pil)
        return 1.0 - product_term

    def kmeans_objective(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """按公式 (18) 计算位置距离和目标函数。"""

        total_distance = 0.0
        for index, point in enumerate(points):
            centroid = centroids[int(labels[index])]
            total_distance += float(np.linalg.norm(point - centroid))
        return total_distance

    def run_kmeans(
        self,
        points: np.ndarray,
        cluster_count: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用 numpy 实现位置 K-Means。"""

        cluster_count = max(1, min(cluster_count, len(points)))
        initial_indices = rng.choice(len(points), size=cluster_count, replace=False)
        centroids = points[initial_indices].copy()

        for _ in range(self.config.kmeans_max_iterations):
            distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for cluster_id in range(cluster_count):
                cluster_points = points[labels == cluster_id]
                if len(cluster_points) > 0:
                    new_centroids[cluster_id] = cluster_points.mean(axis=0)

            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        return labels, centroids

    def centralized_clustering(
        self,
        uavs: list[UAV],
        rng: np.random.Generator,
    ) -> dict[int, ClusterInfo]:
        """执行集中式聚类，确定初始 CH 和 CM 分配。"""

        if not uavs:
            self.cluster_infos = {}
            self.ch_not_center_counters = {}
            return {}

        points = np.array([[uav.position.x_m, uav.position.y_m] for uav in uavs], dtype=float)
        cluster_count = self.optimal_cluster_count(len(uavs))
        labels, centroids = self.run_kmeans(points, cluster_count, rng)

        old_cluster_infos = self.cluster_infos
        cluster_infos: dict[int, ClusterInfo] = {}
        for cluster_id in range(cluster_count):
            member_indices = np.where(labels == cluster_id)[0]
            if len(member_indices) == 0:
                continue

            centroid_xy = centroids[cluster_id]
            centroid = Position(float(centroid_xy[0]), float(centroid_xy[1]), 0.0)
            closest_index = min(
                member_indices.tolist(),
                key=lambda index: math.dist(points[index].tolist(), centroid_xy.tolist()),
            )
            head_uav = uavs[closest_index]
            member_uav_ids = [uavs[index].node_id for index in member_indices.tolist()]

            cluster_infos[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                logical_agent_id="",
                head_uav_id=head_uav.node_id,
                member_uav_ids=member_uav_ids,
                centroid=centroid,
            )

        self.cluster_infos = self._assign_logical_agent_ids(
            old_cluster_infos, cluster_infos
        )
        self.ch_not_center_counters = {
            cluster_id: 0 for cluster_id in self.cluster_infos
        }
        self._apply_cluster_infos(uavs)
        return self.cluster_infos

    def maintain_clusters(self, uavs: list[UAV]) -> dict[int, ClusterInfo]:
        """
        执行分布式簇维护。

        严格按照论文文字：
        - UAV 接收多个 CH 的广播
        - 计算覆盖概率
        - 选择覆盖概率最大的 CH
        - 若未收到任何 CH 信息，则进入孤立状态
        - CH 连续 t_ele 个时隙不是簇中心，则重选 CH
        """

        if not self.cluster_infos:
            return {}

        uav_by_id = {uav.node_id: uav for uav in uavs}

        for uav in uavs:
            best_cluster_id = None
            best_probability = -1.0
            for cluster in self.cluster_infos.values():
                head_uav = uav_by_id.get(cluster.head_uav_id)
                if head_uav is None:
                    continue
                probability = self.coverage_probability(uav.position.distance_to(head_uav.position))
                if probability > best_probability:
                    best_probability = probability
                    best_cluster_id = cluster.cluster_id

            if best_cluster_id is None:
                uav.cluster_id = None
                uav.head_uav_id = None
                uav.is_cluster_head = False
                uav.is_isolated = True
                continue

            uav.cluster_id = best_cluster_id
            uav.head_uav_id = self.cluster_infos[best_cluster_id].head_uav_id
            uav.is_cluster_head = uav.node_id == self.cluster_infos[best_cluster_id].head_uav_id
            uav.is_isolated = False

        rebuilt_members: dict[int, list[str]] = {cluster_id: [] for cluster_id in self.cluster_infos}
        for uav in uavs:
            if uav.cluster_id is not None and uav.cluster_id in rebuilt_members:
                rebuilt_members[uav.cluster_id].append(uav.node_id)

        updated_cluster_infos: dict[int, ClusterInfo] = {}
        for cluster_id, cluster in self.cluster_infos.items():
            member_uav_ids = rebuilt_members.get(cluster_id, [])
            if not member_uav_ids:
                continue

            member_positions = np.array(
                [[uav_by_id[uav_id].position.x_m, uav_by_id[uav_id].position.y_m] for uav_id in member_uav_ids],
                dtype=float,
            )
            centroid_xy = member_positions.mean(axis=0)
            centroid = Position(float(centroid_xy[0]), float(centroid_xy[1]), 0.0)

            center_uav_id = min(
                member_uav_ids,
                key=lambda uav_id: math.dist(
                    [uav_by_id[uav_id].position.x_m, uav_by_id[uav_id].position.y_m],
                    centroid_xy.tolist(),
                ),
            )

            head_uav_id = cluster.head_uav_id
            if head_uav_id != center_uav_id:
                self.ch_not_center_counters[cluster_id] = self.ch_not_center_counters.get(cluster_id, 0) + 1
                if self.ch_not_center_counters[cluster_id] >= self.config.ch_reselection_slots:
                    head_uav_id = center_uav_id
                    self.ch_not_center_counters[cluster_id] = 0
            else:
                self.ch_not_center_counters[cluster_id] = 0

            updated_cluster_infos[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                logical_agent_id=cluster.logical_agent_id,
                head_uav_id=head_uav_id,
                member_uav_ids=member_uav_ids,
                centroid=centroid,
            )

        self.cluster_infos = updated_cluster_infos
        self._apply_cluster_infos(uavs)
        return updated_cluster_infos

    def _apply_cluster_infos(self, uavs: list[UAV]) -> None:
        """将聚类结果回写到 UAV 实体。"""

        membership: dict[str, tuple[int, str]] = {}
        for cluster_id, cluster in self.cluster_infos.items():
            for uav_id in cluster.member_uav_ids:
                membership[uav_id] = (cluster_id, cluster.head_uav_id)

        for uav in uavs:
            member_info = membership.get(uav.node_id)
            if member_info is None:
                uav.cluster_id = None
                uav.head_uav_id = None
                uav.is_cluster_head = False
                uav.is_isolated = True
                continue

            cluster_id, head_uav_id = member_info
            uav.cluster_id = cluster_id
            uav.head_uav_id = head_uav_id
            uav.is_cluster_head = uav.node_id == head_uav_id
            uav.is_isolated = False

    def get_head_uav_id(self, uav_id: str) -> str | None:
        """返回某个 UAV 当前所属簇的簇头。"""

        for cluster in self.cluster_infos.values():
            if uav_id in cluster.member_uav_ids:
                return cluster.head_uav_id
        return None

    def get_logical_agent_id(self, uav_id: str) -> str | None:
        for cluster in self.cluster_infos.values():
            if uav_id in cluster.member_uav_ids:
                return cluster.logical_agent_id
        return None

    def get_logical_agent_id_by_head(self, head_uav_id: str) -> str | None:
        for cluster in self.cluster_infos.values():
            if cluster.head_uav_id == head_uav_id:
                return cluster.logical_agent_id
        return None

    def active_agent_bindings(self) -> dict[str, str]:
        return {
            cluster.logical_agent_id: cluster.head_uav_id
            for cluster in self.cluster_infos.values()
        }
