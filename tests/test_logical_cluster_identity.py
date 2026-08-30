from __future__ import annotations

import numpy as np

from src.clustering import KMDUCManager
from src.config import AreaConfig
from src.config import ClusteringConfig
from src.entities import Position
from src.entities import UAV


def _uav(node_id: str, x_m: float, y_m: float = 0.0) -> UAV:
    return UAV(
        node_id=node_id,
        position=Position(x_m, y_m, 100.0),
        compute_capacity_cycles_per_s=10e9,
    )


def _manager(*, ch_reselection_slots: int = 3) -> KMDUCManager:
    return KMDUCManager(
        clustering_config=ClusteringConfig(
            ch_reselection_slots=ch_reselection_slots
        ),
        area_config=AreaConfig(side_length_m=1_000.0),
    )


def test_initial_logical_ids_follow_centroid_order_not_kmeans_labels(monkeypatch) -> None:
    manager = _manager()
    uavs = [_uav("uav-0", 0.0), _uav("uav-1", 10.0), _uav("uav-2", 100.0), _uav("uav-3", 110.0)]
    monkeypatch.setattr(manager, "optimal_cluster_count", lambda _count: 2)
    monkeypatch.setattr(
        manager,
        "run_kmeans",
        lambda _points, _count, _rng: (
            np.asarray([1, 1, 0, 0]),
            np.asarray([[105.0, 0.0], [5.0, 0.0]]),
        ),
    )

    clusters = manager.centralized_clustering(uavs, np.random.default_rng(1))

    assert clusters[1].logical_agent_id == "ch-agent-0"
    assert clusters[0].logical_agent_id == "ch-agent-1"
    assert manager.get_logical_agent_id("uav-0") == "ch-agent-0"
    assert (
        manager.get_logical_agent_id_by_head(clusters[1].head_uav_id)
        == "ch-agent-0"
    )
    assert manager.active_agent_bindings()["ch-agent-0"] == clusters[1].head_uav_id


def test_reclustering_preserves_logical_ids_when_kmeans_labels_swap(monkeypatch) -> None:
    manager = _manager()
    uavs = [_uav("uav-0", 0.0), _uav("uav-1", 10.0), _uav("uav-2", 100.0), _uav("uav-3", 110.0)]
    monkeypatch.setattr(manager, "optimal_cluster_count", lambda _count: 2)
    results = iter(
        [
            (
                np.asarray([0, 0, 1, 1]),
                np.asarray([[5.0, 0.0], [105.0, 0.0]]),
            ),
            (
                np.asarray([1, 1, 0, 0]),
                np.asarray([[105.0, 0.0], [5.0, 0.0]]),
            ),
        ]
    )
    monkeypatch.setattr(
        manager,
        "run_kmeans",
        lambda _points, _count, _rng: next(results),
    )

    first = manager.centralized_clustering(uavs, np.random.default_rng(2))
    first_ids = {
        frozenset(cluster.member_uav_ids): cluster.logical_agent_id
        for cluster in first.values()
    }
    second = manager.centralized_clustering(uavs, np.random.default_rng(3))
    second_ids = {
        frozenset(cluster.member_uav_ids): cluster.logical_agent_id
        for cluster in second.values()
    }

    assert second_ids == first_ids
    assert second[1].logical_agent_id == first[0].logical_agent_id
    assert second[0].logical_agent_id == first[1].logical_agent_id


def test_physical_head_reselection_preserves_logical_agent(monkeypatch) -> None:
    manager = _manager(ch_reselection_slots=1)
    uavs = [_uav("uav-0", 0.0), _uav("uav-1", 10.0), _uav("uav-2", 20.0)]
    monkeypatch.setattr(manager, "optimal_cluster_count", lambda _count: 1)
    monkeypatch.setattr(
        manager,
        "run_kmeans",
        lambda _points, _count, _rng: (
            np.asarray([0, 0, 0]),
            np.asarray([[10.0, 0.0]]),
        ),
    )
    initial = manager.centralized_clustering(uavs, np.random.default_rng(4))[0]
    assert initial.head_uav_id == "uav-1"

    uavs[1].position = Position(100.0, 0.0, 100.0)
    updated = manager.maintain_clusters(uavs)[0]

    assert updated.head_uav_id == "uav-2"
    assert updated.logical_agent_id == initial.logical_agent_id
    assert manager.active_agent_bindings() == {
        initial.logical_agent_id: "uav-2"
    }


def test_empty_cluster_then_reclustering_reuses_persistent_role_pool(monkeypatch) -> None:
    manager = _manager()
    uavs = [
        _uav("uav-0", 0.0),
        _uav("uav-1", 10.0),
        _uav("uav-2", 100.0),
        _uav("uav-3", 110.0),
    ]
    monkeypatch.setattr(manager, "optimal_cluster_count", lambda _count: 2)
    monkeypatch.setattr(
        manager,
        "run_kmeans",
        lambda _points, _count, _rng: (
            np.asarray([0, 0, 1, 1]),
            np.asarray([[5.0, 0.0], [105.0, 0.0]]),
        ),
    )

    manager.centralized_clustering(uavs, np.random.default_rng(5))
    expected_ids = {"ch-agent-0", "ch-agent-1"}

    for iteration in range(100):
        # Reproduce the failure mode: maintenance has temporarily lost one
        # cluster before centralized clustering restores K clusters.
        retained_cluster_id = min(manager.cluster_infos)
        manager.cluster_infos = {
            retained_cluster_id: manager.cluster_infos[retained_cluster_id]
        }
        manager.ch_not_center_counters = {retained_cluster_id: 0}
        clusters = manager.centralized_clustering(
            uavs, np.random.default_rng(100 + iteration)
        )

        assert {cluster.logical_agent_id for cluster in clusters.values()} == expected_ids
        assert set(manager.logical_agent_ids) == expected_ids


def test_training_scenario_has_exactly_twelve_persistent_roles() -> None:
    from train import build_training_env

    env = build_training_env(seed=42)
    manager = env.base_env.clustering_manager

    assert manager is not None
    assert manager.optimal_cluster_count(len(env.base_env.uavs)) == 12
    assert manager.logical_agent_ids == tuple(f"ch-agent-{index}" for index in range(12))
