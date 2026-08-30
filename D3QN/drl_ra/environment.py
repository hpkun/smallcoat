from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log
from typing import Any

import numpy as np


TASK_PROFILES = {
    "environmental": ((50, 200), (100, 500), (10, 60), (0.85, 0.90)),
    "traffic": ((500, 2000), (1000, 5000), (1, 5), (0.95, 0.99)),
    "meter": ((10, 50), (50, 200), (30, 120), (0.90, 0.95)),
    "safety": ((200, 1000), (500, 2000), (1, 3), (0.98, 0.995)),
}
TASK_NAMES = tuple(TASK_PROFILES)
TASK_PROBABILITIES = np.asarray((0.40, 0.25, 0.20, 0.15))


@dataclass(frozen=True)
class Task:
    kind: str
    data_bits: float
    cycles: float
    deadline_s: float
    reliability_required: float
    device: int


@dataclass(frozen=True)
class Candidate:
    action: int
    layer: str
    available: bool
    delay_s: float
    energy_mj: float
    reliability: float
    link_reliability: float
    node_availability: float
    queue_s: float
    required_capacity: float


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + exp(-value))


def smooth_shortfall(gap: float, temperature: float) -> float:
    """Temperature-scaled smooth positive part used as the CMDP cost.

    The literal unscaled softplus in paper Eq. (35) has a minimum near
    log(2), inconsistent with the reported budget 0.05. This normalized
    form preserves its intended smooth-shortfall interpretation.
    """
    scaled = float(np.clip(gap / temperature, -60.0, 60.0))
    return float(temperature * np.logaddexp(0.0, scaled))


class SAGINEnv:
    """Paper-aligned, lightweight SAGIN task-offloading simulator."""

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config
        self.env_cfg = config["environment"]
        self.reward_cfg = config["reward"]
        self.seed = int(config.get("seed", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.num_devices = int(self.env_cfg["num_devices"])
        self.num_edges = int(self.env_cfg["num_edges"])
        self.num_uavs = int(self.env_cfg["num_uavs"])
        self.num_satellites = int(self.env_cfg["num_satellites"])
        self.action_dim = 1 + self.num_edges + self.num_uavs + self.num_satellites
        self.state_dim = 8 + 5 * self.action_dim
        self.max_steps = int(self.env_cfg["episode_steps"])
        self.time_step_s = 1.0 / (self.num_devices * float(self.env_cfg["arrival_rate"]))
        self.current_task: Task | None = None
        self._last_candidates: list[Candidate] = []
        self._metrics: list[dict[str, float | str | int]] = []
        self.reset(seed=self.seed)

    @property
    def metrics(self) -> list[dict[str, float | str | int]]:
        return list(self._metrics)

    @property
    def candidates(self) -> list[Candidate]:
        return list(self._last_candidates)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        area = float(self.env_cfg["area_km"])
        self.device_positions = self.rng.uniform(0.0, area, size=(self.num_devices, 2))
        grid_side = int(ceil(self.num_edges**0.5))
        grid = np.linspace(area / (2 * grid_side), area - area / (2 * grid_side), grid_side)
        points = np.asarray([(x, y) for x in grid for y in grid])[: self.num_edges]
        self.edge_positions = points + self.rng.normal(0.0, 0.1, size=points.shape)
        self.uav_centers = self.rng.uniform(0.5, area - 0.5, size=(self.num_uavs, 2))
        self.uav_phases = self.rng.uniform(0.0, 2 * np.pi, size=self.num_uavs)
        self.uav_altitudes_km = self.rng.uniform(0.15, 0.30, size=self.num_uavs)
        self.uav_battery = self.rng.uniform(0.75, 1.0, size=self.num_uavs)
        self.edge_capacity = self.rng.uniform(10.0, 50.0, size=self.num_edges) * 1e9
        self.uav_capacity = self.rng.uniform(0.5, 1.5, size=self.num_uavs) * 1e9
        self.sat_capacity = self.rng.uniform(2.0, 4.0, size=self.num_satellites) * 1e9
        self.local_capacity = self.rng.uniform(0.5, 1.5, size=self.num_devices) * 1e9
        self.edge_availability = self._edge_availabilities()
        self.sat_phases = np.linspace(0.0, 1.0, self.num_satellites, endpoint=False)
        self.node_queues = np.zeros(self.action_dim, dtype=np.float64)
        self.step_count = 0
        self.current_time_s = 0.0
        self.arrival_phase = float(self.rng.uniform(0.0, 2 * np.pi))
        self._metrics = []
        self.current_task = self._sample_task()
        state, mask = self._observation()
        return state, {"action_mask": mask}

    def _edge_availabilities(self) -> np.ndarray:
        failure = self.rng.uniform(1e-5, 1e-4, self.num_edges)
        recovery = self.rng.uniform(0.1, 0.5, self.num_edges)
        return recovery / (failure + recovery)

    def _sample_task(self) -> Task:
        device = int(self.rng.integers(self.num_devices))
        kind = str(self.rng.choice(TASK_NAMES, p=TASK_PROBABILITIES))
        data_range, cycle_range, deadline_range, reliability_range = TASK_PROFILES[kind]
        return Task(
            kind=kind,
            data_bits=float(self.rng.uniform(*data_range) * 8_000.0),
            cycles=float(self.rng.uniform(*cycle_range) * 1e6),
            deadline_s=float(self.rng.uniform(*deadline_range)),
            reliability_required=float(self.rng.uniform(*reliability_range)),
            device=device,
        )

    def _uav_positions(self) -> np.ndarray:
        phase = self.uav_phases + 2 * np.pi * self.step_count / max(self.max_steps, 1)
        offsets = np.column_stack((np.cos(phase), np.sin(phase))) * 0.5
        return self.uav_centers + offsets

    def _satellite_state(self, index: int) -> tuple[bool, float]:
        window = float(self.env_cfg["satellite_window_s"])
        cycle = 2.0 * window
        elapsed = self.current_time_s
        phase_time = (elapsed + self.sat_phases[index] * cycle) % cycle
        visible = phase_time < window
        remaining = window - phase_time if visible else 0.0
        return bool(visible), float(remaining)

    def _candidate_evaluations(self, task: Task) -> list[Candidate]:
        candidates = [self._local_candidate(task)]
        position = self.device_positions[task.device]
        for index in range(self.num_edges):
            candidates.append(self._edge_candidate(task, position, index))
        for index in range(self.num_uavs):
            candidates.append(self._uav_candidate(task, position, index))
        for index in range(self.num_satellites):
            candidates.append(self._sat_candidate(task, position, index))
        return candidates

    def _delay_reliability(self, delay_s: float, deadline_s: float) -> float:
        slack = 1.0 - delay_s / max(deadline_s, 1e-9)
        return sigmoid(slack / float(self.env_cfg["tau_smooth"]))

    def _local_candidate(self, task: Task) -> Candidate:
        action = 0
        compute = task.cycles / self.local_capacity[task.device]
        queue = self.node_queues[action]
        delay = compute + queue
        energy = 1e3 * 1e-27 * self.local_capacity[task.device] ** 2 * task.cycles
        availability = 0.999
        reliability = availability * self._delay_reliability(delay, task.deadline_s)
        return Candidate(action, "local", True, delay, energy, reliability, 1.0, availability, queue, compute)

    def _edge_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + index
        distance = float(np.linalg.norm(position - self.edge_positions[index]))
        coverage = float(self.env_cfg["edge_coverage_km"])
        available = distance <= coverage
        snr_db = 28.0 - 18.0 * np.log10(max(distance, 0.05) / 0.1)
        snr = 10 ** (snr_db / 10.0)
        rate = 20e6 * np.log2(1.0 + snr)
        transmission = task.data_bits / max(rate, 1.0) + 0.005
        queue = self.node_queues[action]
        compute = task.cycles / self.edge_capacity[index]
        delay = transmission + queue + compute + 0.002
        link_rel = float(np.clip(0.92 + 0.06 * exp(-distance / coverage), 0.92, 0.98))
        node_rel = float(self.edge_availability[index])
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (0.8 * transmission + 0.05 * (queue + compute))
        return Candidate(action, "edge", available, delay, energy, reliability, link_rel, node_rel, queue, compute)

    def _uav_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + self.num_edges + index
        uav_position = self._uav_positions()[index]
        horizontal = float(np.linalg.norm(position - uav_position))
        altitude = float(self.uav_altitudes_km[index])
        elevation_deg = np.degrees(np.arctan2(altitude, max(horizontal, 1e-4)))
        p_los = 1.0 / (1.0 + 9.61 * exp(-0.16 * (elevation_deg - 9.61)))
        available = horizontal <= 5.0 and self.uav_battery[index] > 0.20
        snr_los = 10 ** ((22.0 - 4.0 * np.log10(max(horizontal, 0.05))) / 10.0)
        snr_nlos = snr_los / 20.0
        rate = 100e6 * (p_los * np.log2(1 + snr_los) + (1 - p_los) * np.log2(1 + snr_nlos))
        transmission = task.data_bits / max(rate, 1.0) + 0.010
        queue = self.node_queues[action]
        compute = task.cycles / self.uav_capacity[index]
        delay = transmission + queue + compute + 0.005
        los_rel = float(np.clip(0.92 + 0.03 * p_los, 0.85, 0.95))
        nlos_rel = 0.85
        link_rel = p_los * los_rel + (1 - p_los) * nlos_rel
        node_rel = 0.995 if available else 0.0
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (0.8 * transmission + 0.05 * (queue + compute))
        return Candidate(action, "uav", available, delay, energy, reliability, link_rel, node_rel, queue, compute)

    def _sat_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + self.num_edges + self.num_uavs + index
        visible, remaining = self._satellite_state(index)
        direct_rate = 200e3 * self.rng.uniform(0.65, 0.9)
        direct_delay = task.data_bits / direct_rate + self.rng.uniform(0.10, 0.50)
        direct_rel = self.rng.uniform(0.80, 0.90)
        relay_options: list[tuple[float, float]] = []
        uav_positions = self._uav_positions()
        for uav_index in range(self.num_uavs):
            horizontal = float(np.linalg.norm(position - uav_positions[uav_index]))
            if horizontal > 5.0 or self.uav_battery[uav_index] <= 0.20:
                continue
            uplink_rate = 100e6 * max(0.5, 1.0 - horizontal / 10.0)
            feeder_rate = 500e6 * self.rng.uniform(0.55, 0.9)
            relay_delay = task.data_bits / uplink_rate + task.data_bits / feeder_rate + self.rng.uniform(0.02, 0.20)
            relay_rel = self.rng.uniform(0.85, 0.95) * self.rng.uniform(0.88, 0.96)
            relay_options.append((relay_delay, relay_rel))
        path_delay, link_rel = direct_delay, direct_rel
        if relay_options:
            best_relay = min(relay_options, key=lambda item: item[0])
            if best_relay[0] < path_delay:
                path_delay, link_rel = best_relay
        queue = self.node_queues[action]
        compute = task.cycles / self.sat_capacity[index]
        delay = path_delay + queue + compute + 0.01
        margin = float(self.env_cfg["satellite_safety_margin_s"])
        available = visible and remaining >= delay + margin
        node_rel = 0.999 if available else 0.0
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (1.2 * path_delay + 0.05 * (queue + compute))
        return Candidate(action, "satellite", available, delay, energy, reliability, link_rel, node_rel, queue, compute)

    def _observation(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.current_task is not None
        task = self.current_task
        candidates = self._candidate_evaluations(task)
        self._last_candidates = candidates
        area = float(self.env_cfg["area_km"])
        position = self.device_positions[task.device] / area
        task_header = np.asarray(
            [
                np.log1p(task.data_bits) / 18.0,
                np.log1p(task.cycles) / 24.0,
                min(task.deadline_s / 120.0, 1.0),
                task.reliability_required,
                position[0],
                position[1],
                self.step_count / max(self.max_steps, 1),
                self.uav_battery.mean(),
            ],
            dtype=np.float32,
        )
        node_features = []
        for candidate in candidates:
            node_features.extend(
                [
                    min(candidate.delay_s / max(task.deadline_s, 1e-6), 3.0) / 3.0,
                    candidate.reliability,
                    candidate.node_availability,
                    min(candidate.queue_s / max(task.deadline_s, 1e-6), 1.0),
                    float(candidate.available),
                ]
            )
        state = np.concatenate((task_header, np.asarray(node_features, dtype=np.float32)))
        mask = np.asarray([candidate.available for candidate in candidates], dtype=bool)
        mask[0] = True
        return state, mask

    def _replica_plan(self, primary: Candidate) -> list[Candidate]:
        assert self.current_task is not None
        task = self.current_task
        if not bool(self.env_cfg["enable_redundancy"]):
            return [primary]
        if task.reliability_required <= float(self.env_cfg["replica_threshold"]):
            return [primary]
        base_rel = float(np.clip(primary.reliability, 1e-6, 1.0 - 1e-6))
        required = ceil(log(1.0 - task.reliability_required) / log(1.0 - base_rel))
        required = int(np.clip(required, 1, int(self.env_cfg["max_replicas"])))
        selected = [primary]
        alternatives = [candidate for candidate in self._last_candidates if candidate.available and candidate.action != primary.action]
        alternatives.sort(key=lambda candidate: (candidate.layer == primary.layer, -candidate.reliability, candidate.delay_s))
        for candidate in alternatives:
            if len(selected) >= required:
                break
            if bool(self.env_cfg["capacity_gating"]) and self.node_queues[candidate.action] + candidate.required_capacity > self.current_task.deadline_s:
                continue
            selected.append(candidate)
        return selected

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")
        assert self.current_task is not None
        task = self.current_task
        primary = self._last_candidates[int(action)]
        invalid = not primary.available
        if invalid:
            primary = self._last_candidates[0]
            action = 0
        replicas = self._replica_plan(primary)
        combined_reliability = 1.0 - float(np.prod([1.0 - item.reliability for item in replicas]))
        latency = min(item.delay_s for item in replicas)
        energy = sum(item.energy_mj for item in replicas)
        cost = smooth_shortfall(
            task.reliability_required - combined_reliability,
            float(self.env_cfg["cost_temperature"]),
        )
        latency_normalized = min(latency / max(task.deadline_s, 1e-6), 3.0)
        energy_normalized = min(energy / 2000.0, 3.0)
        smooth_met = sigmoid((combined_reliability - task.reliability_required) / float(self.env_cfg["tau_smooth"]))
        performance = -float(self.reward_cfg["latency"]) * latency_normalized - float(self.reward_cfg["energy"]) * energy_normalized
        reward = (
            performance
            + float(self.reward_cfg["reliability"]) * combined_reliability * smooth_met
            - float(self.reward_cfg["violation"]) * cost
            - (1.0 if invalid else 0.0)
        )
        period = float(self.env_cfg.get("arrival_period_s", 86400.0))
        amplitude = float(self.env_cfg.get("arrival_amplitude", 0.3))
        intensity = float(self.env_cfg["arrival_rate"]) * (
            1.0 + amplitude * np.sin(2 * np.pi * self.current_time_s / period + self.arrival_phase)
        )
        aggregate_rate = max(1e-6, self.num_devices * intensity)
        self.time_step_s = float(self.rng.exponential(1.0 / aggregate_rate))
        self.current_time_s += self.time_step_s
        self.node_queues = np.maximum(0.0, self.node_queues - self.time_step_s)
        for item in replicas:
            self.node_queues[item.action] += item.required_capacity
            if item.layer == "uav":
                uav_index = item.action - 1 - self.num_edges
                self.uav_battery[uav_index] = max(0.0, self.uav_battery[uav_index] - 2e-5)
        completed = latency <= task.deadline_s
        record: dict[str, float | str | int] = {
            "latency_s": latency,
            "energy_mj": energy,
            "reliability": combined_reliability,
            "required_reliability": task.reliability_required,
            "cost": cost,
            "completed": int(completed),
            "violation": int(combined_reliability < task.reliability_required),
            "replicas": len(replicas),
            "layer": primary.layer,
            "edge_utilization": float(np.mean(self.node_queues[1 : 1 + self.num_edges] > 0.0)),
            "invalid": int(invalid),
        }
        self._metrics.append(record)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        self.current_task = self._sample_task()
        next_state, next_mask = self._observation()
        info: dict[str, Any] = {**record, "action_mask": next_mask, "raw_action": int(action)}
        return next_state, float(reward), terminated, False, info

    def summary(self) -> dict[str, float]:
        if not self._metrics:
            return {}
        numeric = lambda key: np.asarray([float(row[key]) for row in self._metrics], dtype=np.float64)
        return {
            "tasks": float(len(self._metrics)),
            "tcr": float(100.0 * numeric("completed").mean()),
            "latency_ms": float(1000.0 * numeric("latency_s").mean()),
            "energy_mj": float(numeric("energy_mj").mean()),
            "reliability_pct": float(100.0 * numeric("reliability").mean()),
            "resource_utilization_pct": float(100.0 * numeric("edge_utilization").mean()),
            "cvr": float(100.0 * numeric("violation").mean()),
            "expected_cost": float(numeric("cost").mean()),
            "mean_replicas": float(numeric("replicas").mean()),
        }
