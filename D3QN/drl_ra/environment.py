from __future__ import annotations

from dataclasses import dataclass
from math import ceil, erfc, exp, log, pi, sqrt
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
    distance_km: float = float("inf")
    relay_uav: int | None = None


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + exp(-value))


def q_function(value: float) -> float:
    return 0.5 * erfc(float(value) / sqrt(2.0))


def smooth_shortfall(gap: float, temperature: float) -> float:
    """Temperature-scaled smooth positive part used for the CMDP cost."""
    scaled = float(np.clip(gap / max(temperature, 1e-9), -60.0, 60.0))
    return float(temperature * np.logaddexp(0.0, scaled))


class SAGINEnv:
    """SAGIN task-offloading environment with explicit resource accounting."""

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

    @property
    def local_queues(self) -> np.ndarray:
        return self.local_queue_s.copy()

    @property
    def available_capacity(self) -> np.ndarray:
        return np.maximum(0.0, self.total_capacity - self.used_capacity - self.reserved_capacity)

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
        self.uav_phases = self.rng.uniform(0.0, 2 * pi, size=self.num_uavs)
        self.uav_speeds_mps = self.rng.uniform(5.0, 15.0, size=self.num_uavs)
        self.uav_radius_km = self.rng.uniform(0.35, 0.60, size=self.num_uavs)
        self.uav_altitudes_km = self.rng.uniform(0.15, 0.30, size=self.num_uavs)
        self.uav_battery = self.rng.uniform(0.75, 1.0, size=self.num_uavs)
        self.edge_availability = self._edge_availabilities()
        self.sat_phases = np.linspace(0.0, 1.0, self.num_satellites, endpoint=False)
        self.device_types = self._fixed_device_types()
        self.total_capacity = np.concatenate((np.zeros(1), self.rng.uniform(10.0, 50.0, size=self.num_edges) * 1e9, self.rng.uniform(0.5, 1.5, size=self.num_uavs) * 1e9, self.rng.uniform(2.0, 4.0, size=self.num_satellites) * 1e9))
        self.used_capacity = np.zeros(self.action_dim, dtype=np.float64)
        self.reserved_capacity = np.zeros(self.action_dim, dtype=np.float64)
        self.local_capacity = self.rng.uniform(0.5, 1.5, size=self.num_devices) * 1e9
        self.node_queues = np.zeros(self.action_dim, dtype=np.float64)
        self.local_queue_s = np.zeros(self.num_devices, dtype=np.float64)
        self.relay_arrival_rate = np.zeros(self.num_uavs, dtype=np.float64)
        self.active_allocations: list[tuple[float, int, float]] = []
        self.step_count = 0
        self.current_time_s = 0.0
        self.time_step_s = 1.0 / float(self.env_cfg["arrival_rate"])
        self.arrival_phase = float(self.rng.uniform(0.0, 2 * pi))
        self._metrics = []
        self.current_task = self._sample_task()
        state, mask = self._observation()
        return state, {"action_mask": mask}

    def _fixed_device_types(self) -> np.ndarray:
        counts = np.floor(self.num_devices * TASK_PROBABILITIES).astype(int)
        counts[-1] += self.num_devices - int(counts.sum())
        values = np.concatenate([np.repeat(TASK_NAMES[index], count) for index, count in enumerate(counts)])
        self.rng.shuffle(values)
        return values

    def _edge_availabilities(self) -> np.ndarray:
        failure = self.rng.uniform(1e-5, 1e-4, self.num_edges)
        recovery = self.rng.uniform(0.1, 0.5, self.num_edges)
        return recovery / (failure + recovery)

    def _sample_task(self) -> Task:
        device = int(self.rng.integers(self.num_devices))
        kind = str(self.device_types[device])
        data_range, cycle_range, deadline_range, reliability_range = TASK_PROFILES[kind]
        required = self.env_cfg.get("reliability_requirement_override")
        reliability = float(required) if required is not None else float(self.rng.uniform(*reliability_range))
        return Task(kind, float(self.rng.uniform(*data_range) * 8_000.0), float(self.rng.uniform(*cycle_range) * 1e6), float(self.rng.uniform(*deadline_range)), reliability, device)

    def _uav_positions(self, time_s: float | None = None) -> np.ndarray:
        time_s = self.current_time_s if time_s is None else float(time_s)
        angle = self.uav_phases + self.uav_speeds_mps * time_s / (self.uav_radius_km * 1000.0)
        return self.uav_centers + np.column_stack((np.cos(angle), np.sin(angle))) * self.uav_radius_km[:, None]

    def _satellite_state(self, index: int) -> tuple[bool, float]:
        window = float(self.env_cfg["satellite_window_s"])
        phase_time = (self.current_time_s + self.sat_phases[index] * 2.0 * window) % (2.0 * window)
        return bool(phase_time < window), float(max(0.0, window - phase_time))

    def _queue_for(self, action: int, device: int) -> float:
        return float(self.local_queue_s[device] if action == 0 else self.node_queues[action])

    def _required_capacity(self, task: Task) -> float:
        return task.cycles / max(task.deadline_s, 1e-6)

    def _has_capacity(self, action: int, required: float) -> bool:
        if not bool(self.env_cfg.get("capacity_gating", True)):
            return True
        return action == 0 or self.available_capacity[action] >= required

    def _service_time(self, candidate: Candidate, task: Task) -> float:
        if candidate.action == 0:
            return task.cycles / self.local_capacity[task.device]
        return task.cycles / self.total_capacity[candidate.action]

    def _delay_reliability(self, delay_s: float, deadline_s: float) -> float:
        return sigmoid((1.0 - delay_s / max(deadline_s, 1e-9)) / float(self.env_cfg["tau_smooth"]))

    def _local_candidate(self, task: Task) -> Candidate:
        compute = task.cycles / self.local_capacity[task.device]
        queue = self._queue_for(0, task.device)
        delay = compute + queue
        energy = 1e3 * 1e-27 * self.local_capacity[task.device] ** 2 * task.cycles
        reliability = 0.999 * self._delay_reliability(delay, task.deadline_s)
        return Candidate(0, "local", True, delay, energy, reliability, 1.0, 0.999, queue, self._required_capacity(task), 0.0)

    def _edge_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + index
        distance = float(np.linalg.norm(position - self.edge_positions[index]))
        coverage = float(self.env_cfg["edge_coverage_km"])
        snr_db = 28.0 - 18.0 * np.log10(max(distance, 0.05) / 0.1)
        rate = 20e6 * np.log2(1.0 + 10 ** (snr_db / 10.0))
        transmission = task.data_bits / max(rate, 1.0) + 0.005
        queue = self._queue_for(action, task.device)
        compute = task.cycles / self.total_capacity[action]
        delay = transmission + queue + compute + 0.002
        path_loss_db = 128.1 + 37.6 * np.log10(max(distance, 1e-3))
        link_snr_linear = 10 ** ((46.0 - path_loss_db) / 10.0)
        link_rel = float(np.clip(1.0 - q_function(sqrt(2.0 * link_snr_linear / 0.01)), 0.92, 0.98))
        node_rel = float(self.edge_availability[index])
        required = self._required_capacity(task)
        available = distance <= coverage and self._has_capacity(action, required)
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (0.8 * transmission + 0.05 * (queue + compute))
        return Candidate(action, "edge", available, delay, energy, reliability, link_rel, node_rel, queue, required, distance)

    def _uav_link(self, position: np.ndarray, index: int, data_bits: float) -> tuple[float, float, float, bool]:
        distance = float(np.linalg.norm(position - self._uav_positions()[index]))
        elevation = np.degrees(np.arctan2(self.uav_altitudes_km[index], max(distance, 1e-4)))
        p_los = 1.0 / (1.0 + 9.61 * exp(-0.16 * (elevation - 9.61)))
        snr_los = 10 ** ((22.0 - 4.0 * np.log10(max(distance, 0.05))) / 10.0)
        snr_nlos = snr_los / 20.0
        rate = 100e6 * (p_los * np.log2(1 + snr_los) + (1 - p_los) * np.log2(1 + snr_nlos))
        link_rel = float(np.clip(p_los * 0.95 + (1 - p_los) * 0.85, 0.01, 0.99))
        return data_bits / max(rate, 1.0) + 0.010, link_rel, distance, distance <= 5.0

    def _uav_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + self.num_edges + index
        transmission, link_rel, distance, in_range = self._uav_link(position, index, task.data_bits)
        queue = self._queue_for(action, task.device)
        compute = task.cycles / self.total_capacity[action]
        delay = transmission + queue + compute + 0.005
        battery_ok = self.uav_battery[index] > 0.20
        node_rel = 0.995 if battery_ok else 0.0
        required = self._required_capacity(task)
        available = in_range and battery_ok and self._has_capacity(action, required)
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (0.8 * transmission + 0.05 * (queue + compute))
        return Candidate(action, "uav", available, delay, energy, reliability, link_rel, node_rel, queue, required, distance)

    def _sat_candidate(self, task: Task, position: np.ndarray, index: int) -> Candidate:
        action = 1 + self.num_edges + self.num_uavs + index
        visible, remaining = self._satellite_state(index)
        direct_rate = 200e3 * self.rng.uniform(0.65, 0.9)
        direct_delay = task.data_bits / direct_rate + self.rng.uniform(0.10, 0.50)
        rain_db = self.rng.uniform(0.0, 6.0)
        scintillation = self.rng.uniform(0.97, 1.0)
        direct_rel = float(np.exp(-rain_db / 10.0) * scintillation)
        relay_options: list[tuple[float, float, int]] = []
        for uav_index in range(self.num_uavs):
            uplink, uplink_rel, _, in_range = self._uav_link(position, uav_index, task.data_bits)
            if not in_range or self.uav_battery[uav_index] <= 0.20:
                continue
            feeder_rate = 500e6 * self.rng.uniform(0.55, 0.9)
            active = max(1.0, float(np.count_nonzero(self.relay_arrival_rate > 0.0)))
            effective_rate = feeder_rate * (1.0 - 0.2 * min(active / max(self.num_uavs, 1), 1.0))
            service_rate = max(effective_rate / max(task.data_bits, 1.0), 1e-6)
            relay_queue = 1.0 / max(service_rate - self.relay_arrival_rate[uav_index], 1e-6)
            relay_delay = uplink + task.data_bits / effective_rate + relay_queue + self.rng.uniform(0.02, 0.20)
            relay_options.append((relay_delay, uplink_rel * float(np.exp(-rain_db / 12.0) * scintillation), uav_index))
        path_delay, link_rel = direct_delay, direct_rel
        relay_uav = None
        if relay_options and min(relay_options, key=lambda item: item[0])[0] < path_delay:
            path_delay, link_rel, relay_uav = min(relay_options, key=lambda item: item[0])
        queue = self._queue_for(action, task.device)
        compute = task.cycles / self.total_capacity[action]
        delay = path_delay + queue + compute + 0.01
        required = self._required_capacity(task)
        available = visible and remaining >= delay + float(self.env_cfg["satellite_safety_margin_s"]) and self._has_capacity(action, required)
        node_rel = 0.999 if available else 0.0
        reliability = link_rel * node_rel * self._delay_reliability(delay, task.deadline_s)
        energy = 1e3 * (1.2 * path_delay + 0.05 * (queue + compute))
        return Candidate(action, "satellite", available, delay, energy, reliability, link_rel, node_rel, queue, required, float("inf"), relay_uav)

    def _candidate_evaluations(self, task: Task) -> list[Candidate]:
        position = self.device_positions[task.device]
        return [self._local_candidate(task), *[self._edge_candidate(task, position, i) for i in range(self.num_edges)], *[self._uav_candidate(task, position, i) for i in range(self.num_uavs)], *[self._sat_candidate(task, position, i) for i in range(self.num_satellites)]]

    def _observation(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.current_task is not None
        task = self.current_task
        candidates = self._candidate_evaluations(task)
        self._last_candidates = candidates
        pos = self.device_positions[task.device] / float(self.env_cfg["area_km"])
        header = np.asarray([np.log1p(task.data_bits) / 18.0, np.log1p(task.cycles) / 24.0, min(task.deadline_s / 120.0, 1.0), task.reliability_required, pos[0], pos[1], self.step_count / max(self.max_steps, 1), self.uav_battery.mean()], dtype=np.float32)
        features = np.asarray([value for candidate in candidates for value in (min(candidate.delay_s / max(task.deadline_s, 1e-6), 3.0) / 3.0, candidate.reliability, candidate.node_availability, min(candidate.queue_s / max(task.deadline_s, 1e-6), 1.0), float(candidate.available))], dtype=np.float32)
        return np.concatenate((header, features)), np.asarray([candidate.available for candidate in candidates], dtype=bool)

    def _replica_plan(self, primary: Candidate) -> list[Candidate]:
        assert self.current_task is not None
        task = self.current_task
        if not bool(self.env_cfg["enable_redundancy"]) or task.reliability_required <= float(self.env_cfg["replica_threshold"]):
            return [primary]
        available = [candidate for candidate in self._last_candidates if candidate.available]
        mean_reliability = float(np.clip(np.mean([candidate.reliability for candidate in available]) if available else primary.reliability, 1e-6, 1.0 - 1e-6))
        required = int(np.clip(ceil(log(1.0 - task.reliability_required) / log(1.0 - mean_reliability)), 1, int(self.env_cfg["max_replicas"])))
        selected = [primary]
        alternatives = [candidate for candidate in available if candidate.action != primary.action]
        alternatives.sort(key=lambda candidate: (candidate.layer == primary.layer, -candidate.reliability, candidate.delay_s))
        for candidate in alternatives:
            if len(selected) >= required:
                break
            selected.append(candidate)
        return selected

    def _reserve(self, replicas: list[Candidate]) -> None:
        for candidate in replicas:
            if candidate.action != 0:
                self.reserved_capacity[candidate.action] += candidate.required_capacity

    def _release(self, replicas: list[Candidate]) -> None:
        for candidate in replicas:
            if candidate.action != 0:
                self.reserved_capacity[candidate.action] = max(0.0, self.reserved_capacity[candidate.action] - candidate.required_capacity)

    def _promote_winner(self, winner: Candidate, finish_time: float) -> None:
        if winner.action == 0:
            return
        self.reserved_capacity[winner.action] = max(0.0, self.reserved_capacity[winner.action] - winner.required_capacity)
        self.used_capacity[winner.action] += winner.required_capacity
        self.active_allocations.append((finish_time, winner.action, winner.required_capacity))

    def _cancel_losers(self, replicas: list[Candidate], winner: Candidate | None) -> None:
        for candidate in replicas:
            if candidate.action == 0 or candidate is winner:
                continue
            self.reserved_capacity[candidate.action] = max(0.0, self.reserved_capacity[candidate.action] - candidate.required_capacity)

    def _release_completed_allocations(self) -> None:
        remaining: list[tuple[float, int, float]] = []
        for finish_time, action, capacity in self.active_allocations:
            if finish_time <= self.current_time_s:
                self.used_capacity[action] = max(0.0, self.used_capacity[action] - capacity)
            else:
                remaining.append((finish_time, action, capacity))
        self.active_allocations = remaining

    def _advance_clock(self) -> None:
        period = float(self.env_cfg.get("arrival_period_s", 86400.0))
        amplitude = float(self.env_cfg.get("arrival_amplitude", 0.3))
        intensity = float(self.env_cfg["arrival_rate"]) * (1.0 + amplitude * np.sin(2 * pi * self.current_time_s / period + self.arrival_phase))
        self.time_step_s = float(self.rng.exponential(1.0 / max(intensity, 1e-6)))
        self.current_time_s += self.time_step_s
        self._release_completed_allocations()
        self.node_queues = np.maximum(0.0, self.node_queues - self.time_step_s)
        self.local_queue_s = np.maximum(0.0, self.local_queue_s - self.time_step_s)
        self.relay_arrival_rate *= np.exp(-self.time_step_s / 5.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")
        assert self.current_task is not None
        task = self.current_task
        selected = self._last_candidates[int(action)]
        invalid = not selected.available
        if invalid:
            selected = self._last_candidates[0]
            action = 0
        replicas = self._replica_plan(selected)
        self._reserve(replicas)
        combined_reliability = 1.0 - float(np.prod([1.0 - item.reliability for item in replicas]))
        successful_replicas = [item for item in replicas if self.rng.random() < item.reliability]
        winner = min(successful_replicas, key=lambda item: item.delay_s) if successful_replicas else None
        latency = winner.delay_s if winner is not None else max(item.delay_s for item in replicas)
        energy = sum(item.energy_mj for item in replicas)
        deadline_met = latency <= task.deadline_s
        reliability_success = winner is not None
        completed = deadline_met and (reliability_success if bool(self.env_cfg.get("sample_reliability_failures", True)) else True)
        cost = smooth_shortfall(task.reliability_required - combined_reliability, float(self.env_cfg["cost_temperature"]))
        latency_normalized = min(latency / max(task.deadline_s, 1e-6), 3.0)
        energy_normalized = min(energy / 2000.0, 3.0)
        smooth_met = sigmoid((combined_reliability - task.reliability_required) / float(self.env_cfg["tau_smooth"]))
        performance = -float(self.reward_cfg["latency"]) * latency_normalized - float(self.reward_cfg["energy"]) * energy_normalized
        reward = performance + float(self.reward_cfg["reliability"]) * combined_reliability * smooth_met - float(self.reward_cfg["violation"]) * cost - (1.0 if invalid else 0.0)
        # The first successful replica becomes used capacity; every loser is canceled immediately.
        self._cancel_losers(replicas, winner)
        if winner is not None:
            self._promote_winner(winner, self.current_time_s + winner.delay_s)
        else:
            self._release(replicas)
        execution_target = winner if winner is not None else selected
        if execution_target.action == 0:
            self.local_queue_s[task.device] += self._service_time(execution_target, task)
        else:
            self.node_queues[execution_target.action] += self._service_time(execution_target, task)
            if execution_target.layer == "uav":
                uav_index = execution_target.action - 1 - self.num_edges
                self.uav_battery[uav_index] = max(0.0, self.uav_battery[uav_index] - 2e-5 * len(replicas))
                self.relay_arrival_rate[uav_index] += 1.0 / max(self.time_step_s, 1e-6)
            elif execution_target.layer == "satellite" and execution_target.relay_uav is not None:
                self.relay_arrival_rate[execution_target.relay_uav] += 1.0 / max(self.time_step_s, 1e-6)
        self._advance_clock()
        record: dict[str, float | str | int] = {"latency_s": latency, "energy_mj": energy, "reliability": combined_reliability, "required_reliability": task.reliability_required, "cost": cost, "completed": int(completed), "deadline_met": int(deadline_met), "reliability_success": int(reliability_success), "violation": int(combined_reliability < task.reliability_required), "replicas": len(replicas), "layer": selected.layer, "device": task.device, "task_kind": task.kind, "edge_utilization": float(np.mean(self.node_queues[1 : 1 + self.num_edges] > 0.0)), "invalid": int(invalid), "reserved_after_event": float(self.reserved_capacity.sum()), "used_after_event": float(self.used_capacity.sum())}
        self._metrics.append(record)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        self.current_task = self._sample_task()
        next_state, next_mask = self._observation()
        return next_state, float(reward), terminated, False, {**record, "action_mask": next_mask, "raw_action": int(action)}

    def summary(self) -> dict[str, float]:
        if not self._metrics:
            return {}
        numeric = lambda key: np.asarray([float(row[key]) for row in self._metrics], dtype=np.float64)
        return {"tasks": float(len(self._metrics)), "tcr": float(100.0 * numeric("completed").mean()), "deadline_satisfaction_pct": float(100.0 * numeric("deadline_met").mean()), "latency_ms": float(1000.0 * numeric("latency_s").mean()), "energy_mj": float(numeric("energy_mj").mean()), "reliability_pct": float(100.0 * numeric("reliability").mean()), "resource_utilization_pct": float(100.0 * numeric("edge_utilization").mean()), "cvr": float(100.0 * numeric("violation").mean()), "expected_cost": float(numeric("cost").mean()), "mean_replicas": float(numeric("replicas").mean()), "mean_reserved_after_event": float(numeric("reserved_after_event").mean())}
