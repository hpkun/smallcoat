from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

from .environment import Candidate, SAGINEnv, TASK_NAMES, Task


SCHEMA_VERSION = 1
INITIAL_ARRAYS = (
    "device_positions", "edge_positions", "uav_centers", "uav_phases",
    "uav_speeds_mps", "uav_radius_km", "uav_altitudes_km", "uav_battery",
    "edge_availability", "sat_phases", "device_types", "total_capacity",
    "local_capacity",
)
CHANNEL_FIELDS = {"direct_rate": 0, "direct_delay": 1, "rain": 2, "scintillation": 3}
TOPOLOGY_FIELDS = ("num_devices", "num_edges", "num_uavs", "num_satellites")


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trace_digest(trace: dict[str, Any]) -> str:
    return digest({key: value for key, value in trace.items() if key != "sha256"})


def input_sequence_digest(entries: list[dict[str, Any]]) -> str:
    return digest([digest(entry) for entry in entries])


def generate_trace(config: dict[str, Any], seed: int = 42, steps: int = 2000) -> dict[str, Any]:
    """Generate inputs independently of policy actions and replica counts."""
    if steps < 1:
        raise ValueError("steps must be positive")
    config = deepcopy(config)
    config["environment"]["episode_steps"] = steps
    topology, task_stream, arrival_stream, channel_stream, outcome_stream = np.random.SeedSequence(seed).spawn(5)
    topology_seed = int(topology.generate_state(1)[0])
    sampler = SAGINEnv(config, seed=topology_seed)
    initial = {name: getattr(sampler, name).tolist() for name in INITIAL_ARRAYS}
    initial.update(arrival_phase=sampler.arrival_phase, time_step_s=sampler.time_step_s)
    sampler.rng = np.random.default_rng(task_stream)
    arrival_rng = np.random.default_rng(arrival_stream)
    channel_rng = np.random.default_rng(channel_stream)
    outcome_rng = np.random.default_rng(outcome_stream)
    env_cfg = config["environment"]
    current_time = 0.0
    entries = []
    for task_id in range(steps):
        period = float(env_cfg.get("arrival_period_s", 86400.0))
        amplitude = float(env_cfg.get("arrival_amplitude", 0.3))
        intensity = float(env_cfg["arrival_rate"]) * (1.0 + amplitude * np.sin(2 * pi * current_time / period + sampler.arrival_phase))
        interval = float(arrival_rng.exponential(1.0 / max(intensity, 1e-6)))
        entries.append({
            "task_id": task_id,
            "arrival_time_s": current_time,
            "interarrival_s": interval,
            "task": asdict(sampler._sample_task()),
            # Columns 0..3: direct rate, direct delay, rain, scintillation.
            # Then two columns per UAV: feeder rate and relay delay.
            "satellite_uniforms": channel_rng.random((sampler.num_satellites, 4 + 2 * sampler.num_uavs)).tolist(),
            "success_uniforms": outcome_rng.random(sampler.action_dim).tolist(),
        })
        current_time += interval
    trace = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "steps": steps,
        "evaluation_config": {"environment": env_cfg, "reward": config["reward"]},
        "initial_state": initial,
        "entries": entries,
    }
    trace["sha256"] = trace_digest(trace)
    validate_trace(trace)
    return trace


def _numeric_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid trace array {name}: expected finite values with shape {shape}")
    return array


def validate_trace(trace: dict[str, Any]) -> None:
    if trace.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported trace schema")
    if trace.get("sha256") != trace_digest(trace):
        raise ValueError("trace SHA-256 mismatch")
    env_cfg = trace["evaluation_config"]["environment"]
    devices, edges, uavs, satellites = (int(env_cfg[key]) for key in TOPOLOGY_FIELDS)
    if min(devices, edges, uavs, satellites) < 1:
        raise ValueError("trace requires positive topology dimensions")
    action_dim = 1 + edges + uavs + satellites
    steps = trace["steps"]
    if not isinstance(steps, int) or steps < 1 or len(trace["entries"]) != steps or env_cfg["episode_steps"] != steps:
        raise ValueError("trace length does not match episode_steps")
    if int(env_cfg["max_replicas"]) != 3:
        raise ValueError("Step 2-5 audits require max_replicas=3")
    shapes = {
        "device_positions": (devices, 2), "edge_positions": (edges, 2),
        "uav_centers": (uavs, 2), "uav_phases": (uavs,),
        "uav_speeds_mps": (uavs,), "uav_radius_km": (uavs,),
        "uav_altitudes_km": (uavs,), "uav_battery": (uavs,),
        "edge_availability": (edges,), "sat_phases": (satellites,),
        "total_capacity": (action_dim,), "local_capacity": (devices,),
    }
    initial = trace["initial_state"]
    for name, shape in shapes.items():
        values = _numeric_array(initial[name], shape, name)
        if name in ("uav_radius_km", "uav_altitudes_km", "local_capacity") and (values <= 0).any():
            raise ValueError(f"{name} must be positive")
        if name in ("uav_battery", "edge_availability", "sat_phases") and ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} must be in [0, 1]")
    if initial["total_capacity"][0] != 0 or (np.asarray(initial["total_capacity"])[1:] <= 0).any():
        raise ValueError("invalid compute capacities")
    types = initial["device_types"]
    if len(types) != devices or any(kind not in TASK_NAMES for kind in types):
        raise ValueError("invalid device types")
    if not np.isfinite(initial["arrival_phase"]) or not np.isfinite(initial["time_step_s"]) or initial["time_step_s"] <= 0:
        raise ValueError("invalid initial arrival parameters")
    current_time = 0.0
    for task_id, entry in enumerate(trace["entries"]):
        if entry["task_id"] != task_id or entry["arrival_time_s"] != current_time:
            raise ValueError(f"trace task ID or arrival time mismatch at {task_id}")
        interval = entry["interarrival_s"]
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError(f"invalid arrival interval at {task_id}")
        task = Task(**entry["task"])
        if not isinstance(task.device, int) or not 0 <= task.device < devices or task.kind != types[task.device]:
            raise ValueError(f"invalid task device/type at {task_id}")
        values = (task.data_bits, task.cycles, task.deadline_s, task.reliability_required)
        if not np.isfinite(values).all() or min(values[:3]) <= 0 or not 0 <= task.reliability_required <= 1:
            raise ValueError(f"invalid task requirements at {task_id}")
        for name, shape in (("satellite_uniforms", (satellites, 4 + 2 * uavs)), ("success_uniforms", (action_dim,))):
            values = _numeric_array(entry[name], shape, f"{name}[{task_id}]")
            if ((values < 0) | (values >= 1)).any():
                raise ValueError(f"invalid uniform draw at {task_id}")
        current_time += interval


def save_trace(trace: dict[str, Any], path: str | Path, overwrite: bool = False) -> None:
    validate_trace(trace)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(trace, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_trace(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        trace = json.load(stream)
    validate_trace(trace)
    return trace


def evaluation_config(training_config: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Keep network/training settings, use the trace's common evaluation model."""
    common = trace["evaluation_config"]
    for key in TOPOLOGY_FIELDS:
        if training_config["environment"][key] != common["environment"][key]:
            raise ValueError(f"checkpoint topology differs from trace: {key}")
    result = deepcopy(training_config)
    result.update(deepcopy(common))
    # The full learned checkpoint contains both networks; no external warm start.
    result.setdefault("learned_replica", {})["primary_checkpoint"] = None
    return result


class TraceEnv(SAGINEnv):
    """Replay exogenous inputs while resources evolve according to each policy."""

    def __init__(self, config: dict[str, Any], trace: dict[str, Any]) -> None:
        validate_trace(trace)
        self.trace = deepcopy(trace)
        common = trace["evaluation_config"]
        for section in ("environment", "reward"):
            actual, expected = deepcopy(config[section]), deepcopy(common[section])
            if section == "environment":
                actual.pop("enable_redundancy", None)
                expected.pop("enable_redundancy", None)
            if actual != expected:
                raise ValueError(f"use evaluation_config() to share the trace's {section} settings")
        self._restoring = True
        super().__init__(config, seed=int(trace["seed"]))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None and int(seed) != self.trace["seed"]:
            raise ValueError("a fixed trace cannot be reseeded; generate another trace")
        self._restoring = True
        super().reset(seed=int(self.trace["seed"]))
        initial = self.trace["initial_state"]
        for name in INITIAL_ARRAYS:
            setattr(self, name, np.asarray(initial[name], dtype=str if name == "device_types" else np.float64).copy())
        self.arrival_phase = float(initial["arrival_phase"])
        self.time_step_s = float(initial["time_step_s"])
        self._restoring = False
        self.current_task = self._sample_task()
        state, mask = self._observation()
        return state, {"action_mask": mask}

    def _sample_task(self) -> Task:
        if self._restoring:
            return super()._sample_task()
        # Base execution asks for a next task even at termination. No extra
        # trace entry is consumed; _observation returns a terminal sentinel.
        index = min(self.step_count, self.max_steps - 1)
        return Task(**self.trace["entries"][index]["task"])

    def _observation(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._restoring and self.step_count >= self.max_steps:
            self.current_task = None
            self._last_candidates = []
            return np.zeros(self.state_dim, dtype=np.float32), np.zeros(self.action_dim, dtype=bool)
        return super()._observation()

    def _channel_uniform(self, satellite: int, field: str, low: float, high: float, uav: int | None = None) -> float:
        if self._restoring:
            return super()._channel_uniform(satellite, field, low, high, uav)
        if field in CHANNEL_FIELDS:
            column = CHANNEL_FIELDS[field]
        elif field in ("feeder_rate", "relay_delay") and uav is not None:
            column = 4 + 2 * uav + int(field == "relay_delay")
        else:
            raise ValueError(f"unknown channel field: {field}")
        uniform = self.trace["entries"][self.step_count]["satellite_uniforms"][satellite][column]
        return float(low + (high - low) * uniform)

    def _arrival_interval(self) -> float:
        return float(self.trace["entries"][self.step_count]["interarrival_s"])

    def _replica_succeeds(self, candidate: Candidate) -> bool:
        uniform = self.trace["entries"][self.step_count]["success_uniforms"][candidate.action]
        return bool(uniform < candidate.reliability)

    def _require_task(self) -> None:
        if self.step_count >= self.max_steps:
            raise RuntimeError("trace exhausted; reset before stepping again")

    def step(self, action: int):
        self._require_task()
        return super().step(action)

    def step_with_replicas(self, replica_actions: list[int], stop_reason: str = "not_applicable"):
        self._require_task()
        return super().step_with_replicas(replica_actions, stop_reason)
