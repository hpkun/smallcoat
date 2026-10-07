from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class ReplicaCandidate(Protocol):
    action: int
    available: bool
    reliability: float
    energy_mj: float


@dataclass(frozen=True)
class ReplicaTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


def combined_reliability(candidates: Sequence[ReplicaCandidate]) -> float:
    if not candidates:
        return 0.0
    failures = np.asarray([1.0 - item.reliability for item in candidates], dtype=np.float64)
    return 1.0 - float(np.prod(failures))


def redundancy_state(
    base_state: np.ndarray,
    candidates: Sequence[ReplicaCandidate],
    selected_actions: Sequence[int],
    required_reliability: float,
    max_replicas: int,
    energy_scale_mj: float,
) -> np.ndarray:
    """Build [s_108, selected_20, count, reliability, gap, energy]."""
    if not selected_actions:
        raise ValueError("a replica set must contain a primary action")
    selected_mask = np.zeros(len(candidates), dtype=np.float32)
    selected = []
    for action in selected_actions:
        if not 0 <= int(action) < len(candidates):
            raise ValueError(f"replica action {action} is out of range")
        selected_mask[int(action)] = 1.0
        selected.append(candidates[int(action)])
    reliability = combined_reliability(selected)
    energy = sum(item.energy_mj for item in selected)
    aggregate = np.asarray(
        [
            len(selected) / max(int(max_replicas), 1),
            reliability,
            float(required_reliability) - reliability,
            min(energy / max(float(energy_scale_mj), 1e-9), 3.0) / 3.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (np.asarray(base_state, dtype=np.float32), selected_mask, aggregate)
    )


def redundancy_action_mask(
    candidates: Sequence[ReplicaCandidate],
    selected_actions: Sequence[int],
    max_replicas: int,
) -> np.ndarray:
    """Return one action per candidate plus an always-valid STOP action."""
    stop_action = len(candidates)
    mask = np.zeros(stop_action + 1, dtype=bool)
    if len(selected_actions) < int(max_replicas):
        mask[:stop_action] = np.asarray(
            [item.available for item in candidates], dtype=bool
        )
        mask[np.asarray(selected_actions, dtype=np.int64)] = False
    mask[stop_action] = True
    return mask
