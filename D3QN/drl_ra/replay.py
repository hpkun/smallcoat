from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_masks: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self._data: deque[tuple] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self._data.append((state.copy(), action, reward, next_state.copy(), done, next_mask.copy()))

    def sample(self, batch_size: int) -> Batch:
        indices = self._rng.choice(len(self._data), size=batch_size, replace=False)
        rows = [self._data[int(index)] for index in indices]
        return Batch(
            states=np.stack([row[0] for row in rows]).astype(np.float32),
            actions=np.asarray([row[1] for row in rows], dtype=np.int64),
            rewards=np.asarray([row[2] for row in rows], dtype=np.float32),
            next_states=np.stack([row[3] for row in rows]).astype(np.float32),
            dones=np.asarray([row[4] for row in rows], dtype=np.float32),
            next_masks=np.stack([row[5] for row in rows]).astype(bool),
        )
