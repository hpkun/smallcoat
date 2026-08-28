from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class MultiAgentTransition:
    """
    一条联合多智能体经验。

    对应论文公式 (22)-(25) 里的联合样本：
    - joint observation o
    - joint action a
    - shared reward r
    - next joint observation o'
    """

    agent_ids: tuple[str, ...]
    reward: float
    done: float
    local_states: dict[str, np.ndarray]
    next_local_states: dict[str, np.ndarray]
    local_actions: dict[str, np.ndarray]


class MultiAgentReplayBuffer:
    """联合经验回放池。"""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: deque[MultiAgentTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: MultiAgentTransition) -> None:
        """压入一条联合经验。"""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[MultiAgentTransition]:
        """随机采样一批联合经验。"""
        return random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
