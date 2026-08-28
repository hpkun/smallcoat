from __future__ import annotations

import numpy as np

from .action_space import ActionSpec
from .action_space import MixedActionCodec
from .action_space import MultiTaskOffloadingAction
from .action_space import SlotAction


class RandomOffloadingBaseline:
    """随机多任务动作基线。"""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, action_spec: ActionSpec) -> MultiTaskOffloadingAction:
        codec = MixedActionCodec(action_spec)
        return codec.sample_random_action(self.rng)


class HeuristicLatencyBaseline:
    """
    启发式时延优先基线。

    对每个任务槽位：
    - 优先选候选列表中的第一个目标
    - eta 与该槽位任务的时延紧迫度正相关
    """

    def act(self, action_spec: ActionSpec, observation: np.ndarray) -> MultiTaskOffloadingAction:
        num_slots = action_spec.num_task_slots
        per_slot_state_dim = int(observation.shape[0] // num_slots)
        slot_actions: list[SlotAction] = []

        for slot_index in range(num_slots):
            slot_state = observation[
                slot_index * per_slot_state_dim : (slot_index + 1) * per_slot_state_dim
            ]
            task_latency_ms = float(slot_state[-1])
            eta = float(max(0.0, min(1.0, 1.0 - task_latency_ms / 200.0)))
            valid_indices = [
                index for index, allowed in enumerate(action_spec.slot_target_masks[slot_index]) if allowed
            ]
            target_index = valid_indices[0] if valid_indices else 0
            target_node_id = action_spec.target_node_ids[target_index]
            slot_actions.append(
                SlotAction(
                    target_node_id=target_node_id,
                    priority_eta=eta,
                    redundancy_eta=0.0,
                )
            )

        return MultiTaskOffloadingAction(slot_actions=slot_actions)
