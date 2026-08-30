from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .action_space import MultiTaskOffloadingAction
from .action_space import SlotAction


@dataclass(frozen=True)
class BaselineActionSpec:
    """Single-copy action: one target distribution and one priority value."""

    target_node_ids: list[str]
    slot_target_masks: list[list[bool]]
    slot_target_node_ids: list[list[str]] | None = None

    @property
    def num_task_slots(self) -> int:
        return len(self.slot_target_masks)

    @property
    def num_discrete_targets(self) -> int:
        return len(self.target_node_ids)

    @property
    def per_slot_output_dim(self) -> int:
        return self.num_discrete_targets + 1

    @property
    def actor_output_dim(self) -> int:
        return self.num_task_slots * self.per_slot_output_dim


@dataclass(frozen=True)
class BaselineDecodedAction:
    slot_target_indices: list[int]
    slot_target_node_ids: list[str]
    slot_priority_etas: list[float]
    raw_logits: np.ndarray

    def to_multi_task_action(self) -> MultiTaskOffloadingAction:
        return MultiTaskOffloadingAction(
            slot_actions=[
                SlotAction(
                    replica_count=1,
                    replica_target_node_ids=(target_node_id,),
                    priority_eta=priority_eta,
                )
                for target_node_id, priority_eta in zip(
                    self.slot_target_node_ids,
                    self.slot_priority_etas,
                )
            ]
        )


class BaselineActionCodec:
    def __init__(self, spec: BaselineActionSpec) -> None:
        self.spec = spec

    @staticmethod
    def _masked_argmax(logits: np.ndarray, mask: Sequence[bool]) -> int:
        mask_array = np.asarray(mask, dtype=bool)
        if not np.any(mask_array):
            raise ValueError("A task slot has no legal offloading target.")
        masked_logits = np.asarray(logits, dtype=np.float32).copy()
        masked_logits[~mask_array] = -1e9
        return int(np.argmax(masked_logits))

    def _slot_node_ids(self, slot_index: int) -> list[str]:
        if self.spec.slot_target_node_ids is not None:
            return self.spec.slot_target_node_ids[slot_index]
        return self.spec.target_node_ids

    def decode_numpy(self, actor_output: np.ndarray) -> BaselineDecodedAction:
        width = self.spec.per_slot_output_dim
        values = np.asarray(actor_output, dtype=np.float32).reshape(
            self.spec.num_task_slots,
            width,
        )
        target_count = self.spec.num_discrete_targets
        target_indices: list[int] = []
        target_node_ids: list[str] = []
        priority_etas: list[float] = []
        for slot_index, vector in enumerate(values):
            target_index = self._masked_argmax(
                vector[:target_count],
                self.spec.slot_target_masks[slot_index],
            )
            target_indices.append(target_index)
            target_node_ids.append(self._slot_node_ids(slot_index)[target_index])
            priority_etas.append(float(1.0 / (1.0 + np.exp(-vector[-1]))))
        return BaselineDecodedAction(
            slot_target_indices=target_indices,
            slot_target_node_ids=target_node_ids,
            slot_priority_etas=priority_etas,
            raw_logits=values,
        )

    def encode_for_critic(
        self,
        target_indices: Sequence[int],
        priority_etas: Sequence[float],
    ) -> np.ndarray:
        if len(target_indices) != self.spec.num_task_slots:
            raise ValueError("Target index count differs from the number of task slots.")
        if len(priority_etas) != self.spec.num_task_slots:
            raise ValueError("Priority count differs from the number of task slots.")
        encoded = np.zeros(
            (self.spec.num_task_slots, self.spec.per_slot_output_dim),
            dtype=np.float32,
        )
        for slot_index, (target_index, priority_eta) in enumerate(
            zip(target_indices, priority_etas)
        ):
            if not self.spec.slot_target_masks[slot_index][target_index]:
                raise ValueError("Selected target is masked for this task slot.")
            encoded[slot_index, target_index] = 1.0
            encoded[slot_index, -1] = float(priority_eta)
        return encoded.reshape(-1)


def build_baseline_action_spec(
    target_node_ids: Sequence[str],
    slot_target_masks: Sequence[Sequence[bool]],
    *,
    slot_target_node_ids: Sequence[Sequence[str]] | None = None,
) -> BaselineActionSpec:
    targets = [str(node_id) for node_id in target_node_ids]
    masks = [[bool(value) for value in mask] for mask in slot_target_masks]
    if not targets:
        raise ValueError("At least one target slot is required.")
    if any(len(mask) != len(targets) for mask in masks):
        raise ValueError("Every target mask must match target_node_ids.")
    physical_ids = (
        [[str(node_id) for node_id in row] for row in slot_target_node_ids]
        if slot_target_node_ids is not None
        else None
    )
    if physical_ids is not None:
        if len(physical_ids) != len(masks):
            raise ValueError("Physical target rows must match task slots.")
        if any(len(row) != len(targets) for row in physical_ids):
            raise ValueError("Every physical target row must match target_node_ids.")
    return BaselineActionSpec(targets, masks, physical_ids)
