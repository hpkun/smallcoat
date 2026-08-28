from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class SlotAction:
    """一个任务槽位的卸载动作。"""

    target_node_id: str
    priority_eta: float
    redundancy_eta: float = 0.0
    backup_target_node_id: str | None = None


@dataclass(frozen=True)
class MultiTaskOffloadingAction:
    """一个 CH 对多个任务槽位的联合动作。"""

    slot_actions: list[SlotAction]


@dataclass(frozen=True)
class ActionSpec:
    """
    多任务混合动作空间规格。

    - target_node_ids: 当前 CH 在本时隙下所有槽位候选目标的并集
    - slot_target_masks: 每个槽位对并集目标的合法性掩码
    """

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
        return 2 * self.num_discrete_targets + 2

    @property
    def actor_output_dim(self) -> int:
        return self.num_task_slots * self.per_slot_output_dim


@dataclass(frozen=True)
class DecodedAction:
    """解码后的多槽位动作结果。"""

    slot_target_indices: list[int]
    slot_target_node_ids: list[str]
    slot_backup_target_indices: list[int | None]
    slot_backup_target_node_ids: list[str | None]
    slot_priority_etas: list[float]
    slot_redundancy_etas: list[float]
    raw_logits: np.ndarray

    def to_multi_task_action(self) -> MultiTaskOffloadingAction:
        return MultiTaskOffloadingAction(
            slot_actions=[
                SlotAction(
                    target_node_id=target_node_id,
                    priority_eta=priority_eta,
                    redundancy_eta=redundancy_eta,
                    backup_target_node_id=backup_target_node_id,
                )
                for target_node_id, backup_target_node_id, priority_eta, redundancy_eta in zip(
                    self.slot_target_node_ids,
                    self.slot_backup_target_node_ids,
                    self.slot_priority_etas,
                    self.slot_redundancy_etas,
                )
            ]
        )


class MixedActionCodec:
    """多槽位混合动作编解码器，支持槽位级合法目标掩码。"""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    def _masked_argmax(self, logits: np.ndarray, mask: list[bool]) -> int:
        masked_logits = logits.copy()
        mask_array = np.asarray(mask, dtype=bool)
        if not np.any(mask_array):
            return 0
        masked_logits[~mask_array] = -1e9
        return int(np.argmax(masked_logits))

    def _slot_node_ids(self, slot_index: int) -> list[str]:
        if self.spec.slot_target_node_ids is not None:
            return self.spec.slot_target_node_ids[slot_index]
        return self.spec.target_node_ids

    @staticmethod
    def _backup_mask(node_ids: list[str], primary_index: int, candidate_mask: list[bool]) -> list[bool]:
        """根据主节点类型生成异节点备份 mask。"""

        primary_id = node_ids[primary_index]
        if primary_id.startswith("uav-"):
            valid_prefixes = ("bs-", "leo-")
        elif primary_id.startswith("bs-"):
            valid_prefixes = ("leo-",)
        elif primary_id.startswith("leo-"):
            valid_prefixes = ("bs-",)
        else:
            valid_prefixes = ()
        return [
            allowed
            and index != primary_index
            and bool(node_id)
            and node_id.startswith(valid_prefixes)
            for index, (node_id, allowed) in enumerate(zip(node_ids, candidate_mask))
        ]

    def decode_numpy(self, actor_output: np.ndarray) -> DecodedAction:
        if actor_output.shape[-1] != self.spec.actor_output_dim:
            raise ValueError("actor_output has invalid dimension.")

        reshaped = np.asarray(actor_output, dtype=np.float32).reshape(
            self.spec.num_task_slots,
            self.spec.per_slot_output_dim,
        )
        slot_target_indices: list[int] = []
        slot_target_node_ids: list[str] = []
        slot_backup_target_indices: list[int | None] = []
        slot_backup_target_node_ids: list[str | None] = []
        slot_priority_etas: list[float] = []
        slot_redundancy_etas: list[float] = []
        raw_logits: list[np.ndarray] = []

        for slot_index, slot_vector in enumerate(reshaped):
            num_targets = self.spec.num_discrete_targets
            logits = np.asarray(slot_vector[:num_targets], dtype=np.float32)
            backup_logits = np.asarray(slot_vector[num_targets : 2 * num_targets], dtype=np.float32)
            priority_logit = float(slot_vector[-2])
            redundancy_logit = float(slot_vector[-1])
            target_index = self._masked_argmax(logits, self.spec.slot_target_masks[slot_index])
            node_ids = self._slot_node_ids(slot_index)
            backup_mask = self._backup_mask(
                node_ids,
                target_index,
                self.spec.slot_target_masks[slot_index],
            )
            backup_index = (
                self._masked_argmax(backup_logits, backup_mask)
                if any(backup_mask)
                else None
            )
            priority_eta = float(1.0 / (1.0 + np.exp(-priority_logit)))
            redundancy_eta = float(1.0 / (1.0 + np.exp(-redundancy_logit)))
            slot_target_indices.append(target_index)
            slot_target_node_ids.append(node_ids[target_index])
            slot_backup_target_indices.append(backup_index)
            slot_backup_target_node_ids.append(
                node_ids[backup_index] if backup_index is not None else None
            )
            slot_priority_etas.append(priority_eta)
            slot_redundancy_etas.append(redundancy_eta)
            raw_logits.append(np.concatenate([logits, backup_logits]))

        return DecodedAction(
            slot_target_indices=slot_target_indices,
            slot_target_node_ids=slot_target_node_ids,
            slot_backup_target_indices=slot_backup_target_indices,
            slot_backup_target_node_ids=slot_backup_target_node_ids,
            slot_priority_etas=slot_priority_etas,
            slot_redundancy_etas=slot_redundancy_etas,
            raw_logits=np.stack(raw_logits, axis=0),
        )

    def decode_torch(
        self,
        actor_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reshaped = actor_output.view(-1, self.spec.num_task_slots, self.spec.per_slot_output_dim)
        num_targets = self.spec.num_discrete_targets
        logits = reshaped[..., :num_targets]
        backup_logits = reshaped[..., num_targets : 2 * num_targets]
        priority_eta = torch.sigmoid(reshaped[..., -2:-1])
        redundancy_eta = torch.sigmoid(reshaped[..., -1:])
        target_prob = torch.softmax(logits, dim=-1)
        backup_prob = torch.softmax(backup_logits, dim=-1)
        return target_prob, backup_prob, priority_eta, redundancy_eta

    def encode_for_critic(
        self,
        target_indices: list[int],
        backup_target_indices: list[int | None],
        priority_etas: list[float],
        redundancy_etas: list[float],
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for target_index, backup_target_index, priority_eta, redundancy_eta in zip(
            target_indices,
            backup_target_indices,
            priority_etas,
            redundancy_etas,
        ):
            one_hot = np.zeros(self.spec.num_discrete_targets, dtype=np.float32)
            one_hot[target_index] = 1.0
            backup_one_hot = np.zeros(self.spec.num_discrete_targets, dtype=np.float32)
            if backup_target_index is not None:
                backup_one_hot[backup_target_index] = 1.0
            parts.append(
                np.concatenate(
                    [
                        one_hot,
                        backup_one_hot,
                        np.array([priority_eta, redundancy_eta], dtype=np.float32),
                    ]
                )
            )
        return np.concatenate(parts, axis=0)

    def sample_random_action(self, rng: np.random.Generator) -> MultiTaskOffloadingAction:
        slot_actions: list[SlotAction] = []
        for slot_index, mask in enumerate(self.spec.slot_target_masks):
            valid_indices = [i for i, allowed in enumerate(mask) if allowed]
            if not valid_indices:
                valid_indices = [0]
            target_index = int(valid_indices[int(rng.integers(0, len(valid_indices)))])
            node_ids = self._slot_node_ids(slot_index)
            backup_mask = self._backup_mask(node_ids, target_index, mask)
            valid_backup_indices = [i for i, allowed in enumerate(backup_mask) if allowed]
            backup_index = (
                int(valid_backup_indices[int(rng.integers(0, len(valid_backup_indices)))])
                if valid_backup_indices
                else None
            )
            priority_eta = float(rng.uniform(0.0, 1.0))
            redundancy_eta = float(rng.uniform(0.0, 1.0))
            slot_actions.append(
                SlotAction(
                    target_node_id=self.spec.target_node_ids[target_index],
                    priority_eta=priority_eta,
                    redundancy_eta=redundancy_eta,
                    backup_target_node_id=(
                        node_ids[backup_index] if backup_index is not None else None
                    ),
                )
            )
        return MultiTaskOffloadingAction(slot_actions=slot_actions)


def build_action_spec(
    target_node_ids: Sequence[str],
    slot_target_masks: Sequence[Sequence[bool]],
    *,
    slot_target_node_ids: Sequence[Sequence[str]] | None = None,
) -> ActionSpec:
    return ActionSpec(
        target_node_ids=list(target_node_ids),
        slot_target_masks=[list(mask) for mask in slot_target_masks],
        slot_target_node_ids=(
            [list(node_ids) for node_ids in slot_target_node_ids]
            if slot_target_node_ids is not None
            else None
        ),
    )
