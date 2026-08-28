from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


MAX_REPLICA_COUNT = 3


@dataclass(frozen=True, init=False)
class SlotAction:
    """A task action with an explicit replica count and ordered placements."""

    replica_count: int
    replica_target_node_ids: tuple[str, ...]
    priority_eta: float

    def __init__(
        self,
        target_node_id: str | None = None,
        priority_eta: float = 0.5,
        redundancy_eta: float = 0.0,
        backup_target_node_id: str | None = None,
        *,
        replica_count: int | None = None,
        replica_target_node_ids: Sequence[str] | None = None,
    ) -> None:
        """Build a proposed action while accepting the former two-copy shape."""

        if replica_target_node_ids is not None:
            target_ids = tuple(str(node_id) for node_id in replica_target_node_ids if node_id)
            requested_count = len(target_ids) if replica_count is None else int(replica_count)
        else:
            target_ids = tuple(
                node_id
                for node_id in (target_node_id, backup_target_node_id)
                if node_id
            )
            if replica_count is not None:
                requested_count = int(replica_count)
            elif backup_target_node_id is not None and redundancy_eta >= 0.5:
                requested_count = 2
            else:
                requested_count = 1

        if not 1 <= requested_count <= MAX_REPLICA_COUNT:
            raise ValueError(f"replica_count must be in [1, {MAX_REPLICA_COUNT}].")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("replica_target_node_ids must be distinct.")
        if len(target_ids) > requested_count:
            target_ids = target_ids[:requested_count]
        if target_ids and not 0.0 <= priority_eta <= 1.0:
            raise ValueError("priority_eta must be in [0, 1].")

        object.__setattr__(self, "replica_count", requested_count)
        object.__setattr__(self, "replica_target_node_ids", target_ids)
        object.__setattr__(self, "priority_eta", float(priority_eta))

    @property
    def target_node_id(self) -> str:
        return self.replica_target_node_ids[0] if self.replica_target_node_ids else ""

    @property
    def backup_target_node_id(self) -> str | None:
        return self.replica_target_node_ids[1] if len(self.replica_target_node_ids) > 1 else None

    @property
    def redundancy_eta(self) -> float:
        """Deprecated compatibility view; Proposed uses ``replica_count``."""

        return float((self.replica_count - 1) / (MAX_REPLICA_COUNT - 1))


@dataclass(frozen=True)
class MultiTaskOffloadingAction:
    """A CH action over all task slots."""

    slot_actions: list[SlotAction]


@dataclass(frozen=True)
class ActionSpec:
    """Variable-task action specification with slot-level candidate masks."""

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
        return MAX_REPLICA_COUNT + MAX_REPLICA_COUNT * self.num_discrete_targets + 1

    @property
    def actor_output_dim(self) -> int:
        return self.num_task_slots * self.per_slot_output_dim


@dataclass(frozen=True)
class DecodedAction:
    """Decoded task actions with requested counts and valid unique placements."""

    slot_replica_counts: list[int]
    slot_replica_target_indices: list[tuple[int, ...]]
    slot_replica_target_node_ids: list[tuple[str, ...]]
    slot_priority_etas: list[float]
    raw_logits: np.ndarray
    legacy_redundancy_etas: list[float] | None = None

    @property
    def slot_target_indices(self) -> list[int]:
        return [indices[0] for indices in self.slot_replica_target_indices]

    @property
    def slot_target_node_ids(self) -> list[str]:
        return [node_ids[0] for node_ids in self.slot_replica_target_node_ids]

    @property
    def slot_backup_target_indices(self) -> list[int | None]:
        return [indices[1] if len(indices) > 1 else None for indices in self.slot_replica_target_indices]

    @property
    def slot_backup_target_node_ids(self) -> list[str | None]:
        return [node_ids[1] if len(node_ids) > 1 else None for node_ids in self.slot_replica_target_node_ids]

    @property
    def slot_redundancy_etas(self) -> list[float]:
        if self.legacy_redundancy_etas is not None:
            return self.legacy_redundancy_etas
        return [float((count - 1) / (MAX_REPLICA_COUNT - 1)) for count in self.slot_replica_counts]

    def to_multi_task_action(self) -> MultiTaskOffloadingAction:
        return MultiTaskOffloadingAction(
            slot_actions=[
                SlotAction(
                    replica_count=replica_count,
                    replica_target_node_ids=target_node_ids,
                    priority_eta=priority_eta,
                )
                for replica_count, target_node_ids, priority_eta in zip(
                    self.slot_replica_counts,
                    self.slot_replica_target_node_ids,
                    self.slot_priority_etas,
                )
            ]
        )


class MixedActionCodec:
    """Encode and decode replica-count, placement, and priority decisions."""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    @staticmethod
    def _masked_argmax(logits: np.ndarray, mask: Sequence[bool]) -> int:
        mask_array = np.asarray(mask, dtype=bool)
        if not np.any(mask_array):
            raise ValueError("A task slot has no legal replica target.")
        masked_logits = np.asarray(logits, dtype=np.float32).copy()
        masked_logits[~mask_array] = -1e9
        return int(np.argmax(masked_logits))

    def _slot_node_ids(self, slot_index: int) -> list[str]:
        if self.spec.slot_target_node_ids is not None:
            return self.spec.slot_target_node_ids[slot_index]
        return self.spec.target_node_ids

    @staticmethod
    def _legacy_backup_mask(
        node_ids: list[str],
        primary_index: int,
        candidate_mask: Sequence[bool],
    ) -> list[bool]:
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

    def _decode_legacy_numpy(self, actor_output: np.ndarray) -> DecodedAction:
        """Decode archived 2K+2 baseline actions without affecting Proposed."""

        target_count = self.spec.num_discrete_targets
        legacy_width = 2 * target_count + 2
        reshaped = np.asarray(actor_output, dtype=np.float32).reshape(
            self.spec.num_task_slots,
            legacy_width,
        )
        replica_counts: list[int] = []
        target_indices_by_slot: list[tuple[int, ...]] = []
        target_ids_by_slot: list[tuple[str, ...]] = []
        priority_etas: list[float] = []
        redundancy_etas: list[float] = []
        for slot_index, vector in enumerate(reshaped):
            primary_index = self._masked_argmax(
                vector[:target_count],
                self.spec.slot_target_masks[slot_index],
            )
            node_ids = self._slot_node_ids(slot_index)
            backup_mask = self._legacy_backup_mask(
                node_ids,
                primary_index,
                self.spec.slot_target_masks[slot_index],
            )
            backup_index = (
                self._masked_argmax(vector[target_count : 2 * target_count], backup_mask)
                if any(backup_mask)
                else None
            )
            redundancy_eta = float(1.0 / (1.0 + np.exp(-float(vector[-1]))))
            redundancy_etas.append(redundancy_eta)
            use_backup = backup_index is not None and redundancy_eta >= 0.5
            indices = (
                (primary_index, int(backup_index))
                if use_backup
                else (primary_index,)
            )
            replica_counts.append(len(indices))
            target_indices_by_slot.append(indices)
            target_ids_by_slot.append(tuple(node_ids[index] for index in indices))
            priority_etas.append(float(1.0 / (1.0 + np.exp(-float(vector[-2])))))
        return DecodedAction(
            slot_replica_counts=replica_counts,
            slot_replica_target_indices=target_indices_by_slot,
            slot_replica_target_node_ids=target_ids_by_slot,
            slot_priority_etas=priority_etas,
            raw_logits=reshaped.copy(),
            legacy_redundancy_etas=redundancy_etas,
        )

    def decode_numpy(self, actor_output: np.ndarray) -> DecodedAction:
        legacy_dim = self.spec.num_task_slots * (2 * self.spec.num_discrete_targets + 2)
        if actor_output.shape[-1] == legacy_dim:
            return self._decode_legacy_numpy(actor_output)
        if actor_output.shape[-1] != self.spec.actor_output_dim:
            raise ValueError("actor_output has invalid dimension.")

        reshaped = np.asarray(actor_output, dtype=np.float32).reshape(
            self.spec.num_task_slots,
            self.spec.per_slot_output_dim,
        )
        replica_counts: list[int] = []
        replica_target_indices: list[tuple[int, ...]] = []
        replica_target_node_ids: list[tuple[str, ...]] = []
        priority_etas: list[float] = []

        target_count = self.spec.num_discrete_targets
        placement_start = MAX_REPLICA_COUNT
        for slot_index, slot_vector in enumerate(reshaped):
            requested_count = int(np.argmax(slot_vector[:MAX_REPLICA_COUNT])) + 1
            candidate_mask = list(self.spec.slot_target_masks[slot_index])
            selected_indices: list[int] = []
            for head_index in range(MAX_REPLICA_COUNT):
                head_start = placement_start + head_index * target_count
                head_logits = slot_vector[head_start : head_start + target_count]
                head_mask = [
                    allowed and index not in selected_indices
                    for index, allowed in enumerate(candidate_mask)
                ]
                if not any(head_mask):
                    break
                selected_indices.append(self._masked_argmax(head_logits, head_mask))

            effective_indices = tuple(selected_indices[:requested_count])
            node_ids = self._slot_node_ids(slot_index)
            replica_counts.append(requested_count)
            replica_target_indices.append(effective_indices)
            replica_target_node_ids.append(tuple(node_ids[index] for index in effective_indices))
            priority_etas.append(float(1.0 / (1.0 + np.exp(-float(slot_vector[-1])))))

        return DecodedAction(
            slot_replica_counts=replica_counts,
            slot_replica_target_indices=replica_target_indices,
            slot_replica_target_node_ids=replica_target_node_ids,
            slot_priority_etas=priority_etas,
            raw_logits=reshaped.copy(),
        )

    def decode_torch(
        self,
        actor_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reshaped = actor_output.view(-1, self.spec.num_task_slots, self.spec.per_slot_output_dim)
        target_count = self.spec.num_discrete_targets
        count_prob = torch.softmax(reshaped[..., :MAX_REPLICA_COUNT], dim=-1)
        placement_probs = []
        for head_index in range(MAX_REPLICA_COUNT):
            start = MAX_REPLICA_COUNT + head_index * target_count
            placement_probs.append(torch.softmax(reshaped[..., start : start + target_count], dim=-1))
        priority_eta = torch.sigmoid(reshaped[..., -1:])
        return count_prob, *placement_probs, priority_eta

    def encode_for_critic(
        self,
        replica_counts: list[int],
        replica_target_indices: list[tuple[int, ...]] | list[int | None],
        priority_etas: list[float],
        legacy_redundancy_etas: list[float] | None = None,
    ) -> np.ndarray:
        """Encode hard actions using the actor's per-task width."""

        if legacy_redundancy_etas is not None:
            primary_indices = replica_counts
            backup_indices = replica_target_indices
            converted_counts: list[int] = []
            converted_targets: list[tuple[int, ...]] = []
            for primary, backup, redundancy in zip(
                primary_indices,
                backup_indices,
                legacy_redundancy_etas,
            ):
                use_backup = backup is not None and redundancy >= 0.5
                converted_counts.append(2 if use_backup else 1)
                converted_targets.append(
                    (int(primary), int(backup)) if use_backup else (int(primary),)
                )
            replica_counts = converted_counts
            replica_target_indices = converted_targets

        parts: list[np.ndarray] = []
        for replica_count, target_indices, priority_eta in zip(
            replica_counts,
            replica_target_indices,
            priority_etas,
        ):
            count_one_hot = np.zeros(MAX_REPLICA_COUNT, dtype=np.float32)
            count_one_hot[int(replica_count) - 1] = 1.0
            placement_parts: list[np.ndarray] = []
            normalized_indices = tuple(int(index) for index in target_indices)  # type: ignore[arg-type]
            for head_index in range(MAX_REPLICA_COUNT):
                one_hot = np.zeros(self.spec.num_discrete_targets, dtype=np.float32)
                if head_index < len(normalized_indices):
                    one_hot[normalized_indices[head_index]] = 1.0
                placement_parts.append(one_hot)
            parts.append(
                np.concatenate(
                    [count_one_hot, *placement_parts, np.array([priority_eta], dtype=np.float32)]
                )
            )
        return np.concatenate(parts, axis=0)

    def sample_random_action(self, rng: np.random.Generator) -> MultiTaskOffloadingAction:
        slot_actions: list[SlotAction] = []
        for slot_index, mask in enumerate(self.spec.slot_target_masks):
            valid_indices = [index for index, allowed in enumerate(mask) if allowed]
            if not valid_indices:
                raise ValueError("A task slot has no legal replica target.")
            replica_count = int(rng.integers(1, min(MAX_REPLICA_COUNT, len(valid_indices)) + 1))
            selected = rng.choice(valid_indices, size=replica_count, replace=False)
            node_ids = self._slot_node_ids(slot_index)
            slot_actions.append(
                SlotAction(
                    replica_count=replica_count,
                    replica_target_node_ids=tuple(node_ids[int(index)] for index in selected),
                    priority_eta=float(rng.uniform(0.0, 1.0)),
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
