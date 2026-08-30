from __future__ import annotations

from .baseline_action_space import BaselineActionCodec
from .baseline_action_space import BaselineActionSpec
from .baseline_action_space import build_baseline_action_spec
from .rl_env import CMADDPGEnv
from .rl_env import MAX_TARGET_SLOTS


class BaselineCMADDPGEnv(CMADDPGEnv):
    """CMADDPG environment exposing only single-copy baseline actions."""

    def _build_action_spec(
        self,
        slot_target_nodes: list[list],
    ) -> tuple[BaselineActionSpec, list[list[str]]]:
        slot_target_node_ids = [
            [node.node_id if node is not None else "" for node in nodes]
            for nodes in slot_target_nodes
        ]
        slot_target_masks = [
            [node is not None for node in nodes]
            for nodes in slot_target_nodes
        ]
        target_slot_ids = [f"target-slot-{index}" for index in range(MAX_TARGET_SLOTS)]
        return (
            build_baseline_action_spec(
                target_slot_ids,
                slot_target_masks,
                slot_target_node_ids=slot_target_node_ids,
            ),
            slot_target_node_ids,
        )

    @staticmethod
    def encode_action_for_env(
        action_spec: BaselineActionSpec,
        actor_output,
    ):
        return BaselineActionCodec(action_spec).decode_numpy(
            actor_output
        ).to_multi_task_action()
