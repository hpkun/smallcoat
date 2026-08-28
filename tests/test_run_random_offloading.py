from __future__ import annotations

import numpy as np

from scripts.run_random_offloading import sample_random_non_redundant_action
from src.action_space import build_action_spec


def test_random_offloading_uses_real_legal_nodes_without_redundancy() -> None:
    action_spec = build_action_spec(
        ["target-slot-0", "target-slot-1", "target-slot-2"],
        [[True, True, False], [False, True, True]],
        slot_target_node_ids=[
            ["uav-0", "bs-0", ""],
            ["", "bs-1", "leo-0"],
        ],
    )

    action = sample_random_non_redundant_action(
        action_spec,
        np.random.default_rng(42),
    )

    assert action.slot_actions[0].target_node_id in {"uav-0", "bs-0"}
    assert action.slot_actions[1].target_node_id in {"bs-1", "leo-0"}
    assert all(slot.priority_eta == 0.5 for slot in action.slot_actions)
    assert all(slot.redundancy_eta == 0.0 for slot in action.slot_actions)
    assert all(slot.backup_target_node_id is None for slot in action.slot_actions)
