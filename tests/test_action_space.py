from __future__ import annotations

import numpy as np

from src.action_space import MixedActionCodec
from src.action_space import build_action_spec


def test_action_codec_decodes_priority_and_redundancy_separately() -> None:
    spec = build_action_spec(
        ["uav-0", "bs-0", "leo-0"],
        [[True, True, True]],
    )
    codec = MixedActionCodec(spec)

    actor_output = np.array(
        [0.0, 2.0, 1.0, 0.0, 0.0, 3.0, -2.0, 2.0],
        dtype=np.float32,
    )
    decoded = codec.decode_numpy(actor_output)

    assert spec.per_slot_output_dim == 8
    assert decoded.slot_target_node_ids == ["bs-0"]
    assert decoded.slot_backup_target_node_ids == ["leo-0"]
    assert decoded.slot_priority_etas[0] < 0.5
    assert decoded.slot_redundancy_etas[0] > 0.5

    critic_action = codec.encode_for_critic(
        decoded.slot_target_indices,
        decoded.slot_backup_target_indices,
        decoded.slot_priority_etas,
        decoded.slot_redundancy_etas,
    )

    assert critic_action.shape == (8,)


def test_action_codec_uses_base_station_backup_for_leo_primary() -> None:
    spec = build_action_spec(
        ["uav-0", "bs-0", "leo-0"],
        [[True, True, True]],
    )
    codec = MixedActionCodec(spec)

    actor_output = np.array(
        [0.0, 1.0, 3.0, 10.0, 2.0, 20.0, 0.0, 2.0],
        dtype=np.float32,
    )
    decoded = codec.decode_numpy(actor_output)

    assert decoded.slot_target_node_ids == ["leo-0"]
    assert decoded.slot_backup_target_node_ids == ["bs-0"]
