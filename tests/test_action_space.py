from __future__ import annotations

import numpy as np

from src.action_space import MixedActionCodec, build_action_spec


def test_action_codec_decodes_three_unique_replicas() -> None:
    spec = build_action_spec(
        ["uav-0", "bs-0", "leo-0"],
        [[True, True, True]],
    )
    codec = MixedActionCodec(spec)
    actor_output = np.array(
        [
            0.0, 1.0, 5.0,
            5.0, 0.0, 0.0,
            10.0, 9.0, 0.0,
            10.0, 9.0, 8.0,
            -2.0,
        ],
        dtype=np.float32,
    )

    decoded = codec.decode_numpy(actor_output)

    assert spec.per_slot_output_dim == 13
    assert decoded.slot_replica_counts == [3]
    assert decoded.slot_replica_target_node_ids == [("uav-0", "bs-0", "leo-0")]
    assert decoded.slot_priority_etas[0] < 0.5
    critic_action = codec.encode_for_critic(
        decoded.slot_replica_counts,
        decoded.slot_replica_target_indices,
        decoded.slot_priority_etas,
    )
    assert critic_action.shape == (13,)
    assert np.array_equal(critic_action[:3], np.array([0.0, 0.0, 1.0]))


def test_action_codec_has_no_artificial_cross_layer_backup_mask() -> None:
    spec = build_action_spec(
        ["uav-0", "uav-1", "leo-0"],
        [[True, True, True]],
    )
    codec = MixedActionCodec(spec)
    actor_output = np.array(
        [
            0.0, 5.0, 0.0,
            5.0, 0.0, 0.0,
            10.0, 9.0, 0.0,
            0.0, 0.0, 5.0,
            2.0,
        ],
        dtype=np.float32,
    )

    decoded = codec.decode_numpy(actor_output)

    assert decoded.slot_replica_counts == [2]
    assert decoded.slot_replica_target_node_ids == [("uav-0", "uav-1")]
