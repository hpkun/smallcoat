from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.plot_uav_battery import load_battery_series, plot_uav_battery


def _status(remaining: float) -> dict[str, dict[str, float | bool]]:
    return {
        "uav-0": {
            "remaining_energy_j": remaining,
            "battery_level": remaining / 100.0,
            "safe_energy_j": 20.0,
            "episode_energy_consumed_j": 100.0 - remaining,
            "can_serve": remaining > 20.0,
        }
    }


def test_load_and_plot_step_battery_series(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {"record_type": "battery_step", "episode": 0, "step": 1, "battery_status": _status(80.0)},
                {"record_type": "battery_step", "episode": 0, "step": 0, "battery_status": _status(90.0)},
                {"record_type": "episode", "episode": 0, "episode_battery_status": _status(80.0)},
            ]
        ),
        encoding="utf-8",
    )

    labels, remaining, levels = load_battery_series(metrics_path)
    assert labels == ["0:0", "0:1"]
    np.testing.assert_allclose(remaining["uav-0"], [90.0, 80.0])
    np.testing.assert_allclose(levels["uav-0"], [0.9, 0.8])

    output_path = tmp_path / "battery.png"
    assert plot_uav_battery(metrics_path, output_path) == output_path
    assert output_path.stat().st_size > 0


def test_load_episode_battery_series(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {"episode": 1, "episode_battery_status": _status(70.0)},
                {"episode": 0, "episode_battery_status": _status(80.0)},
            ]
        ),
        encoding="utf-8",
    )

    labels, remaining, _ = load_battery_series(metrics_path, level="episode")
    assert labels == ["0", "1"]
    np.testing.assert_allclose(remaining["uav-0"], [80.0, 70.0])
