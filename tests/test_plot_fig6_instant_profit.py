from __future__ import annotations

import json

import numpy as np

from scripts.plot_fig6_instant_profit import load_instant_profit
from scripts.plot_fig6_instant_profit import moving_average
from scripts.plot_fig6_instant_profit import plot_instant_profit


def test_plot_instant_profit_uses_step_profit_and_moving_average(tmp_path) -> None:
    metrics_path = tmp_path / "steps.json"
    metrics_path.write_text(
        json.dumps(
            [
                {
                    "record_type": "evaluation_step",
                    "time_slot": time_slot,
                    "system_profit": float(time_slot * 10),
                }
                for time_slot in range(1, 6)
            ]
        ),
        encoding="utf-8",
    )

    time_slots, profits = load_instant_profit(metrics_path)
    assert time_slots.tolist() == [1, 2, 3, 4, 5]
    assert profits.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    smooth_time, smooth_profit = moving_average(time_slots, profits, window=3)
    assert smooth_time.tolist() == [3, 4, 5]
    assert np.allclose(smooth_profit, [20.0, 30.0, 40.0])

    output_path = tmp_path / "instant_profit.png"
    assert (
        plot_instant_profit(
            metrics_path,
            output_path,
            max_time=5,
            window=3,
        )
        == output_path
    )
    assert output_path.exists()
