from __future__ import annotations

import json

from scripts.plot_fig6_profit import load_cumulative_profit
from scripts.plot_fig6_profit import plot_fig6


def test_plot_fig6_uses_evaluation_time_slots(tmp_path) -> None:
    metrics_path = tmp_path / "steps.json"
    metrics_path.write_text(
        json.dumps(
            [
                {
                    "record_type": "evaluation_step",
                    "time_slot": time_slot,
                    "cumulative_system_profit": float(time_slot * 10),
                }
                for time_slot in range(1, 6)
            ]
        ),
        encoding="utf-8",
    )
    time_slots, profits = load_cumulative_profit(metrics_path)
    assert time_slots.tolist() == [1, 2, 3, 4, 5]
    assert profits.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    output_path = tmp_path / "fig6.png"
    assert plot_fig6(metrics_path, output_path, max_time=5) == output_path
    assert output_path.exists()
