from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_episode_failure_rates import plot_episode_failure_rates


def test_plot_episode_failure_rates(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {
                    "record_type": "episode",
                    "episode": episode,
                    "episode_task_timeout_or_drop_rate": 0.2,
                    "episode_task_deadline_failure_rate": 0.05,
                    "episode_task_capacity_drop_rate": 0.1,
                    "episode_reliability_failure_rate": 0.04,
                }
                for episode in range(10)
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "failure_rates.png"

    plot_episode_failure_rates(metrics_path, output_path, window=3)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
