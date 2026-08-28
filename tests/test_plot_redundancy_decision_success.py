from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_redundancy_decision_success import plot_summaries
from scripts.plot_redundancy_decision_success import summarize_metrics


def test_summarize_and_plot_redundancy_decision_success(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {"record_type": "battery_step", "episode": 0},
                {
                    "record_type": "episode",
                    "episode": 0,
                    "episode_redundancy_requested_tasks": 12,
                    "episode_redundant_tasks": 10,
                    "episode_admitted_redundant_tasks": 10,
                    "episode_redundancy_success_tasks": 8,
                },
                {
                    "record_type": "episode",
                    "episode": 1,
                    "episode_redundancy_requested_tasks": 8,
                    "episode_redundant_tasks": 5,
                    "episode_admitted_redundant_tasks": 5,
                    "episode_redundancy_success_tasks": 4,
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_metrics(metrics_path, "Hybrid Redundancy")
    output_path = tmp_path / "figure.png"
    plot_summaries([summary], output_path)

    assert summary.admitted_tasks == 15
    assert summary.decision_tasks == 15
    assert summary.success_tasks == 12
    assert summary.success_rate == 0.8
    assert output_path.exists()
    assert output_path.stat().st_size > 0
