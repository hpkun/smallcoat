from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_redundancy_scheme_bars import SchemeSummary
from scripts.plot_redundancy_scheme_bars import summarize_scheme
from scripts.plot_task_completion_bars import plot_task_completion


def test_plot_task_completion(tmp_path: Path) -> None:
    summaries = [
        SchemeSummary(
            label="No Redundancy",
            total_tasks=100,
            success_tasks=84,
            total_redundant_tasks=0,
            success_redundant_tasks=0,
        ),
        SchemeSummary(
            label="Hybrid Redundancy",
            total_tasks=100,
            success_tasks=87,
            total_redundant_tasks=10,
            success_redundant_tasks=8,
        ),
    ]
    output_path = tmp_path / "completion.png"

    plot_task_completion(summaries, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_redundancy_scheme_uses_admitted_tasks_as_success_denominator(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {
                    "record_type": "episode",
                    "episode_total_tasks": 100,
                    "episode_completed_tasks": 80,
                    "episode_task_completion_rate": 0.8,
                    "episode_redundancy_requested_tasks": 20,
                    "episode_redundant_tasks": 20,
                    "episode_admitted_redundant_tasks": 15,
                    "episode_redundancy_success_tasks": 12,
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_scheme(metrics_path, "Hybrid", episode_level=True)

    assert summary.total_redundant_tasks == 15
    assert summary.success_redundant_tasks == 12
    assert summary.redundancy_success_rate == 0.8
