from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.plot_scenario_baselines import plot_reliable_completion_comparison
from scripts.plot_scenario_baselines import plot_utility_comparison
from scripts.plot_scenario_baselines import summarize_scenario


def _write_metrics(path: Path) -> None:
    records = [
        {
            "record_type": "episode",
            "episode": 0,
            "episode_u_base": 1.0,
            "episode_u_net": 0.8,
            "episode_total_tasks_for_utility": 10,
            "episode_reliable_on_time_tasks": 6,
        },
        {
            "record_type": "episode",
            "episode": 1,
            "episode_u_base": 2.0,
            "episode_u_net": 1.5,
            "episode_total_tasks_for_utility": 20,
            "episode_reliable_on_time_tasks": 16,
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_scenario_summary_and_plots(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    _write_metrics(metrics_path)

    summary = summarize_scenario(
        metrics_path,
        label="Balanced",
        tail_episodes=2,
    )

    assert math.isclose(summary.u_base, 1.5)
    assert math.isclose(summary.u_net, 1.15)
    assert math.isclose(summary.reliable_on_time_completion_rate, 22.0 / 30.0)
    utility_path = plot_utility_comparison([summary], tmp_path / "utility.png")
    reliability_path = plot_reliable_completion_comparison(
        [summary], tmp_path / "reliability.png"
    )
    assert utility_path.exists() and utility_path.stat().st_size > 0
    assert reliability_path.exists() and reliability_path.stat().st_size > 0
