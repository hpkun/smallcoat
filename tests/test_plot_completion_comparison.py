from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_completion_comparison import plot_completion_comparison


def test_plot_completion_comparison(tmp_path: Path) -> None:
    metrics_paths = []
    for index in range(3):
        metrics_path = tmp_path / f"scheme_{index}.json"
        metrics_path.write_text(
            json.dumps(
                [
                    {
                        "record_type": "episode",
                        "episode": episode,
                        "episode_task_completion_rate": 0.7 + index * 0.05,
                    }
                    for episode in range(5)
                ]
            ),
            encoding="utf-8",
        )
        metrics_paths.append(metrics_path)

    output_path = tmp_path / "comparison.png"
    plot_completion_comparison(
        metrics_paths,
        ["Proposed", "No Redundancy", "Random"],
        output_path,
        window=3,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
