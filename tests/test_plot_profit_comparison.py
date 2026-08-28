from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_profit_comparison import load_episode_profit
from scripts.plot_profit_comparison import plot_paper_cumulative_profit
from scripts.plot_profit_comparison import plot_profit_comparison


def test_plot_profit_comparison_and_per_task_loading(tmp_path: Path) -> None:
    metrics_paths = []
    for scheme_index in range(2):
        metrics_path = tmp_path / f"scheme_{scheme_index}.json"
        metrics_path.write_text(
            json.dumps(
                [
                    {
                        "record_type": "episode",
                        "episode": episode,
                        "episode_system_profit": float(
                            (scheme_index + 1) * (episode + 1) * 1e9
                        ),
                        "episode_total_tasks": episode + 1,
                    }
                    for episode in range(5)
                ]
            ),
            encoding="utf-8",
        )
        metrics_paths.append(metrics_path)

    _, per_task_profit = load_episode_profit(metrics_paths[0], per_task=True)
    output_path = tmp_path / "profit.png"
    paper_output_path = tmp_path / "paper_profit.png"
    plot_profit_comparison(
        metrics_paths,
        ["Scheme A", "Scheme B"],
        output_path,
        window=2,
    )
    plot_paper_cumulative_profit(
        metrics_paths,
        ["Scheme A", "Scheme B"],
        paper_output_path,
        steps_per_episode=50,
        max_time=200,
    )

    assert all(value == 1e9 for value in per_task_profit)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert paper_output_path.exists()
    assert paper_output_path.stat().st_size > 0
