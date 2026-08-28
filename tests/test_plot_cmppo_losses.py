from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_cmppo_losses import plot_cmppo_losses


def test_plot_cmppo_losses(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {
                    "record_type": "episode",
                    "episode": episode,
                    "episode_actor_loss": 0.2 / float(episode + 1),
                    "episode_critic_loss": 1.0 / float(episode + 1),
                }
                for episode in range(10)
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "losses.png"

    plot_cmppo_losses(metrics_path, output_path, window=3)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
