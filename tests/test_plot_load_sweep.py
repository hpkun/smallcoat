import json
from pathlib import Path

from scripts.plot_load_sweep import load_load_sweep
from scripts.plot_load_sweep import plot_load_sweep


def test_load_sweep_summary_and_plot(tmp_path: Path) -> None:
    root = tmp_path / "load_sweep"
    root.mkdir()
    for model, multiplier in (("proposed", 3.0), ("cmaddpg", 2.0), ("cmppo", 1.0)):
        path = root / f"{model}_lambda_25_seed_42.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "record_type": "episode",
                        "episode": episode,
                        "episode_system_profit": multiplier * (episode + 1),
                    }
                    for episode in range(4)
                ]
            ),
            encoding="utf-8",
        )
    summary = load_load_sweep(root, tail_episodes=2, steps_per_episode=2)
    assert summary["proposed"][25.0][0] == 5.25
    output = plot_load_sweep(summary, tmp_path / "fig7.png")
    assert output.exists() and output.stat().st_size > 0
