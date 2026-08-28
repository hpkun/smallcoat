from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILE_PATTERN = re.compile(
    r"(?P<model>[A-Za-z0-9_-]+)_lambda_(?P<rate>[0-9]+(?:\.[0-9]+)?)_seed_(?P<seed>[0-9]+)\.json$"
)


def _episode_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    episodes = [
        record
        for record in records
        if record.get("record_type") == "episode"
        and "episode_system_profit" in record
    ]
    if not episodes:
        raise ValueError(f"No episode system-profit records found in {path}")
    return sorted(episodes, key=lambda record: int(record.get("episode", 0)))


def load_load_sweep(
    root: Path,
    *,
    tail_episodes: int = 100,
    steps_per_episode: int = 50,
) -> dict[str, dict[float, tuple[float, float, int]]]:
    """Return model -> arrival rate -> (mean, std, number of seeds).

    Files must be named ``{model}_lambda_{rate}_seed_{seed}.json``. Values are
    mean system profit per simulation time slot over the selected tail episodes.
    """

    if tail_episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("tail_episodes and steps_per_episode must be positive")
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError(f"No JSON metrics files found under {root}")
    for path in paths:
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        model = match.group("model")
        rate = float(match.group("rate"))
        episodes = _episode_records(path)[-tail_episodes:]
        total_profit = float(sum(float(row["episode_system_profit"]) for row in episodes))
        total_slots = len(episodes) * steps_per_episode
        values[(model, rate)].append(total_profit / float(total_slots))
    if not values:
        raise ValueError(
            f"No files matching {{model}}_lambda_{{rate}}_seed_{{seed}}.json under {root}"
        )
    summary: dict[str, dict[float, tuple[float, float, int]]] = defaultdict(dict)
    for (model, rate), observations in sorted(values.items()):
        array = np.asarray(observations, dtype=float)
        summary[model][rate] = (
            float(array.mean()),
            float(array.std(ddof=1)) if array.size > 1 else 0.0,
            int(array.size),
        )
    return dict(summary)


def plot_load_sweep(
    summary: dict[str, dict[float, tuple[float, float, int]]],
    output_path: Path,
    *,
    scale: float = 1e9,
) -> Path:
    if scale <= 0:
        raise ValueError("scale must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    colors = {"proposed": "#7a4fa3", "cmaddpg": "#d97706", "cmppo": "#1479b8"}
    markers = {"proposed": "o", "cmaddpg": "s", "cmppo": "d"}
    for model in sorted(summary):
        rates = np.asarray(sorted(summary[model]), dtype=float)
        means = np.asarray([summary[model][rate][0] for rate in rates]) / scale
        stds = np.asarray([summary[model][rate][1] for rate in rates]) / scale
        label = {"proposed": "Proposed", "cmaddpg": "CMADDPG", "cmppo": "CMPPO"}.get(
            model, model
        )
        color = colors.get(model, None)
        ax.plot(
            rates,
            means,
            marker=markers.get(model, "o"),
            linewidth=2.0,
            markersize=5.5,
            label=label,
            color=color,
        )
        if np.any(stds > 0):
            ax.fill_between(rates, means - stds, means + stds, color=color, alpha=0.12)
    ax.set_xlabel("Task Arrival Rate (tasks/s)")
    ax.set_ylabel("Average System Profit per Time Slot (billion units)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fig.7 load-sweep system profit.")
    parser.add_argument("--root", type=Path, default=Path("outputs/load_sweep"))
    parser.add_argument("--tail-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--scale", type=float, default=1e9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/fig7_load_sweep_profit.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_load_sweep(
        args.root,
        tail_episodes=args.tail_episodes,
        steps_per_episode=args.steps_per_episode,
    )
    output = plot_load_sweep(summary, args.output, scale=args.scale)
    for model, rates in summary.items():
        for rate, (mean, std, count) in rates.items():
            print(
                f"{model} lambda={rate:g}: mean={mean:.6f} std={std:.6f} seeds={count}"
            )
    print(f"figure={output}")


if __name__ == "__main__":
    main()
