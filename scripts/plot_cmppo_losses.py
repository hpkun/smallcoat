from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cmppo_losses(metrics_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss_records = [
        record
        for record in records
        if record.get("episode_actor_loss") is not None
        and record.get("episode_critic_loss") is not None
    ]
    if not loss_records:
        raise ValueError(f"No CMPPO episode loss records found in {metrics_path}")
    loss_records.sort(key=lambda record: int(record["episode"]))
    episodes = np.asarray(
        [int(record["episode"]) + 1 for record in loss_records], dtype=int
    )
    actor_loss = np.asarray(
        [float(record["episode_actor_loss"]) for record in loss_records], dtype=float
    )
    critic_loss = np.asarray(
        [float(record["episode_critic_loss"]) for record in loss_records], dtype=float
    )
    return episodes, actor_loss, critic_loss


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be positive")
    if window == 1 or values.size < window:
        return np.arange(values.size), values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.arange(window - 1, values.size), np.convolve(values, kernel, mode="valid")


def plot_cmppo_losses(
    metrics_path: Path,
    output_path: Path,
    *,
    window: int = 20,
    critic_upper_quantile: float = 0.95,
) -> Path:
    if not 0.0 < critic_upper_quantile <= 1.0:
        raise ValueError("critic_upper_quantile must be in (0, 1]")
    episodes, actor_loss, critic_loss = load_cmppo_losses(metrics_path)
    critic_upper = float(np.quantile(critic_loss, critic_upper_quantile))
    critic_clipped = np.clip(critic_loss, None, critic_upper)
    actor_indices, actor_smooth = moving_average(actor_loss, window)
    critic_indices, critic_smooth = moving_average(critic_clipped, window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        sharex=True,
        constrained_layout=True,
    )
    series = (
        (
            axes[0],
            actor_loss,
            actor_indices,
            actor_smooth,
            "Actor Loss",
            "#1479b8",
        ),
        (
            axes[1],
            critic_clipped,
            critic_indices,
            critic_smooth,
            f"Critic Loss (clipped at q={critic_upper_quantile:.2f})",
            "#b5483a",
        ),
    )
    for axis, raw, smooth_indices, smooth, title, color in series:
        axis.plot(
            episodes,
            raw,
            color=color,
            linewidth=0.7,
            alpha=0.2,
            label="Raw",
        )
        axis.plot(
            episodes[smooth_indices],
            smooth,
            color=color,
            linewidth=2.0,
            label=f"Moving Average ({window})",
        )
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_xlim(left=1)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot episode-level CMPPO Actor and Critic losses."
    )
    parser.add_argument("metrics_path")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--critic-upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--output",
        default="outputs/figures/cmppo_losses.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_path = plot_cmppo_losses(
        Path(args.metrics_path),
        Path(args.output),
        window=args.window,
        critic_upper_quantile=args.critic_upper_quantile,
    )
    print(f"figure={saved_path}")


if __name__ == "__main__":
    main()
