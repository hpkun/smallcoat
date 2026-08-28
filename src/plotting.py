from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_curve(
    x_values,
    y_values,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot a single metric curve and save it to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, linewidth=2.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def moving_average(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x positions and moving-average values for a 1D sequence."""

    series = _as_float_array(values)
    if series.size == 0:
        return np.array([], dtype=int), series
    if window <= 1 or series.size < window:
        return np.arange(series.size), series

    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(series, kernel, mode="valid")
    x_values = np.arange(window - 1, series.size)
    return x_values, smoothed


def clip_by_quantile(
    values: Sequence[float],
    *,
    lower_quantile: float | None = None,
    upper_quantile: float | None = None,
) -> tuple[np.ndarray, float | None, float | None]:
    """Clip a series by quantiles and return the clipped values with bounds."""

    series = _as_float_array(values)
    if series.size == 0:
        return series, None, None

    lower = float(np.quantile(series, lower_quantile)) if lower_quantile is not None else None
    upper = float(np.quantile(series, upper_quantile)) if upper_quantile is not None else None
    clipped = np.clip(
        series,
        lower if lower is not None else -np.inf,
        upper if upper is not None else np.inf,
    )
    return clipped, lower, upper


def load_training_metrics(metrics_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load step-level and episode-level records from a metrics JSON file."""

    records = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    step_records = [
        record
        for record in records
        if "step" in record and "actor_loss" in record and "critic_loss" in record
    ]
    episode_records = [record for record in records if "episode_shared_reward" in record]
    return step_records, episode_records


def plot_training_metrics_summary(
    metrics_path: str | Path,
    *,
    output_path: str | Path,
    reward_window: int = 10,
    critic_window: int = 100,
    critic_upper_quantile: float = 0.95,
) -> Path:
    """Plot a training summary with clipped and smoothed critic loss."""

    step_records, episode_records = load_training_metrics(metrics_path)
    if not step_records:
        raise ValueError(f"No step records found in {metrics_path}")

    train_step = np.arange(len(step_records))
    actor_loss = _as_float_array([record["actor_loss"] for record in step_records])
    critic_loss = _as_float_array([record["critic_loss"] for record in step_records])
    shared_reward = _as_float_array([record["shared_reward"] for record in step_records])

    critic_clipped, _, critic_upper = clip_by_quantile(
        critic_loss,
        upper_quantile=critic_upper_quantile,
    )
    critic_x, critic_smoothed = moving_average(critic_clipped, critic_window)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(train_step, actor_loss, color="#1f77b4", linewidth=1.2)
    axes[0, 0].set_title("Actor Loss vs Training Step")
    axes[0, 0].set_xlabel("Training step record")
    axes[0, 0].set_ylabel("Actor loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)

    axes[0, 1].plot(
        train_step,
        critic_clipped,
        color="#f2b134",
        alpha=0.35,
        linewidth=0.8,
        label=f"Clipped at q={critic_upper_quantile:.2f}",
    )
    axes[0, 1].plot(
        critic_x,
        critic_smoothed,
        color="#d62728",
        linewidth=1.6,
        label=f"Moving avg ({critic_window})",
    )
    axes[0, 1].set_title("Critic Loss vs Training Step")
    axes[0, 1].set_xlabel("Training step record")
    axes[0, 1].set_ylabel("Critic loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)
    axes[0, 1].legend()
    if critic_upper is not None:
        axes[0, 1].set_ylim(bottom=0.0, top=critic_upper * 1.05)

    if episode_records:
        episodes = np.asarray([record["episode"] for record in episode_records], dtype=int)
        episode_reward = _as_float_array(
            [record["episode_shared_reward"] for record in episode_records]
        )
        reward_x_offset, reward_smoothed = moving_average(episode_reward, reward_window)
        reward_x = episodes[reward_x_offset] if reward_x_offset.size else episodes

        axes[1, 0].plot(
            episodes,
            episode_reward,
            color="#2ca02c",
            alpha=0.45,
            linewidth=1.1,
            label="Episode reward",
        )
        axes[1, 0].plot(
            reward_x,
            reward_smoothed,
            color="#ff7f0e",
            linewidth=2.0,
            label=f"Moving avg ({reward_window})",
        )
        axes[1, 0].legend()
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No episode-level reward records",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    axes[1, 0].set_title("Episode Shared Reward")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].grid(True, linestyle="--", alpha=0.35)

    axes[1, 1].plot(train_step, shared_reward, color="#9467bd", linewidth=1.0)
    axes[1, 1].set_title("Step Shared Reward vs Training Step")
    axes[1, 1].set_xlabel("Training step record")
    axes[1, 1].set_ylabel("Shared reward")
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"Training Metrics Summary: {Path(metrics_path).name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_service_metrics_summary(
    metrics_path: str | Path,
    *,
    output_path: str | Path,
    smoothing_window: int = 10,
) -> Path:
    """Plot completion, deadline failure, capacity drop, and avg completion delay."""

    step_records, episode_records = load_training_metrics(metrics_path)
    if not step_records:
        raise ValueError(f"No step records found in {metrics_path}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if episode_records and "episode_task_completion_rate" in episode_records[0]:
        x_values = np.asarray([record["episode"] for record in episode_records], dtype=int)
        completion_rate = _as_float_array(
            [record.get("episode_task_completion_rate", 0.0) for record in episode_records]
        )
        timeout_or_drop_rate = _as_float_array(
            [record.get("episode_task_timeout_or_drop_rate", 0.0) for record in episode_records]
        )
        deadline_failure_rate = _as_float_array(
            [
                record.get(
                    "episode_task_deadline_failure_rate",
                    record.get("episode_task_timeout_or_drop_rate", 0.0),
                )
                for record in episode_records
            ]
        )
        capacity_drop_rate = _as_float_array(
            [record.get("episode_task_capacity_drop_rate", 0.0) for record in episode_records]
        )
        avg_completion_delay_s = _as_float_array(
            [record.get("episode_avg_task_completion_delay_s", 0.0) for record in episode_records]
        )
        avg_all_delay_s = _as_float_array(
            [record.get("episode_avg_actual_finish_delay_all_tasks_s", 0.0) for record in episode_records]
        )
        avg_failed_delay_s = _as_float_array(
            [record.get("episode_avg_actual_finish_delay_failed_tasks_s", 0.0) for record in episode_records]
        )
        xlabel = "Episode"
    else:
        x_values = np.arange(len(step_records))
        completion_rate = _as_float_array(
            [record.get("task_completion_rate", 0.0) for record in step_records]
        )
        timeout_or_drop_rate = _as_float_array(
            [record.get("task_timeout_or_drop_rate", 0.0) for record in step_records]
        )
        deadline_failure_rate = _as_float_array(
            [
                record.get(
                    "task_deadline_failure_rate",
                    record.get("task_timeout_or_drop_rate", 0.0),
                )
                for record in step_records
            ]
        )
        capacity_drop_rate = _as_float_array(
            [record.get("task_capacity_drop_rate", 0.0) for record in step_records]
        )
        avg_completion_delay_s = _as_float_array(
            [record.get("avg_task_completion_delay_s", 0.0) for record in step_records]
        )
        avg_all_delay_s = _as_float_array(
            [record.get("avg_actual_finish_delay_all_tasks_s", 0.0) for record in step_records]
        )
        avg_failed_delay_s = _as_float_array(
            [record.get("avg_actual_finish_delay_failed_tasks_s", 0.0) for record in step_records]
        )
        xlabel = "Training step record"

    smooth_idx, completion_rate_smooth = moving_average(completion_rate, smoothing_window)
    _, timeout_or_drop_rate_smooth = moving_average(timeout_or_drop_rate, smoothing_window)
    _, deadline_failure_rate_smooth = moving_average(deadline_failure_rate, smoothing_window)
    _, capacity_drop_rate_smooth = moving_average(capacity_drop_rate, smoothing_window)
    _, avg_completion_delay_smooth = moving_average(avg_completion_delay_s, smoothing_window)
    _, avg_all_delay_smooth = moving_average(avg_all_delay_s, smoothing_window)
    _, avg_failed_delay_smooth = moving_average(avg_failed_delay_s, smoothing_window)
    smooth_x = x_values[smooth_idx] if smooth_idx.size else x_values

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(x_values, completion_rate, color="#2ca02c", alpha=0.35, linewidth=1.0)
    axes[0].plot(smooth_x, completion_rate_smooth, color="#1b7f3b", linewidth=2.0)
    axes[0].set_ylabel("Completion rate")
    axes[0].set_title("Task Completion Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x_values, deadline_failure_rate, color="#d62728", alpha=0.30, linewidth=1.0)
    axes[1].plot(smooth_x, deadline_failure_rate_smooth, color="#8c1d18", linewidth=2.0)
    axes[1].set_ylabel("Deadline failure rate")
    axes[1].set_title("Task Deadline Failure Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(x_values, capacity_drop_rate, color="#ff7f0e", alpha=0.30, linewidth=1.0)
    axes[2].plot(smooth_x, capacity_drop_rate_smooth, color="#b85c00", linewidth=2.0)
    axes[2].set_ylabel("Capacity drop rate")
    axes[2].set_title("Task Capacity Drop Rate")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    axes[3].plot(x_values, avg_completion_delay_s, color="#1f77b4", alpha=0.35, linewidth=1.0)
    axes[3].plot(x_values, avg_all_delay_s, color="#9467bd", alpha=0.18, linewidth=0.9)
    axes[3].plot(x_values, avg_failed_delay_s, color="#7f7f7f", alpha=0.18, linewidth=0.9)
    axes[3].plot(smooth_x, avg_completion_delay_smooth, color="#0d3d8f", linewidth=2.0)
    axes[3].plot(smooth_x, avg_all_delay_smooth, color="#6a3d9a", linewidth=1.6)
    axes[3].plot(smooth_x, avg_failed_delay_smooth, color="#4d4d4d", linewidth=1.6)
    axes[3].set_ylabel("Delay (s)")
    axes[3].set_xlabel(xlabel)
    axes[3].set_title("Average Actual Finish Delay")
    axes[3].grid(True, linestyle="--", alpha=0.35)
    axes[3].legend(
        ["completed raw", "all raw", "failed raw", "completed avg", "all avg", "failed avg"],
        loc="upper right",
        fontsize=8,
    )

    if not (
        ("episode_task_deadline_failure_rate" in episode_records[0])
        if episode_records and "episode_task_completion_rate" in episode_records[0]
        else ("task_deadline_failure_rate" in step_records[0])
    ):
        axes[1].text(
            0.01,
            0.88,
            "Legacy log: using timeout/drop as deadline failure proxy",
            transform=axes[1].transAxes,
            fontsize=9,
            color="#8c1d18",
        )

    fig.suptitle(f"Service Metrics Summary: {Path(metrics_path).name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_layer_arrival_rates_summary(
    metrics_path: str | Path,
    *,
    output_path: str | Path,
    smoothing_window: int = 10,
) -> Path:
    """Plot task arrival/offloading rates across UAV, BS, and LEO layers."""

    step_records, episode_records = load_training_metrics(metrics_path)
    if not step_records:
        raise ValueError(f"No step records found in {metrics_path}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if episode_records and "episode_uav_arrival_rate" in episode_records[0]:
        x_values = np.asarray([record["episode"] for record in episode_records], dtype=int)
        uav_rate = _as_float_array([record.get("episode_uav_arrival_rate", 0.0) for record in episode_records])
        bs_rate = _as_float_array([record.get("episode_bs_arrival_rate", 0.0) for record in episode_records])
        leo_rate = _as_float_array([record.get("episode_leo_arrival_rate", 0.0) for record in episode_records])
        xlabel = "Episode"
    else:
        x_values = np.arange(len(step_records))
        uav_rate = _as_float_array([record.get("uav_arrival_rate", 0.0) for record in step_records])
        bs_rate = _as_float_array([record.get("bs_arrival_rate", 0.0) for record in step_records])
        leo_rate = _as_float_array([record.get("leo_arrival_rate", 0.0) for record in step_records])
        xlabel = "Training step record"

    smooth_idx, uav_smooth = moving_average(uav_rate, smoothing_window)
    _, bs_smooth = moving_average(bs_rate, smoothing_window)
    _, leo_smooth = moving_average(leo_rate, smoothing_window)
    smooth_x = x_values[smooth_idx] if smooth_idx.size else x_values

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x_values, uav_rate, color="#7aa6c2", alpha=0.25, linewidth=1.0)
    ax.plot(x_values, bs_rate, color="#f2b134", alpha=0.25, linewidth=1.0)
    ax.plot(x_values, leo_rate, color="#c7674f", alpha=0.25, linewidth=1.0)
    ax.plot(smooth_x, uav_smooth, color="#1f77b4", linewidth=2.0, label="UAV layer")
    ax.plot(smooth_x, bs_smooth, color="#d99000", linewidth=2.0, label="BS layer")
    ax.plot(smooth_x, leo_smooth, color="#a83220", linewidth=2.0, label="LEO layer")
    ax.set_title(f"Layer Task Arrival Rates: {Path(metrics_path).name}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Arrival/offloading rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_layer_diagnostics_summary(
    metrics_path: str | Path,
    *,
    output_path: str | Path,
    smoothing_window: int = 10,
) -> Path:
    """Plot per-layer deadline failure rates and actual finish delays."""

    step_records, episode_records = load_training_metrics(metrics_path)
    if not step_records:
        raise ValueError(f"No step records found in {metrics_path}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if episode_records and "episode_uav_deadline_failure_rate" in episode_records[0]:
        x_values = np.asarray([record["episode"] for record in episode_records], dtype=int)
        prefix = "episode_"
        records = episode_records
        xlabel = "Episode"
    else:
        x_values = np.arange(len(step_records))
        prefix = ""
        records = step_records
        xlabel = "Training step record"

    uav_failure = _as_float_array([record.get(f"{prefix}uav_deadline_failure_rate", 0.0) for record in records])
    bs_failure = _as_float_array([record.get(f"{prefix}bs_deadline_failure_rate", 0.0) for record in records])
    leo_failure = _as_float_array([record.get(f"{prefix}leo_deadline_failure_rate", 0.0) for record in records])
    uav_delay = _as_float_array([record.get(f"{prefix}uav_avg_delay_s", 0.0) for record in records])
    bs_delay = _as_float_array([record.get(f"{prefix}bs_avg_delay_s", 0.0) for record in records])
    leo_delay = _as_float_array([record.get(f"{prefix}leo_avg_delay_s", 0.0) for record in records])

    smooth_idx, uav_failure_smooth = moving_average(uav_failure, smoothing_window)
    _, bs_failure_smooth = moving_average(bs_failure, smoothing_window)
    _, leo_failure_smooth = moving_average(leo_failure, smoothing_window)
    _, uav_delay_smooth = moving_average(uav_delay, smoothing_window)
    _, bs_delay_smooth = moving_average(bs_delay, smoothing_window)
    _, leo_delay_smooth = moving_average(leo_delay, smoothing_window)
    smooth_x = x_values[smooth_idx] if smooth_idx.size else x_values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(smooth_x, uav_failure_smooth, color="#1f77b4", linewidth=2.0, label="UAV")
    axes[0].plot(smooth_x, bs_failure_smooth, color="#d99000", linewidth=2.0, label="BS")
    axes[0].plot(smooth_x, leo_failure_smooth, color="#a83220", linewidth=2.0, label="LEO")
    axes[0].set_title("Per-Layer Deadline Failure Rate")
    axes[0].set_ylabel("Failure rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(smooth_x, uav_delay_smooth, color="#1f77b4", linewidth=2.0, label="UAV")
    axes[1].plot(smooth_x, bs_delay_smooth, color="#d99000", linewidth=2.0, label="BS")
    axes[1].plot(smooth_x, leo_delay_smooth, color="#a83220", linewidth=2.0, label="LEO")
    axes[1].set_title("Per-Layer Average Actual Finish Delay")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Delay (s)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle(f"Layer Diagnostics: {Path(metrics_path).name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
