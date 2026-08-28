from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import TASK_SCENARIO_NAMES
from src.cmppo import CMPPOConfig
from src.cmppo import CMPPOSystem
from src.metrics_logger import MetricsLogger
from train import build_medium_env
from train import build_small_scale_env
from train import build_training_env


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_cmppo(
    *,
    env,
    episodes: int,
    steps: int,
    device: str,
    redundancy_mode: str,
    config: CMPPOConfig,
    progress_interval: int,
) -> tuple[MetricsLogger, CMPPOSystem]:
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")

    logger = MetricsLogger()
    system: CMPPOSystem | None = None
    started_at = time.perf_counter()

    for episode in range(episodes):
        observations, action_specs = env.reset()

        trajectory = []
        rewards: list[float] = []
        all_delays: list[float] = []
        completed_delays: list[float] = []
        failed_delays: list[float] = []
        total_tasks = 0
        completed_tasks = 0
        deadline_failure_tasks = 0
        capacity_drop_tasks = 0
        reliability_failure_tasks = 0
        redundancy_requested_tasks = 0
        redundant_tasks = 0
        redundancy_success_tasks = 0
        transmission_energy_j = 0.0
        computing_energy_j = 0.0
        total_energy_j = 0.0
        system_profit = 0.0
        uav_arrival_tasks = 0
        bs_arrival_tasks = 0
        leo_arrival_tasks = 0

        for _ in range(steps):
            if system is None and observations:
                first_agent_id = sorted(observations)[0]
                system = CMPPOSystem(
                    state_dim=int(observations[first_agent_id].shape[0]),
                    action_spec=action_specs[first_agent_id],
                    device=device,
                    redundancy_mode=redundancy_mode,
                    config=config,
                )
            if system is None:
                actions = {}
                trajectory_step = None
            else:
                actions, trajectory_step = system.sample_actions(
                    observations, action_specs
                )
            next_observations, _, _, info = env.step(actions)
            if trajectory_step is not None:
                trajectory_step.shared_reward = float(info["shared_reward"])
                trajectory.append(trajectory_step)
            observations = next_observations
            action_specs = info["action_specs"]
            records = info["records"]

            rewards.append(float(info["shared_reward"]))
            system_profit += float(info["equation8_objective"].total_profit)
            total_tasks += len(records)
            completed_tasks += sum(
                1 for record in records if record.completed_before_deadline
            )
            deadline_failure_tasks += sum(
                1
                for record in records
                if record.constraint_check is not None
                and not record.constraint_check.satisfies_deadline
            )
            capacity_drop_tasks += sum(
                1
                for record in records
                if record.constraint_check is not None
                and not record.constraint_check.satisfies_capacity
            )
            reliability_failure_tasks += sum(
                1
                for record in records
                if record.failed_due_to_reliability
                or not record.satisfies_reliability
            )
            redundancy_requested_tasks += sum(
                1 for record in records if record.redundancy_requested
            )
            redundant_tasks += sum(1 for record in records if record.is_redundant_task)
            redundancy_success_tasks += sum(
                1 for record in records if record.redundancy_succeeded
            )
            transmission_energy_j += float(
                sum(record.transmission_energy_j for record in records)
            )
            computing_energy_j += float(
                sum(record.computing_energy_j for record in records)
            )
            total_energy_j += float(sum(record.total_energy_j for record in records))
            uav_arrival_tasks += sum(
                1 for record in records if record.target_node_type == "uav"
            )
            bs_arrival_tasks += sum(
                1 for record in records if record.target_node_type == "bs"
            )
            leo_arrival_tasks += sum(
                1 for record in records if record.target_node_type == "leo"
            )
            for record in records:
                delay = float(record.actual_finish_delay_s)
                all_delays.append(delay)
                if record.completed_before_deadline:
                    completed_delays.append(delay)
                else:
                    failed_delays.append(delay)

        update_result = system.update(trajectory) if system is not None else None
        timeout_or_drop_tasks = total_tasks - completed_tasks
        logger.log(
            record_type="episode",
            algorithm="cmppo",
            redundancy_mode=redundancy_mode,
            episode=episode,
            episode_shared_reward=float(sum(rewards)),
            episode_actor_loss=(
                update_result.actor_loss if update_result is not None else None
            ),
            episode_critic_loss=(
                update_result.critic_loss if update_result is not None else None
            ),
            episode_policy_entropy=(
                update_result.entropy if update_result is not None else None
            ),
            episode_updated_agents=(
                update_result.updated_agents if update_result is not None else 0
            ),
            episode_known_agents=len(system.agents),
            episode_system_profit=system_profit,
            episode_transmission_energy_j=transmission_energy_j,
            episode_computing_energy_j=computing_energy_j,
            episode_total_energy_j=total_energy_j,
            episode_total_tasks=total_tasks,
            episode_completed_tasks=completed_tasks,
            episode_timeout_or_drop_tasks=timeout_or_drop_tasks,
            episode_deadline_failure_tasks=deadline_failure_tasks,
            episode_capacity_drop_tasks=capacity_drop_tasks,
            episode_reliability_failure_tasks=reliability_failure_tasks,
            episode_redundancy_requested_tasks=redundancy_requested_tasks,
            episode_redundant_tasks=redundancy_requested_tasks,
            episode_admitted_redundant_tasks=redundant_tasks,
            episode_redundancy_success_tasks=redundancy_success_tasks,
            episode_task_completion_rate=_rate(completed_tasks, total_tasks),
            episode_task_timeout_or_drop_rate=_rate(timeout_or_drop_tasks, total_tasks),
            episode_task_deadline_failure_rate=_rate(
                deadline_failure_tasks, total_tasks
            ),
            episode_task_capacity_drop_rate=_rate(capacity_drop_tasks, total_tasks),
            episode_reliability_failure_rate=_rate(
                reliability_failure_tasks, total_tasks
            ),
            episode_redundancy_request_rate=_rate(
                redundancy_requested_tasks, total_tasks
            ),
            episode_redundancy_rate=_rate(redundancy_requested_tasks, total_tasks),
            episode_backup_admission_rate=_rate(
                redundant_tasks, redundancy_requested_tasks
            ),
            episode_redundancy_success_rate=_rate(
                redundancy_success_tasks, redundancy_requested_tasks
            ),
            episode_avg_task_completion_delay_s=_mean(completed_delays),
            episode_avg_actual_finish_delay_all_tasks_s=_mean(all_delays),
            episode_avg_actual_finish_delay_failed_tasks_s=_mean(failed_delays),
            episode_uav_arrival_tasks=uav_arrival_tasks,
            episode_bs_arrival_tasks=bs_arrival_tasks,
            episode_leo_arrival_tasks=leo_arrival_tasks,
            episode_uav_arrival_rate=_rate(uav_arrival_tasks, total_tasks),
            episode_bs_arrival_rate=_rate(bs_arrival_tasks, total_tasks),
            episode_leo_arrival_rate=_rate(leo_arrival_tasks, total_tasks),
        )

        if progress_interval > 0 and (
            (episode + 1) % progress_interval == 0 or episode + 1 == episodes
        ):
            actor_loss = update_result.actor_loss if update_result is not None else 0.0
            critic_loss = update_result.critic_loss if update_result is not None else 0.0
            print(
                f"[cmppo] {episode + 1:>4}/{episodes} "
                f"completion={_rate(completed_tasks, total_tasks):.4f} "
                f"actor={actor_loss:.4f} critic={critic_loss:.4f} "
                f"elapsed={time.perf_counter() - started_at:.1f}s",
                flush=True,
            )

    if system is None:
        raise RuntimeError("CMPPO system was not initialized")
    return logger, system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an isolated centralized multi-agent PPO baseline."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson task arrival rate in tasks/s; defaults to the environment preset.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--env", choices=["small", "medium", "training"], default="training"
    )
    parser.add_argument(
        "--task-mode", choices=["independent", "workflow"], default="independent"
    )
    parser.add_argument("--scenario", choices=TASK_SCENARIO_NAMES, default="balanced")
    parser.add_argument(
        "--redundancy-mode", choices=["none", "hybrid"], default="hybrid"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--actor-attention",
        action="store_true",
        help="Use the same self-attention Actor encoder available to CMADDPG.",
    )
    parser.add_argument(
        "--output", default="outputs/metrics/cmppo_hybrid_500.json"
    )
    parser.add_argument(
        "--checkpoint-output", default="outputs/checkpoints/cmppo_hybrid_500.pt"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    builders = {
        "small": build_small_scale_env,
        "medium": build_medium_env,
        "training": build_training_env,
    }
    env = builders[args.env](
        enable_redundancy=args.redundancy_mode == "hybrid",
        task_mode=args.task_mode,
        scenario_name=args.scenario,
        arrival_rate_tasks_per_s=args.arrival_rate,
        seed=args.seed,
    )
    config = CMPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        use_actor_self_attention=args.actor_attention,
    )
    print(
        f"[cmppo] device={device} env={args.env} task_mode={args.task_mode} "
        f"scenario={args.scenario} redundancy_mode={args.redundancy_mode} "
        f"actor_attention={args.actor_attention}",
        flush=True,
    )
    logger, system = run_cmppo(
        env=env,
        episodes=args.episodes,
        steps=args.steps,
        device=device,
        redundancy_mode=args.redundancy_mode,
        config=config,
        progress_interval=args.progress_interval,
    )
    effective_arrival_rate = env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s
    for record in logger.records:
        record.setdefault("arrival_rate_tasks_per_s", effective_arrival_rate)
        record.setdefault("arrival_scope", "system")
        record.setdefault("seed", args.seed)
        record.setdefault("scenario", args.scenario)
        record.setdefault("actor_attention", args.actor_attention)
    metrics_path = PROJECT_ROOT / args.output
    logger.to_json(metrics_path)
    checkpoint_path = system.save(PROJECT_ROOT / args.checkpoint_output)
    print(f"metrics={metrics_path}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
