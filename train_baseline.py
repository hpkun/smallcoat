from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.baseline_cmaddpg import BaselineCMADDPGSystem
from src.baseline_env import BaselineCMADDPGEnv
from src.baseline_observation import BaselineObservationBuilder
from src.maddpg_agent import AgentHyperParameters
from src.reward import RewardConfig
from src.reward import SharedRewardCalculator
from src.scenario_generator import TASK_SCENARIO_NAMES
from src.trainer import CMADDPGTrainer
from src.trainer import TrainerConfig
from train import build_medium_env
from train import build_small_scale_env
from train import build_training_env


PROJECT_ROOT = Path(__file__).resolve().parent


def build_baseline_reward_config() -> RewardConfig:
    """Pure system-profit reward with every Proposed shaping term disabled."""

    return replace(
        RewardConfig(normalize_profit_scale=1_000_000_000.0),
        deadline_failure_penalty=0.0,
        capacity_drop_penalty=0.0,
        reliability_failure_penalty=0.0,
        completion_delay_penalty=0.0,
        energy_penalty_weight=0.0,
        completion_constraint_dual_lr=0.0,
        long_term_energy_budget_j_per_step=None,
        energy_constraint_dual_lr=0.0,
        advantage_reward_weight=0.0,
    )


def build_baseline_env(
    *,
    env_name: str = "training",
    task_mode: str = "independent",
    scenario_name: str = "balanced",
    arrival_rate_tasks_per_s: float | None = None,
    seed: int | None = None,
) -> BaselineCMADDPGEnv:
    """Build B0 on the same physical environment used by Proposed."""

    builders = {
        "small": build_small_scale_env,
        "medium": build_medium_env,
        "training": build_training_env,
    }
    if env_name not in builders:
        raise ValueError(f"Unknown environment preset: {env_name}")
    source_env = builders[env_name](
        enable_redundancy=False,
        enable_resource_awareness=False,
        task_mode=task_mode,
        scenario_name=scenario_name,
        arrival_rate_tasks_per_s=arrival_rate_tasks_per_s,
        seed=seed,
    )
    base_env = source_env.base_env
    observation_builder = BaselineObservationBuilder(
        communication_model=base_env.communication_model,
        network_profiles=base_env.network_profiles,
        area_side_length_m=base_env.simulation_config.area.side_length_m,
        energy_config=base_env.simulation_config.energy,
        queue_capacity=base_env.simulation_config.queue_capacity,
    )
    return BaselineCMADDPGEnv(
        base_env=base_env,
        observation_builder=observation_builder,
        reward_calculator=SharedRewardCalculator(build_baseline_reward_config()),
        task_mode=task_mode,
        workflow_generator=source_env.workflow_generator,
        workflow_encoder=source_env.workflow_encoder,
    )


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train CMADDPG-Baseline: KMDUC, single-copy actions, baseline "
            "observations, and profit-only reward."
        )
    )
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--output",
        default="outputs/cmaddpg_baseline_seed42/train_metrics.json",
    )
    parser.add_argument(
        "--checkpoint-output",
        default="outputs/cmaddpg_baseline_seed42/cmaddpg_baseline.pt",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--env",
        choices=["small", "medium", "training"],
        default="training",
    )
    parser.add_argument(
        "--task-mode",
        choices=["independent", "workflow"],
        default="independent",
    )
    parser.add_argument(
        "--scenario",
        choices=TASK_SCENARIO_NAMES,
        default="balanced",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = build_baseline_env(
        env_name=args.env,
        task_mode=args.task_mode,
        scenario_name=args.scenario,
        arrival_rate_tasks_per_s=args.arrival_rate,
        seed=args.seed,
    )
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    reward_config = env.reward_calculator.config
    print(
        "[baseline] "
        f"device={device} env={args.env} task_mode={args.task_mode} "
        f"scenario={args.scenario} observation=baseline action=single-copy "
        "attention=off resource_awareness=off redundancy=off "
        "reward=profit-only energy_dual=off physical_battery=on",
        flush=True,
    )

    system = BaselineCMADDPGSystem(
        device=device,
        agent_hyper_params=AgentHyperParameters(
            use_actor_self_attention=False,
            use_actor_resource_awareness=False,
        ),
    )
    manager = env.base_env.clustering_manager
    if manager is not None:
        system.configure_agent_pool(manager.logical_agent_ids)
    metrics_path = _project_path(args.output)
    checkpoint_path = _project_path(args.checkpoint_output)
    trainer = CMADDPGTrainer(
        env=env,
        system=system,
        config=TrainerConfig(
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            batch_size=args.batch_size,
            progress_print_interval=args.progress_interval,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
    )
    logger = trainer.train()
    effective_arrival_rate = (
        env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s
    )
    metadata = {
        "algorithm": "cmaddpg-baseline",
        "baseline_definition": "B0",
        "arrival_rate_tasks_per_s": effective_arrival_rate,
        "arrival_scope": "system",
        "seed": args.seed,
        "observation_profile": "baseline",
        "action_profile": "single-copy",
        "redundancy_mode": "none",
        "actor_attention": False,
        "resource_awareness": False,
        "reliability_awareness": False,
        "reward_profile": "profit-only",
        "energy_penalty_weight": reward_config.energy_penalty_weight,
        "long_term_energy_budget_j_per_step": (
            reward_config.long_term_energy_budget_j_per_step
        ),
        "energy_constraint_dual_lr": reward_config.energy_constraint_dual_lr,
        "physical_failures_enabled": True,
        "physical_battery_constraint_enabled": True,
        "energy_accounting_enabled": True,
    }
    for record in logger.records:
        for key, value in metadata.items():
            record.setdefault(key, value)
    logger.to_json(metrics_path)
    saved_checkpoint = system.save(checkpoint_path)
    print(f"checkpoint={saved_checkpoint}")
    print(f"metrics={metrics_path}")
    print(f"training_logs={len(logger.records)}")


if __name__ == "__main__":
    main()
