from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cmaddpg import CMADDPGSystem
from src.cmppo import CMPPOConfig
from src.cmppo import CMPPOSystem
from src.metrics_logger import MetricsLogger
from src.maddpg_agent import AgentHyperParameters
from src.observation_builder import OBSERVATION_INPUT_DIM
from train import build_medium_env
from train import build_small_scale_env
from train import build_training_env


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate_checkpoint(
    *,
    algorithm: str,
    checkpoint: Path,
    env,
    episodes: int,
    steps: int,
    seed: int,
    redundancy_mode: str,
    actor_attention: bool,
    resource_awareness: bool,
    log_steps: bool = False,
) -> MetricsLogger:
    if algorithm not in {"cmaddpg", "cmppo"}:
        raise ValueError("algorithm must be cmaddpg or cmppo")
    logger = MetricsLogger()
    system = None
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_data.get("architecture") != "variable_task_v1":
        raise ValueError(
            "The checkpoint uses the retired fixed-task-slot architecture. "
            "Train a new variable-task checkpoint first."
        )

    global_time_slot = 0
    global_cumulative_profit = 0.0
    for episode in range(episodes):
        observations, action_specs = env.reset()
        if system is None:
            if algorithm == "cmaddpg":
                system = CMADDPGSystem(
                    device="cpu",
                    agent_hyper_params=AgentHyperParameters(
                        use_actor_self_attention=actor_attention,
                        use_actor_resource_awareness=resource_awareness,
                    ),
                )
                for agent_id, observation in observations.items():
                    system.ensure_agent(agent_id, int(observation.shape[0]), action_specs[agent_id])
                # A checkpoint's centralized critics may include CH agents that
                # are not active in the first evaluation reset. Materialize all
                # saved agents before rebuilding the joint critic dimensions.
                template_spec = next(iter(action_specs.values()))
                template_state_dim = (
                    template_spec.num_task_slots * OBSERVATION_INPUT_DIM
                )
                for agent_id in checkpoint_data.get("agents", {}):
                    system.ensure_agent(agent_id, template_state_dim, template_spec)
                system.rebuild_joint_critics()
                system.load(checkpoint)
            else:
                first_id = sorted(observations)[0]
                system = CMPPOSystem(
                    state_dim=int(observations[first_id].shape[0]),
                    action_spec=action_specs[first_id],
                    device="cpu",
                    redundancy_mode=redundancy_mode,
                    config=CMPPOConfig(use_actor_self_attention=actor_attention),
                )
                system.sample_actions(observations, action_specs, deterministic=True)
                template_spec = next(iter(action_specs.values()))
                template_state_dim = (
                    template_spec.num_task_slots * OBSERVATION_INPUT_DIM
                )
                for agent_id in checkpoint_data.get("agents", {}):
                    system.ensure_agent(agent_id, template_state_dim, template_spec)
                system._rebuild_joint_critics()
                system.load(checkpoint)

        total_profit = 0.0
        total_tasks = 0
        completed_tasks = 0
        reliable_tasks = 0
        total_energy = 0.0
        for step in range(steps):
            if algorithm == "cmaddpg":
                for agent_id, observation in observations.items():
                    system.ensure_agent(agent_id, int(observation.shape[0]), action_specs[agent_id])
                raw_actions = system.act(observations, add_noise=False)
                actions, _ = system.decode_actions(raw_actions)
            else:
                actions, _ = system.sample_actions(
                    observations,
                    action_specs,
                    deterministic=True,
                )
            observations, _, _, info = env.step(actions)
            action_specs = info["action_specs"]
            records = info["records"]
            step_profit = float(info["equation8_objective"].total_profit)
            total_profit += step_profit
            global_time_slot += 1
            global_cumulative_profit += step_profit
            total_tasks += len(records)
            completed_tasks += sum(record.completed_before_deadline for record in records)
            reliable_tasks += sum(
                record.completed_before_deadline
                and record.satisfies_reliability
                and not record.failed_due_to_reliability
                for record in records
            )
            total_energy += sum(record.total_energy_j for record in records)
            if log_steps:
                logger.log(
                    record_type="evaluation_step",
                    algorithm=algorithm,
                    evaluation=True,
                    seed=seed,
                    episode=episode,
                    step=step,
                    time_slot=global_time_slot,
                    system_profit=step_profit,
                    cumulative_system_profit=global_cumulative_profit,
                    task_count=len(records),
                    completed_task_count=sum(
                        record.completed_before_deadline for record in records
                    ),
                    arrival_rate_tasks_per_s=(
                        env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s
                    ),
                    arrival_scope="system",
                    redundancy_mode=redundancy_mode,
                    actor_attention=actor_attention,
                )

        logger.log(
            record_type="episode",
            algorithm=algorithm,
            evaluation=True,
            seed=seed,
            episode=episode,
            arrival_rate_tasks_per_s=env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s,
            arrival_scope="system",
            redundancy_mode=redundancy_mode,
            actor_attention=actor_attention,
            episode_system_profit=total_profit,
            episode_total_energy_j=float(total_energy),
            episode_total_tasks=total_tasks,
            episode_completed_tasks=completed_tasks,
            episode_reliable_on_time_tasks=reliable_tasks,
            episode_task_completion_rate=_rate(completed_tasks, total_tasks),
            episode_reliable_on_time_completion_rate=_rate(reliable_tasks, total_tasks),
        )
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen CMADDPG or CMPPO checkpoint.")
    parser.add_argument("--algorithm", choices=["cmaddpg", "cmppo"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", choices=["small", "medium", "training"], default="training")
    parser.add_argument("--task-mode", choices=["independent", "workflow"], default="independent")
    parser.add_argument("--scenario", default="balanced")
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--redundancy-mode", choices=["none", "hybrid"], default="none")
    parser.add_argument("--actor-attention", action="store_true")
    parser.add_argument("--resource-awareness", action="store_true")
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Include one cumulative-profit record per evaluation time slot.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    builders = {
        "small": build_small_scale_env,
        "medium": build_medium_env,
        "training": build_training_env,
    }
    env = builders[args.env](
        enable_redundancy=args.redundancy_mode == "hybrid",
        enable_resource_awareness=args.resource_awareness,
        task_mode=args.task_mode,
        scenario_name=args.scenario,
        arrival_rate_tasks_per_s=args.arrival_rate,
        seed=args.seed,
    )
    logger = evaluate_checkpoint(
        algorithm=args.algorithm,
        checkpoint=PROJECT_ROOT / args.checkpoint,
        env=env,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        redundancy_mode=args.redundancy_mode,
        actor_attention=args.actor_attention,
        resource_awareness=args.resource_awareness,
        log_steps=args.log_steps,
    )
    logger.to_json(PROJECT_ROOT / args.output)
    print(f"metrics={PROJECT_ROOT / args.output} episodes={len(logger.records)}")


if __name__ == "__main__":
    main()
