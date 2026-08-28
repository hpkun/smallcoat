from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import AreaConfig
from src import BaseStation
from src import ClusteringConfig
from src import CMADDPGEnv
from src import CMADDPGSystem
from src import CMADDPGTrainer
from src import CommunicationModel
from src import KMDUCManager
from src import LEOSatellite
from src import MobilityConfig
from src import ObservationBuilder
from src import Position
from src import RewardConfig
from src import QueueCapacityConfig
from src import SAGINEnvironment
from src import SharedRewardCalculator
from src import SimulationConfig
from src import TaskGenerator
from src import TaskModelConfig
from src import TASK_SCENARIO_NAMES
from src import TrainerConfig
from src import UAV
from src import UniformRange
from src import SyntheticWorkflowGenerator
from src import WorkflowModelConfig
from src import build_task_scenario
from src.maddpg_agent import AgentHyperParameters
from src.config import build_default_network_profiles
from src.task_model import MBIT
from src.task_model import MCYCLE


def build_reward_config(
    profile: str = "proposed",
    *,
    energy_penalty_weight: float | None = None,
) -> RewardConfig:
    if profile not in {"proposed", "profit-only"}:
        raise ValueError("profile must be 'proposed' or 'profit-only'.")
    config = RewardConfig(normalize_profit_scale=1_000_000_000.0)
    if profile == "profit-only":
        config = replace(
            config,
            deadline_failure_penalty=0.0,
            capacity_drop_penalty=0.0,
            reliability_failure_penalty=0.0,
            completion_delay_penalty=0.0,
            energy_penalty_weight=0.0,
            completion_constraint_dual_lr=0.0,
            advantage_reward_weight=0.0,
        )
    if energy_penalty_weight is not None:
        if energy_penalty_weight < 0.0:
            raise ValueError("energy_penalty_weight must be non-negative.")
        config = replace(config, energy_penalty_weight=float(energy_penalty_weight))
    return config


def build_medium_env(
    *,
    enable_redundancy: bool = True,
    enable_resource_awareness: bool = False,
    task_mode: str = "independent",
    scenario_name: str = "balanced",
    arrival_rate_tasks_per_s: float | None = None,
    seed: int | None = None,
) -> CMADDPGEnv:
    simulation_config = SimulationConfig(
        slot_length_s=0.1,
        rng_seed=21 if seed is None else int(seed),
        queue_capacity=QueueCapacityConfig(),
        area=AreaConfig(side_length_m=5_000.0),
        mobility=MobilityConfig(mean_speed_m_per_s=15.0, std_speed_m_per_s=3.0),
        clustering=ClusteringConfig(
            communication_radius_m=1_000.0,
            clustering_period_slots=8,
            ch_reselection_slots=3,
        ),
        network_profiles=build_default_network_profiles(),
    )
    rng = np.random.default_rng(simulation_config.rng_seed)
    task_config = TaskModelConfig(
        arrival_rate_tasks_per_s=5.0,
        input_size_bits=UniformRange(10 * MBIT, 90 * MBIT),
        total_compute_cycles=UniformRange(1000 * MCYCLE, 4000 * MCYCLE),
        tolerable_latency_s=UniformRange(0.0, 0.2),
        parallel_efficiency=UniformRange(0.85, 1.0),
        delay_sensitivity_lambda=4.0,
    )
    task_config = build_task_scenario(scenario_name, task_config).task_config
    if arrival_rate_tasks_per_s is not None:
        if arrival_rate_tasks_per_s <= 0:
            raise ValueError("arrival_rate_tasks_per_s must be positive")
        task_config = replace(
            task_config,
            arrival_rate_tasks_per_s=float(arrival_rate_tasks_per_s),
        )
    task_generator = TaskGenerator(task_model_config=task_config)
    workflow_generator = SyntheticWorkflowGenerator(
        task_model_config=task_config,
        workflow_model_config=WorkflowModelConfig(
            arrival_rate_workflows_per_s=2.0,
            task_count=UniformRange(3, 6),
        ),
    )

    num_uavs = 40
    num_base_stations = 25
    side = simulation_config.area.side_length_m

    uavs = [
        UAV(
            f"uav-{idx}",
            Position(
                float(rng.uniform(0.0, side)),
                float(rng.uniform(0.0, side)),
                140.0,
            ),
            compute_capacity_cycles_per_s=12e9,
            execution_failure_rate=float(rng.uniform(0.08, 0.25)),
            restart_time_s=0.4,
            speed_m_per_s=max(1.0, float(rng.normal(15.0, 3.0))),
            heading_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for idx in range(num_uavs)
    ]
    base_stations = [
        BaseStation(
            f"bs-{idx}",
            Position(
                float(rng.uniform(0.0, side)),
                float(rng.uniform(0.0, side)),
                0.0,
            ),
            compute_capacity_cycles_per_s=float(rng.uniform(30e9, 50e9)),
            execution_failure_rate=float(rng.uniform(0.03, 0.12)),
            restart_time_s=0.4,
        )
        for idx in range(num_base_stations)
    ]
    leo_satellite = LEOSatellite(
        "leo-0",
        Position(side / 2.0, side / 2.0, 550_000.0),
        60e9,
        execution_failure_rate=0.02,
        restart_time_s=0.4,
    )

    base_env = SAGINEnvironment(
        uavs=uavs,
        base_stations=base_stations,
        leo_satellite=leo_satellite,
        communication_model=CommunicationModel(),
        network_profiles=simulation_config.network_profiles,
        task_generator=task_generator,
        simulation_config=simulation_config,
        clustering_manager=KMDUCManager(
            clustering_config=simulation_config.clustering,
            area_config=simulation_config.area,
        ),
        rng=rng,
        enable_redundancy=enable_redundancy,
    )
    observation_builder = ObservationBuilder(
        communication_model=base_env.communication_model,
        network_profiles=base_env.network_profiles,
        area_side_length_m=simulation_config.area.side_length_m,
        enable_resource_awareness=enable_resource_awareness,
    )
    reward_calculator = SharedRewardCalculator(
        RewardConfig(normalize_profit_scale=1_000_000_000.0)
    )
    return CMADDPGEnv(
        base_env=base_env,
        observation_builder=observation_builder,
        reward_calculator=reward_calculator,
        task_mode=task_mode,
        workflow_generator=workflow_generator,
    )


def build_training_env(
    *,
    enable_redundancy: bool = True,
    enable_resource_awareness: bool = False,
    task_mode: str = "independent",
    scenario_name: str = "balanced",
    arrival_rate_tasks_per_s: float | None = None,
    seed: int | None = None,
) -> CMADDPGEnv:
    simulation_config = SimulationConfig(
        slot_length_s=0.1,
        rng_seed=42 if seed is None else int(seed),
        queue_capacity=QueueCapacityConfig(),
        area=AreaConfig(side_length_m=5_000.0),
        mobility=MobilityConfig(mean_speed_m_per_s=10.0, std_speed_m_per_s=2.0),
        clustering=ClusteringConfig(communication_radius_m=1_200.0),
        network_profiles=build_default_network_profiles(),
    )
    rng = np.random.default_rng(simulation_config.rng_seed)
    task_config = TaskModelConfig(
        arrival_rate_tasks_per_s=25.0,
        tolerable_latency_s=UniformRange(0.0, 0.5),
        delay_sensitivity_lambda=6.0,
    )
    task_config = build_task_scenario(scenario_name, task_config).task_config
    if arrival_rate_tasks_per_s is not None:
        if arrival_rate_tasks_per_s <= 0:
            raise ValueError("arrival_rate_tasks_per_s must be positive")
        task_config = replace(
            task_config,
            arrival_rate_tasks_per_s=float(arrival_rate_tasks_per_s),
        )
    task_generator = TaskGenerator(task_model_config=task_config)
    workflow_generator = SyntheticWorkflowGenerator(
        task_model_config=task_config,
        workflow_model_config=WorkflowModelConfig(
            arrival_rate_workflows_per_s=5.0,
            task_count=UniformRange(3, 6),
        ),
    )

    num_uavs = 40
    num_base_stations = 25
    side = simulation_config.area.side_length_m

    uavs = [
        UAV(
            f"uav-{idx}",
            Position(
                float(rng.uniform(0.0, side)),
                float(rng.uniform(0.0, side)),
                150.0,
            ),
            compute_capacity_cycles_per_s=10e9,
            execution_failure_rate=float(rng.uniform(0.08, 0.25)),
            restart_time_s=0.4,
            speed_m_per_s=max(1.0, float(rng.normal(20.0, 5.0))),
            heading_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for idx in range(num_uavs)
    ]
    base_stations = [
        BaseStation(
            f"bs-{idx}",
            Position(
                float(rng.uniform(0.0, side)),
                float(rng.uniform(0.0, side)),
                0.0,
            ),
            compute_capacity_cycles_per_s=float(rng.uniform(20e9, 40e9)),
            execution_failure_rate=float(rng.uniform(0.03, 0.12)),
            restart_time_s=0.4,
        )
        for idx in range(num_base_stations)
    ]
    leo_satellite = LEOSatellite(
        "leo-0",
        Position(2500.0, 2500.0, 550_000.0),
        60e9,
        execution_failure_rate=0.02,
        restart_time_s=0.4,
    )

    base_env = SAGINEnvironment(
        uavs=uavs,
        base_stations=base_stations,
        leo_satellite=leo_satellite,
        communication_model=CommunicationModel(),
        network_profiles=simulation_config.network_profiles,
        task_generator=task_generator,
        simulation_config=simulation_config,
        clustering_manager=KMDUCManager(
            clustering_config=simulation_config.clustering,
            area_config=simulation_config.area,
        ),
        rng=rng,
        enable_redundancy=enable_redundancy,
    )
    observation_builder = ObservationBuilder(
        communication_model=base_env.communication_model,
        network_profiles=base_env.network_profiles,
        area_side_length_m=simulation_config.area.side_length_m,
        enable_resource_awareness=enable_resource_awareness,
    )
    reward_calculator = SharedRewardCalculator(RewardConfig(normalize_profit_scale=1_000_000_000.0))
    return CMADDPGEnv(
        base_env=base_env,
        observation_builder=observation_builder,
        reward_calculator=reward_calculator,
        task_mode=task_mode,
        workflow_generator=workflow_generator,
    )


def build_small_scale_env(
    *,
    enable_redundancy: bool = True,
    enable_resource_awareness: bool = False,
    task_mode: str = "independent",
    scenario_name: str = "balanced",
    arrival_rate_tasks_per_s: float | None = None,
    seed: int | None = None,
) -> CMADDPGEnv:
    simulation_config = SimulationConfig(
        slot_length_s=0.1,
        rng_seed=7 if seed is None else int(seed),
        queue_capacity=QueueCapacityConfig(),
        area=AreaConfig(side_length_m=2_000.0),
        mobility=MobilityConfig(mean_speed_m_per_s=12.0, std_speed_m_per_s=2.0),
        clustering=ClusteringConfig(
            communication_radius_m=900.0,
            clustering_period_slots=5,
            ch_reselection_slots=2,
        ),
        network_profiles=build_default_network_profiles(),
    )
    rng = np.random.default_rng(simulation_config.rng_seed)
    task_config = TaskModelConfig(
        arrival_rate_tasks_per_s=1.0,
        input_size_bits=UniformRange(1 * MBIT, 8 * MBIT),
        total_compute_cycles=UniformRange(50 * MCYCLE, 200 * MCYCLE),
        tolerable_latency_s=UniformRange(0.5, 1.5),
        parallel_efficiency=UniformRange(0.85, 1.0),
        delay_sensitivity_lambda=8.0,
    )
    task_config = build_task_scenario(scenario_name, task_config).task_config
    if arrival_rate_tasks_per_s is not None:
        if arrival_rate_tasks_per_s <= 0:
            raise ValueError("arrival_rate_tasks_per_s must be positive")
        task_config = replace(
            task_config,
            arrival_rate_tasks_per_s=float(arrival_rate_tasks_per_s),
        )
    task_generator = TaskGenerator(task_model_config=task_config)
    workflow_generator = SyntheticWorkflowGenerator(
        task_model_config=task_config,
        workflow_model_config=WorkflowModelConfig(
            arrival_rate_workflows_per_s=1.0,
            task_count=UniformRange(3, 5),
        ),
    )

    uavs = [
        UAV(
            "uav-0",
            Position(200.0, 180.0, 120.0),
            15e9,
            execution_failure_rate=0.12,
            restart_time_s=0.4,
            speed_m_per_s=8.0,
            heading_rad=0.4,
        ),
        UAV(
            "uav-1",
            Position(950.0, 500.0, 120.0),
            15e9,
            execution_failure_rate=0.18,
            restart_time_s=0.4,
            speed_m_per_s=9.0,
            heading_rad=2.6,
        ),
    ]
    base_stations = [
        BaseStation(
            "bs-0",
            Position(0.0, 0.0, 0.0),
            60e9,
            execution_failure_rate=0.06,
            restart_time_s=0.4,
        ),
        BaseStation(
            "bs-1",
            Position(1200.0, 800.0, 0.0),
            80e9,
            execution_failure_rate=0.04,
            restart_time_s=0.4,
        ),
    ]
    leo_satellite = LEOSatellite(
        "leo-0",
        Position(1000.0, 1000.0, 550_000.0),
        120e9,
        execution_failure_rate=0.02,
        restart_time_s=0.4,
    )

    base_env = SAGINEnvironment(
        uavs=uavs,
        base_stations=base_stations,
        leo_satellite=leo_satellite,
        communication_model=CommunicationModel(),
        network_profiles=simulation_config.network_profiles,
        task_generator=task_generator,
        simulation_config=simulation_config,
        clustering_manager=KMDUCManager(
            clustering_config=simulation_config.clustering,
            area_config=simulation_config.area,
        ),
        rng=rng,
        enable_redundancy=enable_redundancy,
    )
    observation_builder = ObservationBuilder(
        communication_model=base_env.communication_model,
        network_profiles=base_env.network_profiles,
        area_side_length_m=simulation_config.area.side_length_m,
        enable_resource_awareness=enable_resource_awareness,
    )
    reward_calculator = SharedRewardCalculator(RewardConfig(normalize_profit_scale=1_000_000_000.0))
    return CMADDPGEnv(
        base_env=base_env,
        observation_builder=observation_builder,
        reward_calculator=reward_calculator,
        task_mode=task_mode,
        workflow_generator=workflow_generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMADDPG on the paper-scale scenario.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used in updates.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Training and exploration seed.")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson task arrival rate in tasks/s; defaults to the environment preset.",
    )
    parser.add_argument("--progress-interval", type=int, default=10, help="How often to print progress rows.")
    parser.add_argument("--output", default="outputs/metrics/train_metrics.json", help="Path to metrics JSON output.")
    parser.add_argument(
        "--checkpoint-output",
        default=None,
        help="Optional CMADDPG checkpoint output path.",
    )
    parser.add_argument(
        "--env",
        choices=["small", "medium", "training"],
        default="training",
        help="Environment preset to train on.",
    )
    parser.add_argument(
        "--redundancy-mode",
        choices=["none", "hybrid"],
        default="hybrid",
        help="Use none for plain offloading baseline or hybrid for redundancy.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["independent", "workflow"],
        default="independent",
        help="Use independent for original Poisson tasks or workflow for synthetic DAG workflows.",
    )
    parser.add_argument(
        "--scenario",
        choices=TASK_SCENARIO_NAMES,
        default="balanced",
        help="Baseline-compatible task distribution.",
    )
    parser.add_argument(
        "--actor-attention",
        action="store_true",
        help="Enable the self-attention encoder in each actor network.",
    )
    parser.add_argument(
        "--resource-awareness",
        action="store_true",
        help="Replace candidate link slots with live compute, queue, and deadline-feasibility features.",
    )
    parser.add_argument(
        "--reward-profile",
        choices=["proposed", "profit-only"],
        default="proposed",
        help="Use profit-only for the plain CMADDPG reward baseline.",
    )
    parser.add_argument(
        "--energy-penalty-weight",
        type=float,
        default=None,
        help="Override the reward energy weight; use 0 to ignore energy in learning.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    enable_redundancy = args.redundancy_mode == "hybrid"
    if args.env == "small":
        env = build_small_scale_env(
            enable_redundancy=enable_redundancy,
            enable_resource_awareness=args.resource_awareness,
            task_mode=args.task_mode,
            scenario_name=args.scenario,
            arrival_rate_tasks_per_s=args.arrival_rate,
            seed=args.seed,
        )
    elif args.env == "medium":
        env = build_medium_env(
            enable_redundancy=enable_redundancy,
            enable_resource_awareness=args.resource_awareness,
            task_mode=args.task_mode,
            scenario_name=args.scenario,
            arrival_rate_tasks_per_s=args.arrival_rate,
            seed=args.seed,
        )
    else:
        env = build_training_env(
            enable_redundancy=enable_redundancy,
            enable_resource_awareness=args.resource_awareness,
            task_mode=args.task_mode,
            scenario_name=args.scenario,
            arrival_rate_tasks_per_s=args.arrival_rate,
            seed=args.seed,
        )
    reward_config = build_reward_config(
        args.reward_profile,
        energy_penalty_weight=args.energy_penalty_weight,
    )
    env.reward_calculator = SharedRewardCalculator(reward_config)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(
        f"[train] using device={device} env={args.env} "
        f"task_mode={args.task_mode} "
        f"scenario={args.scenario} "
        f"redundancy_mode={args.redundancy_mode} "
        f"actor_attention={args.actor_attention} "
        f"resource_awareness={args.resource_awareness} "
        f"reward_profile={args.reward_profile} "
        f"energy_penalty_weight={reward_config.energy_penalty_weight}",
        flush=True,
    )
    system = CMADDPGSystem(
        device=device,
        agent_hyper_params=AgentHyperParameters(
            use_actor_self_attention=args.actor_attention,
            use_actor_resource_awareness=args.resource_awareness,
        ),
    )
    trainer = CMADDPGTrainer(
        env=env,
        system=system,
        config=TrainerConfig(
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            batch_size=args.batch_size,
            progress_print_interval=args.progress_interval,
        ),
    )
    logger = trainer.train()
    effective_arrival_rate = env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s
    for record in logger.records:
        record.setdefault("algorithm", "cmaddpg")
        record.setdefault("arrival_rate_tasks_per_s", effective_arrival_rate)
        record.setdefault("arrival_scope", "system")
        record.setdefault("seed", args.seed)
        record.setdefault("redundancy_mode", args.redundancy_mode)
        record.setdefault("actor_attention", args.actor_attention)
        record.setdefault("resource_awareness", args.resource_awareness)
        record.setdefault("reward_profile", args.reward_profile)
        record.setdefault("energy_penalty_weight", reward_config.energy_penalty_weight)
    logger.to_json(PROJECT_ROOT / args.output)
    if args.checkpoint_output:
        checkpoint_path = system.save(PROJECT_ROOT / args.checkpoint_output)
        print(f"checkpoint={checkpoint_path}")
    print(f"training_logs={len(logger.records)}")


if __name__ == "__main__":
    main()
