from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import TASK_SCENARIO_NAMES
from src.action_space import ActionSpec
from src.action_space import MultiTaskOffloadingAction
from src.action_space import SlotAction
from src.metrics_logger import MetricsLogger
from train import build_medium_env
from train import build_small_scale_env
from train import build_training_env


def sample_random_non_redundant_action(
    action_spec: ActionSpec,
    rng: np.random.Generator,
) -> MultiTaskOffloadingAction:
    """Sample real legal targets while explicitly disabling redundancy."""

    slot_actions: list[SlotAction] = []
    for slot_index, mask in enumerate(action_spec.slot_target_masks):
        node_ids = (
            action_spec.slot_target_node_ids[slot_index]
            if action_spec.slot_target_node_ids is not None
            else action_spec.target_node_ids
        )
        valid_indices = [
            index
            for index, (allowed, node_id) in enumerate(zip(mask, node_ids))
            if allowed and bool(node_id)
        ]
        if valid_indices:
            target_index = int(rng.choice(valid_indices))
            target_node_id = node_ids[target_index]
        else:
            # Empty task slots are ignored by CMADDPGEnv.step, but still need a
            # structurally valid action entry.
            target_node_id = next((node_id for node_id in node_ids if node_id), "")

        slot_actions.append(
            SlotAction(
                target_node_id=target_node_id,
                priority_eta=0.5,
                redundancy_eta=0.0,
                backup_target_node_id=None,
            )
        )
    return MultiTaskOffloadingAction(slot_actions=slot_actions)


def sample_fully_random_non_redundant_actions(
    env,
    rng: np.random.Generator,
) -> dict[str, MultiTaskOffloadingAction]:
    """Sample uniformly from ingress UAV, every BS, and LEO without screening."""

    actions: dict[str, MultiTaskOffloadingAction] = {}
    all_bs_ids = [bs.node_id for bs in env.base_env.base_stations]
    leo_id = env.base_env.leo_satellite.node_id

    for agent_id, context in env.pending_contexts.items():
        slot_actions: list[SlotAction] = []
        for task_instance, ingress_uav, accepted_target_ids in zip(
            context.task_slots,
            context.ingress_uav_slots,
            context.slot_target_node_ids,
        ):
            if task_instance is None:
                target_node_id = next(
                    (node_id for node_id in accepted_target_ids if node_id),
                    ingress_uav.node_id,
                )
            else:
                target_pool = [ingress_uav.node_id, *all_bs_ids, leo_id]
                target_node_id = str(rng.choice(target_pool))

                # CMADDPGEnv normally accepts only its screened Top-K target IDs.
                # Register this baseline's sampled target so step() passes it to
                # the base environment unchanged.
                if target_node_id not in accepted_target_ids:
                    accepted_target_ids.append(target_node_id)

            slot_actions.append(
                SlotAction(
                    target_node_id=target_node_id,
                    priority_eta=0.5,
                    redundancy_eta=0.0,
                    backup_target_node_id=None,
                )
            )
        actions[agent_id] = MultiTaskOffloadingAction(slot_actions=slot_actions)

    return actions


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_random_offloading(
    *,
    env,
    episodes: int,
    steps: int,
    seed: int,
    progress_interval: int,
    target_pool: str = "screened",
) -> MetricsLogger:
    """Evaluate uniform random offloading and return episode-level metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(seed)
    logger = MetricsLogger()
    started_at = time.perf_counter()

    for episode in range(episodes):
        _, action_specs = env.reset()
        episode_rewards: list[float] = []
        all_delays: list[float] = []
        completed_delays: list[float] = []
        failed_delays: list[float] = []
        total_tasks = 0
        completed_tasks = 0
        deadline_failure_tasks = 0
        capacity_drop_tasks = 0
        reliability_failure_tasks = 0
        redundancy_requested_tasks = 0
        transmission_energy_j = 0.0
        computing_energy_j = 0.0
        total_energy_j = 0.0
        system_profit = 0.0
        uav_arrival_tasks = 0
        bs_arrival_tasks = 0
        leo_arrival_tasks = 0

        for _ in range(steps):
            if target_pool == "all":
                actions = sample_fully_random_non_redundant_actions(env, rng)
            else:
                actions = {
                    agent_id: sample_random_non_redundant_action(action_spec, rng)
                    for agent_id, action_spec in action_specs.items()
                }
            _, _, _, info = env.step(actions)
            action_specs = info["action_specs"]
            records = info["records"]

            episode_rewards.append(float(info["shared_reward"]))
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

        timeout_or_drop_tasks = total_tasks - completed_tasks
        logger.log(
            record_type="episode",
            policy=(
                "fully_random_offloading"
                if target_pool == "all"
                else "random_offloading"
            ),
            redundancy_mode="none",
            episode=episode,
            episode_shared_reward=float(sum(episode_rewards)),
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
            episode_redundant_tasks=0,
            episode_redundancy_success_tasks=0,
            episode_task_completion_rate=_rate(completed_tasks, total_tasks),
            episode_task_timeout_or_drop_rate=_rate(timeout_or_drop_tasks, total_tasks),
            episode_task_deadline_failure_rate=_rate(
                deadline_failure_tasks, total_tasks
            ),
            episode_task_capacity_drop_rate=_rate(capacity_drop_tasks, total_tasks),
            episode_reliability_failure_rate=_rate(
                reliability_failure_tasks, total_tasks
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
            elapsed_s = time.perf_counter() - started_at
            print(
                f"[random] {episode + 1:>4}/{episodes} "
                f"completion={_rate(completed_tasks, total_tasks):.4f} "
                f"tasks={total_tasks} elapsed={elapsed_s:.1f}s",
                flush=True,
            )

    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible, non-redundant random-offloading baseline."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Random policy seed.")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson task arrival rate in tasks/s; defaults to the environment preset.",
    )
    parser.add_argument("--progress-interval", type=int, default=10)
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
    parser.add_argument(
        "--target-pool",
        choices=["screened", "all"],
        default="screened",
        help=(
            "screened: ingress UAV + ranked Top-9 reachable BS + LEO; "
            "all: ingress UAV + every BS + LEO, without reachability/ranking screening."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics/random_offloading_500.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builders = {
        "small": build_small_scale_env,
        "medium": build_medium_env,
        "training": build_training_env,
    }
    env = builders[args.env](
        enable_redundancy=False,
        task_mode=args.task_mode,
        scenario_name=args.scenario,
        arrival_rate_tasks_per_s=args.arrival_rate,
        seed=args.seed,
    )
    print(
        f"[random] env={args.env} task_mode={args.task_mode} "
        f"scenario={args.scenario} redundancy_mode=none "
        f"target_pool={args.target_pool} seed={args.seed}",
        flush=True,
    )
    logger = run_random_offloading(
        env=env,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        progress_interval=args.progress_interval,
        target_pool=args.target_pool,
    )
    effective_arrival_rate = env.base_env.task_generator.task_model_config.arrival_rate_tasks_per_s
    for record in logger.records:
        record.setdefault("arrival_rate_tasks_per_s", effective_arrival_rate)
        record.setdefault("arrival_scope", "system")
        record.setdefault("seed", args.seed)
        record.setdefault("scenario", args.scenario)
    output_path = PROJECT_ROOT / args.output
    logger.to_json(output_path)
    print(f"metrics={output_path} episode_records={len(logger.records)}")


if __name__ == "__main__":
    main()
