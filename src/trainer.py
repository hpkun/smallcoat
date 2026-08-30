from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch

from .cmaddpg import CMADDPGSystem
from .metrics_logger import MetricsLogger
from .rl_env import CMADDPGEnv


@dataclass(frozen=True)
class TrainerConfig:
    """训练器配置。"""

    num_episodes: int = 200
    steps_per_episode: int = 50
    update_every: int = 10
    batch_size: int = 8
    progress_print_interval: int = 10
    checkpoint_interval: int = 0
    checkpoint_path: str | Path | None = None
    metrics_path: str | Path | None = None


class CMADDPGTrainer:
    """CMADDPG 训练器。"""

    def __init__(
        self,
        env: CMADDPGEnv,
        system: CMADDPGSystem,
        config: TrainerConfig | None = None,
    ) -> None:
        self.env = env
        self.system = system
        self.config = config or TrainerConfig()
        self.logger = MetricsLogger()

    def _active_logical_agent_ids(self) -> set[str]:
        manager = self.env.base_env.clustering_manager
        if manager is None:
            return set()
        return set(manager.active_agent_bindings())

    def _ensure_actor_pool(self, observations, action_specs) -> None:
        """Create the fixed logical Actor pool using any current action shape."""

        manager = self.env.base_env.clustering_manager
        if manager is not None and self.system.allowed_agent_ids is None:
            self.system.configure_agent_pool(manager.logical_agent_ids)
        if observations and self.system.allowed_agent_ids is not None:
            template_agent_id = next(iter(observations))
            template_spec = action_specs[template_agent_id]
            template_state_dim = int(observations[template_agent_id].shape[0])
            for agent_id in sorted(self.system.allowed_agent_ids):
                if agent_id not in self.system.actors:
                    self.system.ensure_agent(
                        agent_id=agent_id,
                        state_dim=template_state_dim,
                        action_spec=template_spec,
                    )
        for agent_id, observation in observations.items():
            self.system.ensure_agent(
                agent_id=agent_id,
                state_dim=int(observation.shape[0]),
                action_spec=action_specs[agent_id],
            )

    def _gpu_memory_metrics(self) -> tuple[float, float]:
        if self.system.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0, 0.0
        device = self.system.device
        gib = float(1024 ** 3)
        return (
            float(torch.cuda.memory_allocated(device) / gib),
            float(torch.cuda.memory_reserved(device) / gib),
        )

    def _periodic_checkpoint(self, completed_episodes: int) -> Path | None:
        if (
            self.config.checkpoint_interval <= 0
            or self.config.checkpoint_path is None
            or completed_episodes % self.config.checkpoint_interval != 0
        ):
            return None
        base_path = Path(self.config.checkpoint_path)
        checkpoint_path = base_path.with_name(
            f"{base_path.stem}_{completed_episodes}{base_path.suffix or '.pt'}"
        )
        saved_path = self.system.save(checkpoint_path)
        if self.config.metrics_path is not None:
            self.logger.to_json(self.config.metrics_path)
        print(f"[train] checkpoint={saved_path}", flush=True)
        return saved_path

    @staticmethod
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "       n/a"
        abs_value = abs(value)
        if abs_value >= 1_000:
            return f"{value:>10.3e}"
        return f"{value:>10.4f}"

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 60.0:
            return f"{seconds:>6.1f}s"
        minutes, remaining_seconds = divmod(seconds, 60.0)
        return f"{int(minutes):02d}m{int(remaining_seconds):02d}s"

    def _print_training_header(self) -> None:
        print(
            "[train] "
            f"{'Episode':>9} "
            f"{'Progress':>9} "
            f"{'EpReward':>12} "
            f"{'AvgStepR':>12} "
            f"{'ActorLoss':>10} "
            f"{'CriticLoss':>10} "
            f"{'Buffer':>8} "
            f"{'Active':>7} "
            f"{'Total':>6} "
            f"{'MaxTasks':>8} "
            f"{'GPUA/GiB':>9} "
            f"{'GPUR/GiB':>9} "
            f"{'EnvTime':>8} "
            f"{'UpdTime':>8} "
            f"{'EpTime':>8} "
            f"{'Elapsed':>8}",
            flush=True,
        )

    def _print_progress_row(
        self,
        *,
        episode: int,
        total_episodes: int,
        episode_shared_reward_sum: float,
        mean_shared_reward: float,
        last_actor_loss: float | None,
        last_critic_loss: float | None,
        episode_env_time_s: float,
        episode_update_time_s: float,
        episode_total_time_s: float,
        elapsed_s: float,
        max_tasks: int,
        gpu_allocated_gib: float,
        gpu_reserved_gib: float,
    ) -> None:
        progress = 100.0 * float(episode + 1) / float(total_episodes)
        print(
            "[train] "
            f"{episode + 1:>4}/{total_episodes:<4} "
            f"{progress:>7.1f}% "
            f"{episode_shared_reward_sum:>12.4f} "
            f"{mean_shared_reward:>12.4f} "
            f"{self._format_metric(last_actor_loss)} "
            f"{self._format_metric(last_critic_loss)} "
            f"{len(self.system.replay_buffer):>8} "
            f"{self.system.active_actor_count:>7} "
            f"{self.system.total_actor_count:>6} "
            f"{max_tasks:>8} "
            f"{gpu_allocated_gib:>9.3f} "
            f"{gpu_reserved_gib:>9.3f} "
            f"{self._format_seconds(episode_env_time_s):>8} "
            f"{self._format_seconds(episode_update_time_s):>8} "
            f"{self._format_seconds(episode_total_time_s):>8} "
            f"{self._format_seconds(elapsed_s):>8}",
            flush=True,
        )

    def train(self) -> MetricsLogger:
        """执行训练，并输出每轮进度。"""

        training_started_at = time.perf_counter()
        reward_config = self.env.reward_calculator.config
        training_time_slot = 0
        training_cumulative_system_profit = 0.0
        manager = self.env.base_env.clustering_manager
        if manager is not None:
            self.system.configure_agent_pool(manager.logical_agent_ids)
        if self.config.progress_print_interval > 0:
            self._print_training_header()

        for episode in range(self.config.num_episodes):
            episode_started_at = time.perf_counter()

            # 每个 episode 开始时重置环境：清空队列、重置时间、生成第一批任务。
            observations, action_specs = self.env.reset()
            self.system.set_active_agent_ids(self._active_logical_agent_ids())

            # 根据当前出现的 CH/孤立 UAV 创建或确认对应 agent。
            self._ensure_actor_pool(observations, action_specs)

            # OU 探索噪声按 episode 重置，避免上一轮噪声状态影响本轮探索。
            self.system.reset_noise()

            episode_shared_rewards: list[float] = []
            last_actor_loss: float | None = None
            last_critic_loss: float | None = None
            episode_env_time_s = 0.0
            episode_update_time_s = 0.0
            episode_transmission_energy_j = 0.0
            episode_computing_energy_j = 0.0
            episode_total_energy_j = 0.0
            episode_system_profit = 0.0
            episode_total_tasks_for_utility = 0
            episode_reliable_on_time_tasks = 0
            episode_completed_tasks = 0
            episode_timeout_or_drop_tasks = 0
            episode_deadline_failure_tasks = 0
            episode_capacity_drop_tasks = 0
            episode_redundancy_requested_tasks = 0
            episode_redundant_tasks = 0
            episode_admitted_redundant_tasks = 0
            episode_redundancy_success_tasks = 0
            episode_backup_success_tasks = 0
            episode_backup_selected_tasks = 0
            episode_reliability_failure_tasks = 0
            episode_max_tasks = 0
            battery_status = self.env.base_env.battery_status()

            for step in range(self.config.steps_per_episode):
                # 聚类结构可能变化，因此每个 step 都再次确认当前活跃 agent 已存在。
                for agent_id, observation in observations.items():
                    self.system.ensure_agent(
                        agent_id=agent_id,
                        state_dim=int(observation.shape[0]),
                        action_spec=action_specs[agent_id],
                    )

                # Actor 根据局部观测输出原始动作向量。
                raw_actions = self.system.act(observations)

                # 动作一份解码给环境执行，一份编码成 critic/replay buffer 使用的动作。
                env_actions, critic_actions = self.system.decode_actions(raw_actions)

                # 环境执行一步：处理当前 pending tasks，返回下一时刻观测和任务执行记录。
                env_started_at = time.perf_counter()
                next_observations, rewards, done, info = self.env.step(env_actions)
                self.system.set_active_agent_ids(self._active_logical_agent_ids())
                step_env_time_s = time.perf_counter() - env_started_at
                episode_env_time_s += step_env_time_s
                episode_shared_rewards.append(info["shared_reward"])
                next_action_specs = info["action_specs"]
                for agent_id, observation in next_observations.items():
                    self.system.ensure_agent(
                        agent_id=agent_id,
                        state_dim=int(observation.shape[0]),
                        action_spec=next_action_specs[agent_id],
                    )
                workflow_summary = info.get("workflow_summary", {})
                battery_status = info["battery_status"]

                # 当前 step 的任务执行记录，是完成率、超时率、冗余率等指标的来源。
                records = info["records"]
                total_tasks = len(records)
                episode_max_tasks = max(episode_max_tasks, total_tasks)
                replica_metrics = self.env.extract_record_metrics(records)
                # 能耗按所有实际执行任务累计，包含冗余副本和取消前的部分能耗。
                step_transmission_energy_j = float(
                    sum(record.transmission_energy_j for record in records)
                )
                step_computing_energy_j = float(
                    sum(record.computing_energy_j for record in records)
                )
                step_total_energy_j = float(sum(record.total_energy_j for record in records))
                step_system_profit = float(info["equation8_objective"].total_profit)
                training_time_slot += 1
                training_cumulative_system_profit += step_system_profit
                step_reliable_on_time_tasks = sum(
                    1
                    for record in records
                    if record.completed_before_deadline
                    and record.satisfies_reliability
                    and not record.failed_due_to_reliability
                )
                step_u_base = step_system_profit / reward_config.normalize_profit_scale
                step_u_net = step_u_base - (
                    reward_config.energy_penalty_weight
                    * step_total_energy_j
                    / reward_config.normalize_energy_j
                )
                episode_transmission_energy_j += step_transmission_energy_j
                episode_computing_energy_j += step_computing_energy_j
                episode_total_energy_j += step_total_energy_j
                episode_system_profit += step_system_profit
                episode_total_tasks_for_utility += total_tasks
                episode_reliable_on_time_tasks += step_reliable_on_time_tasks
                completed_tasks = sum(1 for record in records if record.completed_before_deadline)
                timeout_or_drop_tasks = total_tasks - completed_tasks
                self.logger.log(
                    record_type="battery_step",
                    episode=episode,
                    step=step,
                    time_slot=training_time_slot,
                    system_profit=step_system_profit,
                    cumulative_system_profit=training_cumulative_system_profit,
                    task_count=total_tasks,
                    completed_task_count=completed_tasks,
                    avg_requested_replica_count=replica_metrics.get(
                        "avg_requested_replica_count", 0.0
                    ),
                    replica_count_1_rate=replica_metrics.get("replica_count_1_rate", 0.0),
                    replica_count_2_rate=replica_metrics.get("replica_count_2_rate", 0.0),
                    replica_count_3_rate=replica_metrics.get("replica_count_3_rate", 0.0),
                    avg_admitted_replica_count=replica_metrics.get(
                        "avg_admitted_replica_count", 0.0
                    ),
                    capacity_rejected_replica_rate=replica_metrics.get(
                        "capacity_rejected_replica_rate", 0.0
                    ),
                    uav_replica_share=replica_metrics.get("uav_replica_share", 0.0),
                    bs_replica_share=replica_metrics.get("bs_replica_share", 0.0),
                    leo_replica_share=replica_metrics.get("leo_replica_share", 0.0),
                    same_layer_replica_rate=replica_metrics.get(
                        "same_layer_replica_rate", 0.0
                    ),
                    cross_layer_replica_rate=replica_metrics.get(
                        "cross_layer_replica_rate", 0.0
                    ),
                    reliable_on_time_completion_rate=replica_metrics.get(
                        "reliable_on_time_completion_rate", 0.0
                    ),
                    energy_per_completed_task=replica_metrics.get(
                        "energy_per_completed_task", 0.0
                    ),
                    cancellation_energy_saved_j=replica_metrics.get(
                        "cancellation_energy_saved_j", 0.0
                    ),
                    avg_combined_reliability=replica_metrics.get(
                        "avg_combined_reliability", 0.0
                    ),
                    energy_constraint_multiplier=info.get(
                        "energy_constraint_multiplier", 0.0
                    ),
                    energy_budget_violation_j=info.get(
                        "energy_budget_violation_j", 0.0
                    ),
                    battery_status=battery_status,
                )
                uav_arrival_tasks = sum(1 for record in records if record.target_node_type == "uav")
                bs_arrival_tasks = sum(1 for record in records if record.target_node_type == "bs")
                leo_arrival_tasks = sum(1 for record in records if record.target_node_type == "leo")
                deadline_failure_tasks = sum(
                    1
                    for record in records
                    if record.constraint_check is not None
                    and not record.constraint_check.satisfies_deadline
                )
                capacity_drop_tasks = sum(
                    record.capacity_rejected_replica_count
                    if record.capacity_rejected_replica_count > 0
                    else int(
                        record.constraint_check is not None
                        and not record.constraint_check.satisfies_capacity
                    )
                    for record in records
                )
                redundancy_requested_tasks = sum(
                    1 for record in records if record.redundancy_requested
                )
                admitted_redundant_tasks = sum(
                    1 for record in records if record.is_redundant_task
                )
                redundant_tasks = admitted_redundant_tasks
                redundancy_success_tasks = sum(
                    1 for record in records if record.redundancy_succeeded
                )
                backup_success_tasks = sum(
                    1 for record in records if record.backup_succeeded
                )
                backup_selected_tasks = sum(
                    1
                    for record in records
                    if record.is_redundant_task and record.selected_replica_role == "backup"
                )
                reliability_failure_tasks = sum(
                    1
                    for record in records
                    if record.failed_due_to_reliability
                    or not record.satisfies_reliability
                )
                episode_completed_tasks += completed_tasks
                episode_timeout_or_drop_tasks += timeout_or_drop_tasks
                episode_deadline_failure_tasks += deadline_failure_tasks
                episode_capacity_drop_tasks += capacity_drop_tasks
                episode_redundancy_requested_tasks += redundancy_requested_tasks
                episode_redundant_tasks += redundant_tasks
                episode_admitted_redundant_tasks += admitted_redundant_tasks
                episode_redundancy_success_tasks += redundancy_success_tasks
                episode_backup_success_tasks += backup_success_tasks
                episode_backup_selected_tasks += backup_selected_tasks
                episode_reliability_failure_tasks += reliability_failure_tasks
                avg_end_to_end_reliability = (
                    float(np.mean([record.end_to_end_reliability for record in records]))
                    if total_tasks > 0
                    else 0.0
                )
                avg_completion_delay_s = (
                    float(
                        np.mean(
                            [
                                record.actual_finish_delay_s
                                for record in records
                                if record.completed_before_deadline
                            ]
                        )
                    )
                    if completed_tasks > 0
                    else 0.0
                )
                avg_actual_finish_delay_all_tasks_s = (
                    float(np.mean([record.actual_finish_delay_s for record in records]))
                    if total_tasks > 0
                    else 0.0
                )
                avg_actual_finish_delay_failed_tasks_s = (
                    float(
                        np.mean(
                            [
                                record.actual_finish_delay_s
                                for record in records
                                if not record.completed_before_deadline
                            ]
                        )
                    )
                    if timeout_or_drop_tasks > 0
                    else 0.0
                )
                completion_rate = (
                    float(completed_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                timeout_or_drop_rate = (
                    float(timeout_or_drop_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                deadline_failure_rate = (
                    float(deadline_failure_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                capacity_drop_rate = (
                    float(capacity_drop_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                redundancy_rate = (
                    float(redundant_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                redundancy_request_rate = (
                    float(redundancy_requested_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                redundancy_success_rate = (
                    float(redundancy_success_tasks)
                    / float(admitted_redundant_tasks)
                    if admitted_redundant_tasks > 0
                    else 0.0
                )
                backup_admission_rate = (
                    float(admitted_redundant_tasks) / float(redundancy_requested_tasks)
                    if redundancy_requested_tasks > 0
                    else 0.0
                )
                backup_success_rate = (
                    float(backup_success_tasks) / float(redundancy_requested_tasks)
                    if redundancy_requested_tasks > 0
                    else 0.0
                )
                backup_selection_rate = (
                    float(backup_selected_tasks) / float(redundant_tasks)
                    if redundant_tasks > 0
                    else 0.0
                )
                reliability_failure_rate = (
                    float(reliability_failure_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                uav_arrival_rate = (
                    float(uav_arrival_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                bs_arrival_rate = (
                    float(bs_arrival_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )
                leo_arrival_rate = (
                    float(leo_arrival_tasks) / float(total_tasks)
                    if total_tasks > 0
                    else 0.0
                )

                # 分层统计 UAV/BS/LEO 上的 deadline、capacity、reliability 和 delay 表现。
                layer_stats = {}
                for layer_name in ("uav", "bs", "leo"):
                    layer_records = [
                        record for record in records if record.target_node_type == layer_name
                    ]
                    layer_total = len(layer_records)
                    layer_deadline_failures = sum(
                        1
                        for record in layer_records
                        if record.constraint_check is not None
                        and not record.constraint_check.satisfies_deadline
                    )
                    layer_capacity_drops = sum(
                        1
                        for record in layer_records
                        if record.constraint_check is not None
                        and not record.constraint_check.satisfies_capacity
                    )
                    layer_reliability_failures = sum(
                        1
                        for record in layer_records
                        if record.failed_due_to_reliability
                        or not record.satisfies_reliability
                    )
                    layer_stats[layer_name] = {
                        "deadline_failure_rate": (
                            float(layer_deadline_failures) / float(layer_total)
                            if layer_total > 0
                            else 0.0
                        ),
                        "capacity_drop_rate": (
                            float(layer_capacity_drops) / float(layer_total)
                            if layer_total > 0
                            else 0.0
                        ),
                        "reliability_failure_rate": (
                            float(layer_reliability_failures) / float(layer_total)
                            if layer_total > 0
                            else 0.0
                        ),
                        "avg_reliability": (
                            float(np.mean([record.end_to_end_reliability for record in layer_records]))
                            if layer_total > 0
                            else 0.0
                        ),
                        "avg_delay_s": (
                            float(np.mean([record.actual_finish_delay_s for record in layer_records]))
                            if layer_total > 0
                            else 0.0
                        ),
                    }

                # 将多智能体联合经验写入 replay buffer，供后续 critic/actor 更新。
                self.system.store_transitions(
                    observations=observations,
                    critic_actions=critic_actions,
                    shared_reward=info["shared_reward"],
                    next_observations=next_observations,
                    done=done,
                )

                # 每隔 update_every 个 step 从 replay buffer 采样，并更新 Actor/Critic。
                if step % self.config.update_every == 0:
                    update_started_at = time.perf_counter()
                    update_result = self.system.update(batch_size=self.config.batch_size)
                    step_update_time_s = time.perf_counter() - update_started_at
                    episode_update_time_s += step_update_time_s
                    if update_result is not None:
                        last_actor_loss = update_result.actor_loss
                        last_critic_loss = update_result.critic_loss

                        # 只有网络实际更新后才记录 step-level 指标，避免 warm-up 阶段日志过密。
                        self.logger.log(
                            episode=episode,
                            step=step,
                            actor_loss=update_result.actor_loss,
                            critic_loss=update_result.critic_loss,
                            shared_reward=info["shared_reward"],
                            system_profit=step_system_profit,
                            u_base=step_u_base,
                            u_net=step_u_net,
                            reliable_on_time_tasks=step_reliable_on_time_tasks,
                            reliable_on_time_completion_rate=(
                                float(step_reliable_on_time_tasks) / float(total_tasks)
                                if total_tasks > 0
                                else 0.0
                            ),
                            system_transmission_energy_j=step_transmission_energy_j,
                            system_computing_energy_j=step_computing_energy_j,
                            system_total_energy_j=step_total_energy_j,
                            avg_requested_replica_count=replica_metrics.get(
                                "avg_requested_replica_count", 0.0
                            ),
                            replica_count_1_rate=replica_metrics.get("replica_count_1_rate", 0.0),
                            replica_count_2_rate=replica_metrics.get("replica_count_2_rate", 0.0),
                            replica_count_3_rate=replica_metrics.get("replica_count_3_rate", 0.0),
                            avg_admitted_replica_count=replica_metrics.get(
                                "avg_admitted_replica_count", 0.0
                            ),
                            capacity_rejected_replica_rate=replica_metrics.get(
                                "capacity_rejected_replica_rate", 0.0
                            ),
                            uav_replica_share=replica_metrics.get("uav_replica_share", 0.0),
                            bs_replica_share=replica_metrics.get("bs_replica_share", 0.0),
                            leo_replica_share=replica_metrics.get("leo_replica_share", 0.0),
                            same_layer_replica_rate=replica_metrics.get(
                                "same_layer_replica_rate", 0.0
                            ),
                            cross_layer_replica_rate=replica_metrics.get(
                                "cross_layer_replica_rate", 0.0
                            ),
                            energy_per_completed_task=replica_metrics.get(
                                "energy_per_completed_task", 0.0
                            ),
                            cancellation_energy_saved_j=replica_metrics.get(
                                "cancellation_energy_saved_j", 0.0
                            ),
                            avg_combined_reliability=replica_metrics.get(
                                "avg_combined_reliability", 0.0
                            ),
                            energy_constraint_multiplier=info.get(
                                "energy_constraint_multiplier", 0.0
                            ),
                            energy_budget_violation_j=info.get(
                                "energy_budget_violation_j", 0.0
                            ),
                            task_completion_rate=completion_rate,
                            task_timeout_or_drop_rate=timeout_or_drop_rate,
                            task_deadline_failure_rate=deadline_failure_rate,
                            task_capacity_drop_rate=capacity_drop_rate,
                            avg_task_completion_delay_s=avg_completion_delay_s,
                            avg_actual_finish_delay_all_tasks_s=avg_actual_finish_delay_all_tasks_s,
                            avg_actual_finish_delay_failed_tasks_s=avg_actual_finish_delay_failed_tasks_s,
                            total_tasks=total_tasks,
                            completed_tasks=completed_tasks,
                            timeout_or_drop_tasks=timeout_or_drop_tasks,
                            deadline_failure_tasks=deadline_failure_tasks,
                            capacity_drop_tasks=capacity_drop_tasks,
                            redundancy_requested_tasks=redundancy_requested_tasks,
                            redundant_tasks=redundant_tasks,
                            admitted_redundant_tasks=admitted_redundant_tasks,
                            redundancy_success_tasks=redundancy_success_tasks,
                            backup_success_tasks=backup_success_tasks,
                            backup_selected_tasks=backup_selected_tasks,
                            reliability_failure_tasks=reliability_failure_tasks,
                            uav_arrival_tasks=uav_arrival_tasks,
                            bs_arrival_tasks=bs_arrival_tasks,
                            leo_arrival_tasks=leo_arrival_tasks,
                            uav_arrival_rate=uav_arrival_rate,
                            bs_arrival_rate=bs_arrival_rate,
                            leo_arrival_rate=leo_arrival_rate,
                            redundancy_request_rate=redundancy_request_rate,
                            redundancy_rate=redundancy_rate,
                            backup_admission_rate=backup_admission_rate,
                            redundancy_success_rate=redundancy_success_rate,
                            backup_success_rate=backup_success_rate,
                            backup_selection_rate=backup_selection_rate,
                            avg_end_to_end_reliability=avg_end_to_end_reliability,
                            reliability_failure_rate=reliability_failure_rate,
                            uav_deadline_failure_rate=layer_stats["uav"]["deadline_failure_rate"],
                            bs_deadline_failure_rate=layer_stats["bs"]["deadline_failure_rate"],
                            leo_deadline_failure_rate=layer_stats["leo"]["deadline_failure_rate"],
                            uav_capacity_drop_rate=layer_stats["uav"]["capacity_drop_rate"],
                            bs_capacity_drop_rate=layer_stats["bs"]["capacity_drop_rate"],
                            leo_capacity_drop_rate=layer_stats["leo"]["capacity_drop_rate"],
                            uav_reliability_failure_rate=layer_stats["uav"]["reliability_failure_rate"],
                            bs_reliability_failure_rate=layer_stats["bs"]["reliability_failure_rate"],
                            leo_reliability_failure_rate=layer_stats["leo"]["reliability_failure_rate"],
                            uav_avg_reliability=layer_stats["uav"]["avg_reliability"],
                            bs_avg_reliability=layer_stats["bs"]["avg_reliability"],
                            leo_avg_reliability=layer_stats["leo"]["avg_reliability"],
                            uav_avg_delay_s=layer_stats["uav"]["avg_delay_s"],
                            bs_avg_delay_s=layer_stats["bs"]["avg_delay_s"],
                            leo_avg_delay_s=layer_stats["leo"]["avg_delay_s"],
                            active_workflows=workflow_summary.get("active_workflows", 0),
                            ready_workflow_tasks=workflow_summary.get("ready_tasks", 0),
                            completed_workflows=workflow_summary.get("completed_workflows", 0),
                            failed_workflows=workflow_summary.get("failed_workflows", 0),
                            workflow_sla_violations=workflow_summary.get("workflow_sla_violations", 0),
                            pending_workflow_task_completions=workflow_summary.get("pending_task_completions", 0),
                            avg_completed_workflow_makespan_s=workflow_summary.get(
                                "avg_completed_workflow_makespan_s",
                                0.0,
                            ),
                            max_completed_workflow_makespan_s=workflow_summary.get(
                                "max_completed_workflow_makespan_s",
                                0.0,
                            ),
                            sum_completed_workflow_makespan_s=workflow_summary.get(
                                "sum_completed_workflow_makespan_s",
                                0.0,
                            ),
                            step_env_time_s=step_env_time_s,
                            step_update_time_s=step_update_time_s,
                        )

                # 切换到下一时刻观测，继续本 episode 的下一个 step。
                observations = next_observations
                action_specs = next_action_specs

            # 汇总本 episode 内所有 step-level 指标，形成 episode-level 曲线数据。
            episode_shared_reward_sum = sum(episode_shared_rewards)
            episode_u_base = (
                episode_system_profit / reward_config.normalize_profit_scale
            )
            episode_u_net = episode_u_base - (
                reward_config.energy_penalty_weight
                * episode_total_energy_j
                / reward_config.normalize_energy_j
            )
            episode_reliable_on_time_completion_rate = (
                float(episode_reliable_on_time_tasks)
                / float(episode_total_tasks_for_utility)
                if episode_total_tasks_for_utility > 0
                else 0.0
            )
            episode_total_time_s = time.perf_counter() - episode_started_at
            gpu_allocated_gib, gpu_reserved_gib = self._gpu_memory_metrics()
            episode_step_records = [
                record
                for record in self.logger.records
                if record.get("episode") == episode and "task_completion_rate" in record
            ]
            episode_total_tasks = episode_total_tasks_for_utility
            episode_completion_rate = (
                float(episode_completed_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_timeout_or_drop_rate = (
                float(episode_timeout_or_drop_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_deadline_failure_rate = (
                float(episode_deadline_failure_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_capacity_drop_rate = (
                float(episode_capacity_drop_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_uav_arrival_rate = (
                float(np.mean([record["uav_arrival_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_bs_arrival_rate = (
                float(np.mean([record["bs_arrival_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_leo_arrival_rate = (
                float(np.mean([record["leo_arrival_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_uav_arrival_tasks = (
                int(sum(record["uav_arrival_tasks"] for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_bs_arrival_tasks = (
                int(sum(record["bs_arrival_tasks"] for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_leo_arrival_tasks = (
                int(sum(record["leo_arrival_tasks"] for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_redundancy_rate = (
                float(episode_redundant_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_redundancy_request_rate = (
                float(episode_redundancy_requested_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_redundancy_success_rate = (
                float(episode_redundancy_success_tasks)
                / float(episode_admitted_redundant_tasks)
                if episode_admitted_redundant_tasks > 0
                else 0.0
            )
            episode_backup_admission_rate = (
                float(episode_admitted_redundant_tasks)
                / float(episode_redundancy_requested_tasks)
                if episode_redundancy_requested_tasks > 0
                else 0.0
            )
            episode_backup_success_rate = (
                float(episode_backup_success_tasks)
                / float(episode_redundancy_requested_tasks)
                if episode_redundancy_requested_tasks > 0
                else 0.0
            )
            episode_backup_selection_rate = (
                float(episode_backup_selected_tasks) / float(episode_redundant_tasks)
                if episode_redundant_tasks > 0
                else 0.0
            )
            episode_avg_end_to_end_reliability = (
                float(np.mean([record["avg_end_to_end_reliability"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_reliability_failure_rate = (
                float(episode_reliability_failure_tasks) / float(episode_total_tasks)
                if episode_total_tasks > 0
                else 0.0
            )
            episode_avg_completion_delay_s = (
                float(np.mean([record["avg_task_completion_delay_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_avg_actual_finish_delay_all_tasks_s = (
                float(np.mean([record["avg_actual_finish_delay_all_tasks_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_avg_actual_finish_delay_failed_tasks_s = (
                float(np.mean([record["avg_actual_finish_delay_failed_tasks_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_uav_deadline_failure_rate = (
                float(np.mean([record["uav_deadline_failure_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_bs_deadline_failure_rate = (
                float(np.mean([record["bs_deadline_failure_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_leo_deadline_failure_rate = (
                float(np.mean([record["leo_deadline_failure_rate"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_uav_avg_delay_s = (
                float(np.mean([record["uav_avg_delay_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_bs_avg_delay_s = (
                float(np.mean([record["bs_avg_delay_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_leo_avg_delay_s = (
                float(np.mean([record["leo_avg_delay_s"] for record in episode_step_records]))
                if episode_step_records
                else 0.0
            )
            episode_completed_workflows = (
                int(sum(record.get("completed_workflows", 0) for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_failed_workflows = (
                int(sum(record.get("failed_workflows", 0) for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_workflow_sla_violations = (
                int(sum(record.get("workflow_sla_violations", 0) for record in episode_step_records))
                if episode_step_records
                else 0
            )
            episode_sum_completed_workflow_makespan_s = (
                float(
                    sum(
                        record.get("sum_completed_workflow_makespan_s", 0.0)
                        for record in episode_step_records
                    )
                )
                if episode_step_records
                else 0.0
            )
            episode_avg_completed_workflow_makespan_s = (
                episode_sum_completed_workflow_makespan_s / float(episode_completed_workflows)
                if episode_completed_workflows > 0
                else 0.0
            )
            episode_max_completed_workflow_makespan_s = (
                float(
                    max(
                        record.get("max_completed_workflow_makespan_s", 0.0)
                        for record in episode_step_records
                    )
                )
                if episode_step_records
                else 0.0
            )
            episode_workflow_sla_violation_rate = (
                float(episode_workflow_sla_violations) / float(episode_completed_workflows)
                if episode_completed_workflows > 0
                else 0.0
            )
            self.logger.log(
                record_type="episode",
                episode=episode,
                episode_battery_status=battery_status,
                episode_shared_reward=episode_shared_reward_sum,
                episode_system_profit=episode_system_profit,
                episode_u_base=episode_u_base,
                episode_u_net=episode_u_net,
                episode_total_tasks_for_utility=episode_total_tasks_for_utility,
                episode_reliable_on_time_tasks=episode_reliable_on_time_tasks,
                episode_reliable_on_time_completion_rate=(
                    episode_reliable_on_time_completion_rate
                ),
                episode_transmission_energy_j=episode_transmission_energy_j,
                episode_computing_energy_j=episode_computing_energy_j,
                episode_total_energy_j=episode_total_energy_j,
                episode_total_tasks=episode_total_tasks,
                episode_completed_tasks=episode_completed_tasks,
                episode_timeout_or_drop_tasks=episode_timeout_or_drop_tasks,
                episode_deadline_failure_tasks=episode_deadline_failure_tasks,
                episode_capacity_drop_tasks=episode_capacity_drop_tasks,
                episode_task_completion_rate=episode_completion_rate,
                episode_task_timeout_or_drop_rate=episode_timeout_or_drop_rate,
                episode_task_deadline_failure_rate=episode_deadline_failure_rate,
                episode_task_capacity_drop_rate=episode_capacity_drop_rate,
                episode_uav_arrival_rate=episode_uav_arrival_rate,
                episode_bs_arrival_rate=episode_bs_arrival_rate,
                episode_leo_arrival_rate=episode_leo_arrival_rate,
                episode_uav_arrival_tasks=episode_uav_arrival_tasks,
                episode_bs_arrival_tasks=episode_bs_arrival_tasks,
                episode_leo_arrival_tasks=episode_leo_arrival_tasks,
                episode_redundancy_requested_tasks=(
                    episode_redundancy_requested_tasks
                ),
                episode_redundant_tasks=episode_redundant_tasks,
                episode_admitted_redundant_tasks=episode_admitted_redundant_tasks,
                episode_redundancy_success_tasks=episode_redundancy_success_tasks,
                episode_backup_success_tasks=episode_backup_success_tasks,
                episode_backup_selected_tasks=episode_backup_selected_tasks,
                episode_reliability_failure_tasks=episode_reliability_failure_tasks,
                episode_redundancy_request_rate=episode_redundancy_request_rate,
                episode_redundancy_rate=episode_redundancy_rate,
                episode_backup_admission_rate=episode_backup_admission_rate,
                episode_redundancy_success_rate=episode_redundancy_success_rate,
                episode_backup_success_rate=episode_backup_success_rate,
                episode_backup_selection_rate=episode_backup_selection_rate,
                episode_avg_end_to_end_reliability=episode_avg_end_to_end_reliability,
                episode_reliability_failure_rate=episode_reliability_failure_rate,
                episode_avg_task_completion_delay_s=episode_avg_completion_delay_s,
                episode_avg_actual_finish_delay_all_tasks_s=episode_avg_actual_finish_delay_all_tasks_s,
                episode_avg_actual_finish_delay_failed_tasks_s=episode_avg_actual_finish_delay_failed_tasks_s,
                episode_uav_deadline_failure_rate=episode_uav_deadline_failure_rate,
                episode_bs_deadline_failure_rate=episode_bs_deadline_failure_rate,
                episode_leo_deadline_failure_rate=episode_leo_deadline_failure_rate,
                episode_uav_avg_delay_s=episode_uav_avg_delay_s,
                episode_bs_avg_delay_s=episode_bs_avg_delay_s,
                episode_leo_avg_delay_s=episode_leo_avg_delay_s,
                episode_completed_workflows=episode_completed_workflows,
                episode_failed_workflows=episode_failed_workflows,
                episode_workflow_sla_violations=episode_workflow_sla_violations,
                episode_workflow_sla_violation_rate=episode_workflow_sla_violation_rate,
                episode_avg_completed_workflow_makespan_s=episode_avg_completed_workflow_makespan_s,
                episode_max_completed_workflow_makespan_s=episode_max_completed_workflow_makespan_s,
                episode_sum_completed_workflow_makespan_s=episode_sum_completed_workflow_makespan_s,
                episode_env_time_s=episode_env_time_s,
                episode_update_time_s=episode_update_time_s,
                episode_total_time_s=episode_total_time_s,
                active_actor_count=self.system.active_actor_count,
                total_actor_count=self.system.total_actor_count,
                max_tasks=episode_max_tasks,
                gpu_allocated_gib=gpu_allocated_gib,
                gpu_reserved_gib=gpu_reserved_gib,
            )

            # 按设定间隔向终端打印训练进度，方便长实验观察是否正常推进。
            if (
                self.config.progress_print_interval > 0
                and (episode + 1) % self.config.progress_print_interval == 0
            ):
                mean_shared_reward = (
                    episode_shared_reward_sum / len(episode_shared_rewards)
                    if episode_shared_rewards
                    else 0.0
                )
                self._print_progress_row(
                    episode=episode,
                    total_episodes=self.config.num_episodes,
                    episode_shared_reward_sum=episode_shared_reward_sum,
                    mean_shared_reward=mean_shared_reward,
                    last_actor_loss=last_actor_loss,
                    last_critic_loss=last_critic_loss,
                    episode_env_time_s=episode_env_time_s,
                    episode_update_time_s=episode_update_time_s,
                    episode_total_time_s=episode_total_time_s,
                    elapsed_s=time.perf_counter() - training_started_at,
                    max_tasks=episode_max_tasks,
                    gpu_allocated_gib=gpu_allocated_gib,
                    gpu_reserved_gib=gpu_reserved_gib,
                )

            self._periodic_checkpoint(episode + 1)

        if self.config.progress_print_interval > 0:
            print(
                f"[train] completed {self.config.num_episodes} episodes in "
                f"{self._format_seconds(time.perf_counter() - training_started_at).strip()}",
                flush=True,
            )

        return self.logger
