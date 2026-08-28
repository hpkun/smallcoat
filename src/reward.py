from __future__ import annotations

from dataclasses import dataclass

from .entities import ExecutionRecord
from .objective import compute_equation_8_objective

DEFAULT_PROFIT_SCALE = 1_000_000_000.0
DEFAULT_ENERGY_SCALE_J = 1_000.0


@dataclass(frozen=True)
class RewardBreakdown:
    """论文公式 (19) 对应的共享奖励分解。"""

    total_reward: float
    system_profit: float
    energy_penalty: float
    completion_constraint_penalty: float = 0.0
    energy_budget_constraint_penalty: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    """
    共享奖励配置。

    论文公式 (19) 直接使用系统总收益作为共享奖励，因此这里只保留一个
    可选归一化系数，避免训练时数值过大。
    """

    normalize_profit_scale: float = DEFAULT_PROFIT_SCALE
    reward_clip_abs: float | None = None
    deadline_failure_penalty: float = 0.005
    capacity_drop_penalty: float = 0.005
    reliability_failure_penalty: float = 0.005
    completion_delay_penalty: float = 0.001
    # 在原有目标和约束惩罚不变的基础上，增加最小化任务能耗目标。
    energy_penalty_weight: float = 0.01
    normalize_energy_j: float = DEFAULT_ENERGY_SCALE_J
    minimum_long_term_completion_rate: float = 0.90
    completion_rate_ema_alpha: float = 0.05
    completion_constraint_dual_lr: float = 0.01
    completion_constraint_dual_max: float = 1.0
    long_term_energy_budget_j_per_step: float | None = 1_000.0
    energy_budget_ema_alpha: float = 0.05
    energy_constraint_dual_lr: float = 0.01
    energy_constraint_dual_max: float = 1.0
    advantage_reward_weight: float = 0.5
    advantage_reward_clip: float | None = 0.05
    profit_baseline_ema_alpha: float = 0.05


class SharedRewardCalculator:
    """
    论文公式 (19) 的共享奖励实现：

        r(s, a) = max_{alpha_k,j} sum_j sum_k alpha_{k,j} G_k

    在环境执行后，当前一批 records 的 realized_profit 总和就是本时隙下
    已经落实的系统总收益，因此所有 agent 共享同一个 reward。
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._profit_reward_baseline: float | None = None
        self._completion_rate_ema: float | None = None
        self._completion_constraint_multiplier = 0.0
        self._energy_ema_j: float | None = None
        self._energy_constraint_multiplier = 0.0

    @property
    def energy_constraint_multiplier(self) -> float:
        return float(self._energy_constraint_multiplier)

    @property
    def energy_budget_violation_j(self) -> float:
        budget = self.config.long_term_energy_budget_j_per_step
        if budget is None or self._energy_ema_j is None:
            return 0.0
        return float(max(0.0, self._energy_ema_j - budget))

    def _normalize_reward(self, profit: float) -> float:
        scale = self.config.normalize_profit_scale
        if scale <= 0:
            raise ValueError("normalize_profit_scale must be positive.")

        reward = float(profit) / scale
        clip_abs = self.config.reward_clip_abs
        if clip_abs is not None:
            if clip_abs <= 0:
                raise ValueError("reward_clip_abs must be positive when provided.")
            reward = max(-clip_abs, min(clip_abs, reward))
        return float(reward)

    def _profit_advantage_bonus(self, profit_reward: float) -> float:
        alpha = self.config.profit_baseline_ema_alpha
        if not 0.0 < alpha <= 1.0:
            raise ValueError("profit_baseline_ema_alpha must be in (0, 1].")

        baseline = self._profit_reward_baseline
        self._profit_reward_baseline = (
            profit_reward
            if baseline is None
            else (1.0 - alpha) * baseline + alpha * profit_reward
        )
        if baseline is None:
            return 0.0

        bonus = self.config.advantage_reward_weight * max(0.0, profit_reward - baseline)
        clip_abs = self.config.advantage_reward_clip
        if clip_abs is not None:
            if clip_abs <= 0:
                raise ValueError("advantage_reward_clip must be positive when provided.")
            bonus = min(bonus, clip_abs)
        return float(bonus)

    def _energy_penalty(self, average_energy_j: float) -> float:
        """将系统任务能耗归一化后转换为长期净收益中的能耗成本。"""

        if self.config.energy_penalty_weight < 0:
            raise ValueError("energy_penalty_weight must be non-negative.")
        if self.config.normalize_energy_j <= 0:
            raise ValueError("normalize_energy_j must be positive.")
        if average_energy_j < 0:
            raise ValueError("average_energy_j must be non-negative.")
        return float(
            self.config.energy_penalty_weight
            * average_energy_j
            / self.config.normalize_energy_j
        )

    def _completion_constraint_penalty(self, completion_rate: float) -> float:
        """用自适应拉格朗日乘子维护最低长期完成率，不设置固定完成率奖励。"""

        target = self.config.minimum_long_term_completion_rate
        alpha = self.config.completion_rate_ema_alpha
        dual_lr = self.config.completion_constraint_dual_lr
        dual_max = self.config.completion_constraint_dual_max
        if not 0.0 <= target <= 1.0:
            raise ValueError("minimum_long_term_completion_rate must be in [0, 1].")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("completion_rate_ema_alpha must be in (0, 1].")
        if dual_lr < 0.0 or dual_max < 0.0:
            raise ValueError("Completion constraint dual parameters must be non-negative.")

        previous = self._completion_rate_ema
        self._completion_rate_ema = (
            completion_rate
            if previous is None
            else (1.0 - alpha) * previous + alpha * completion_rate
        )
        violation = target - self._completion_rate_ema
        self._completion_constraint_multiplier = min(
            dual_max,
            max(0.0, self._completion_constraint_multiplier + dual_lr * violation),
        )
        return float(self._completion_constraint_multiplier * max(0.0, violation))

    def _energy_budget_constraint_penalty(self, energy_j: float) -> float:
        """Update the long-term energy dual and return lambda_E * g_E."""

        budget = self.config.long_term_energy_budget_j_per_step
        if budget is None:
            return 0.0
        alpha = self.config.energy_budget_ema_alpha
        dual_lr = self.config.energy_constraint_dual_lr
        dual_max = self.config.energy_constraint_dual_max
        if budget < 0.0:
            raise ValueError("long_term_energy_budget_j_per_step must be non-negative.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("energy_budget_ema_alpha must be in (0, 1].")
        if dual_lr < 0.0 or dual_max < 0.0:
            raise ValueError("Energy constraint dual parameters must be non-negative.")
        previous = self._energy_ema_j
        self._energy_ema_j = (
            energy_j
            if previous is None
            else (1.0 - alpha) * previous + alpha * energy_j
        )
        normalized_violation = (
            self._energy_ema_j - budget
        ) / self.config.normalize_energy_j
        self._energy_constraint_multiplier = min(
            dual_max,
            max(0.0, self._energy_constraint_multiplier + dual_lr * normalized_violation),
        )
        return float(
            self._energy_constraint_multiplier * max(0.0, normalized_violation)
        )

    def compute(self, record: ExecutionRecord) -> RewardBreakdown:
        """为单条执行记录返回其对应收益项。"""

        system_profit = record.realized_profit
        energy_penalty = self._energy_penalty(record.total_energy_j)
        total_reward = self._normalize_reward(system_profit) - energy_penalty
        return RewardBreakdown(
            total_reward=float(total_reward),
            system_profit=float(system_profit),
            energy_penalty=energy_penalty,
        )

    def aggregate(self, records: list[ExecutionRecord]) -> float:
        """对一个时隙内的全部记录计算论文式 (19) 的共享奖励。"""

        objective = compute_equation_8_objective(records)
        profit_reward = self._normalize_reward(objective.total_profit)
        if not records:
            return profit_reward - self._energy_budget_constraint_penalty(0.0)
        reward = profit_reward + self._profit_advantage_bonus(profit_reward)

        deadline_failure_rate = sum(
            1.0
            for record in records
            if record.constraint_check is not None
            and not record.constraint_check.satisfies_deadline
        ) / len(records)
        requested_replica_count = sum(
            max(record.requested_replica_count, 1) for record in records
        )
        capacity_drop_rate = sum(
            record.capacity_rejected_replica_count
            if record.capacity_rejected_replica_count > 0
            else float(
                record.constraint_check is not None
                and not record.constraint_check.satisfies_capacity
            )
            for record in records
        ) / requested_replica_count
        reliability_failure_rate = sum(
            1.0
            for record in records
            if record.failed_due_to_reliability
            or not record.satisfies_reliability
        ) / len(records)
        completed_delays = [
            record.actual_finish_delay_s
            for record in records
            if record.completed_before_deadline
        ]
        avg_completion_delay_s = (
            sum(completed_delays) / len(completed_delays)
            if completed_delays
            else 0.0
        )
        reward -= self.config.deadline_failure_penalty * deadline_failure_rate
        reward -= self.config.capacity_drop_penalty * capacity_drop_rate
        reward -= self.config.reliability_failure_penalty * reliability_failure_rate
        reward -= self.config.completion_delay_penalty * avg_completion_delay_s
        # 成功任务总收益与系统总能耗成本逐时隙对应，形成可靠净收益。
        reward -= self._energy_penalty(objective.total_energy_j)
        reward -= self._energy_budget_constraint_penalty(objective.total_energy_j)
        # 完成率通过长期约束的对偶变量处理，不再作为固定权重奖励项重复计算。
        reward -= self._completion_constraint_penalty(objective.completion_rate)

        clip_abs = self.config.reward_clip_abs
        if clip_abs is not None:
            reward = max(-clip_abs, min(clip_abs, reward))
        return float(reward)

    def aggregate_equation_8_profit(self, records: list[ExecutionRecord]) -> float:
        """提取一批执行记录中的系统总收益。"""

        return compute_equation_8_objective(records).total_profit
