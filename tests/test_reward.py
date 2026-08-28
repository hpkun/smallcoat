from __future__ import annotations

import math

from src.constraints import ConstraintCheckResult
from src.entities import ExecutionRecord
from src.reward import RewardConfig
from src.reward import SharedRewardCalculator


def _record(
    *,
    profit: float,
    completed: bool = True,
    satisfies_deadline: bool = True,
    satisfies_capacity: bool = True,
    finish_delay_s: float = 1.0,
    total_energy_j: float = 0.0,
    requested_replica_count: int = 1,
    capacity_rejected_replica_count: int = 0,
) -> ExecutionRecord:
    return ExecutionRecord(
        task_id="task-0",
        ingress_uav_id="uav-0",
        decision_uav_id="uav-0",
        target_node_id="bs-0",
        target_node_type="bs",
        compute_priority_eta=1.0,
        created_at_s=0.0,
        arrival_at_uav_s=0.0,
        arrival_at_target_s=0.0,
        start_compute_s=0.0,
        finish_time_s=finish_delay_s,
        ingress_transmission_delay_s=0.0,
        ingress_propagation_delay_s=0.0,
        backhaul_transmission_delay_s=0.0,
        backhaul_propagation_delay_s=0.0,
        queue_delay_s=0.0,
        compute_delay_s=finish_delay_s,
        communication_delay_s=0.0,
        total_delay_s=finish_delay_s,
        actual_finish_delay_s=finish_delay_s,
        completed_before_deadline=completed,
        realized_profit=profit,
        total_energy_j=total_energy_j,
        requested_replica_count=requested_replica_count,
        capacity_rejected_replica_count=capacity_rejected_replica_count,
        constraint_check=ConstraintCheckResult(
            satisfies_unique_offload=True,
            satisfies_deadline=satisfies_deadline,
            satisfies_binary_action=True,
            satisfies_capacity=satisfies_capacity,
        ),
    )


def test_default_reward_penalties_are_conservative_not_zero() -> None:
    config = RewardConfig()

    assert config.deadline_failure_penalty == 0.005
    assert config.capacity_drop_penalty == 0.005
    assert config.reliability_failure_penalty == 0.005
    assert config.completion_delay_penalty == 0.001
    assert config.energy_penalty_weight == 0.01
    assert config.normalize_energy_j == 1_000.0


def test_empty_records_do_not_seed_advantage_baseline() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(normalize_profit_scale=100.0, profit_baseline_ema_alpha=1.0)
    )

    assert calculator.aggregate([]) == 0.0
    assert calculator.aggregate([_record(profit=10.0, finish_delay_s=0.0)]) == 0.1


def test_profit_above_baseline_gets_advantage_bonus() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(
            normalize_profit_scale=100.0,
            completion_delay_penalty=0.0,
            advantage_reward_weight=0.5,
            advantage_reward_clip=None,
            profit_baseline_ema_alpha=1.0,
        )
    )

    first_reward = calculator.aggregate([_record(profit=10.0, finish_delay_s=0.0)])
    second_reward = calculator.aggregate([_record(profit=20.0, finish_delay_s=0.0)])
    lower_reward = calculator.aggregate([_record(profit=15.0, finish_delay_s=0.0)])

    assert first_reward == 0.1
    assert second_reward == 0.25
    assert lower_reward == 0.15


def test_energy_penalty_is_added_without_changing_existing_reward_terms() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(
            normalize_profit_scale=100.0,
            completion_delay_penalty=0.0,
            energy_penalty_weight=0.2,
            normalize_energy_j=10.0,
            advantage_reward_weight=0.0,
        )
    )

    reward = calculator.aggregate(
        [_record(profit=10.0, finish_delay_s=0.0, total_energy_j=5.0)]
    )

    # 原收益为 0.1，新增能耗惩罚为 0.2 * 5 / 10 = 0.1。
    assert reward == 0.0


def test_higher_energy_gets_lower_reward_for_same_execution_result() -> None:
    config = RewardConfig(
        normalize_profit_scale=100.0,
        completion_delay_penalty=0.0,
        energy_penalty_weight=0.1,
        normalize_energy_j=10.0,
        advantage_reward_weight=0.0,
    )
    low_energy_calculator = SharedRewardCalculator(config)
    high_energy_calculator = SharedRewardCalculator(config)

    low_energy_reward = low_energy_calculator.aggregate(
        [_record(profit=10.0, finish_delay_s=0.0, total_energy_j=1.0)]
    )
    high_energy_reward = high_energy_calculator.aggregate(
        [_record(profit=10.0, finish_delay_s=0.0, total_energy_j=9.0)]
    )

    assert low_energy_reward > high_energy_reward


def test_completion_rate_is_enforced_as_adaptive_constraint_not_fixed_reward() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(
            normalize_profit_scale=100.0,
            energy_penalty_weight=0.0,
            completion_delay_penalty=0.0,
            advantage_reward_weight=0.0,
            minimum_long_term_completion_rate=0.9,
            completion_rate_ema_alpha=1.0,
            completion_constraint_dual_lr=0.1,
        )
    )

    feasible_reward = calculator.aggregate([_record(profit=10.0, completed=True)])
    violated_reward = calculator.aggregate([_record(profit=0.0, completed=False)])

    assert feasible_reward == 0.1
    assert violated_reward < 0.0


def test_energy_budget_dual_rises_on_violation_and_never_goes_negative() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(
            normalize_profit_scale=100.0,
            normalize_energy_j=1.0,
            energy_penalty_weight=0.0,
            completion_delay_penalty=0.0,
            advantage_reward_weight=0.0,
            minimum_long_term_completion_rate=0.0,
            long_term_energy_budget_j_per_step=1.0,
            energy_budget_ema_alpha=1.0,
            energy_constraint_dual_lr=0.1,
            energy_constraint_dual_max=2.0,
        )
    )

    calculator.aggregate([_record(profit=0.0, total_energy_j=5.0)])
    raised_multiplier = calculator.energy_constraint_multiplier
    assert raised_multiplier > 0.0
    assert calculator.energy_budget_violation_j == 4.0

    calculator.aggregate([_record(profit=0.0, total_energy_j=0.0)])
    assert 0.0 <= calculator.energy_constraint_multiplier < raised_multiplier


def test_partial_replica_capacity_rejection_is_penalized() -> None:
    calculator = SharedRewardCalculator(
        RewardConfig(
            normalize_profit_scale=100.0,
            capacity_drop_penalty=0.3,
            completion_delay_penalty=0.0,
            energy_penalty_weight=0.0,
            advantage_reward_weight=0.0,
            minimum_long_term_completion_rate=0.0,
            long_term_energy_budget_j_per_step=None,
        )
    )

    reward = calculator.aggregate(
        [
            _record(
                profit=10.0,
                requested_replica_count=2,
                capacity_rejected_replica_count=1,
            )
        ]
    )

    assert math.isclose(reward, -0.05)  # 0.1 profit - 0.3 * (1 / 2)
