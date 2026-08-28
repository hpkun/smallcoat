from __future__ import annotations

from train import build_reward_config


def test_profit_only_reward_profile_disables_proposed_reward_terms() -> None:
    config = build_reward_config("profit-only")

    assert config.energy_penalty_weight == 0.0
    assert config.deadline_failure_penalty == 0.0
    assert config.capacity_drop_penalty == 0.0
    assert config.reliability_failure_penalty == 0.0
    assert config.completion_delay_penalty == 0.0
    assert config.completion_constraint_dual_lr == 0.0
    assert config.advantage_reward_weight == 0.0


def test_energy_penalty_weight_can_be_overridden() -> None:
    config = build_reward_config("proposed", energy_penalty_weight=0.0)

    assert config.energy_penalty_weight == 0.0
