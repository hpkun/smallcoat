from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ObservationBuilder
from src import SharedRewardCalculator
from src import build_balanced_scenario
from train import build_training_env


def main() -> None:
    """最小烟雾测试。"""

    env = build_training_env()
    observations, action_specs = env.reset()
    assert isinstance(observations, dict)
    assert isinstance(action_specs, dict)

    scenario = build_balanced_scenario()
    assert scenario.name == "balanced"

    reward_calculator = SharedRewardCalculator()
    assert reward_calculator.aggregate([]) == 0.0

    base_env = env.base_env
    builder = ObservationBuilder(
        communication_model=base_env.communication_model,
        network_profiles=base_env.network_profiles,
        area_side_length_m=base_env.simulation_config.area.side_length_m,
    )
    assert builder is not None
    print("smoke-ok")


if __name__ == "__main__":
    main()
