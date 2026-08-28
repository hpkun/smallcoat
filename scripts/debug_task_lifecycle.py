from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import format_records_debug_report
from train import build_small_scale_env


def main() -> None:
    """打印一个小规模环境中的任务生命周期调试信息。"""

    env = build_small_scale_env()
    records = env.base_env.step(
        slot_length_s=env.base_env.simulation_config.slot_length_s,
        current_time_s=0.0,
        delay_sensitivity_lambda=env.base_env.task_generator.task_model_config.delay_sensitivity_lambda,
        move_uavs=True,
    )
    print(format_records_debug_report(records))


if __name__ == "__main__":
    main()
