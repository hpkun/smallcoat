from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import build_small_scale_env


def main() -> None:
    """Run one simplified SAGIN scheduling slot."""

    env = build_small_scale_env()
    records = env.base_env.step(
        slot_length_s=env.base_env.simulation_config.slot_length_s,
        current_time_s=0.0,
        delay_sensitivity_lambda=env.base_env.task_generator.task_model_config.delay_sensitivity_lambda,
        move_uavs=True,
    )

    print(f"scheduled_tasks={len(records)}")
    for record in records[:10]:
        print(
            record.task_id,
            record.ingress_uav_id,
            record.decision_uav_id,
            record.target_node_id,
            f"eta={record.compute_priority_eta:.3f}",
            f"delay={record.total_delay_s:.6f}s",
            f"deadline_ok={record.completed_before_deadline}",
            f"profit={record.realized_profit:.2f}",
        )


if __name__ == "__main__":
    main()
