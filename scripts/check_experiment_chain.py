from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cmaddpg import CMADDPGSystem
from train import build_small_scale_env


def main() -> None:
    env = build_small_scale_env()
    system = CMADDPGSystem(device="cpu")
    observations, action_specs = env.reset()
    initial_energy_j = sum(uav.remaining_energy_j for uav in env.base_env.uavs)
    task_count = 0
    completed_count = 0
    task_energy_j = 0.0
    update_count = 0

    for step in range(100):
        for agent_id, observation in observations.items():
            system.ensure_agent(
                agent_id=agent_id,
                state_dim=int(observation.shape[0]),
                action_spec=action_specs[agent_id],
            )
        raw_actions = system.act(observations)
        env_actions, critic_actions = system.decode_actions(raw_actions)
        next_observations, _, done, info = env.step(env_actions)
        records = info["records"]
        task_count += len(records)
        completed_count += sum(record.completed_before_deadline for record in records)
        task_energy_j += sum(record.total_energy_j for record in records)
        system.store_transitions(
            observations=observations,
            critic_actions=critic_actions,
            shared_reward=info["shared_reward"],
            next_observations=next_observations,
            done=done,
        )
        if step % 10 == 0 and system.update(batch_size=8) is not None:
            update_count += 1
        observations = next_observations
        action_specs = info["action_specs"]

    final_energy_j = sum(uav.remaining_energy_j for uav in env.base_env.uavs)
    battery_drop_j = initial_energy_j - final_energy_j
    propulsion_energy_j = (
        len(env.base_env.uavs)
        * env.base_env.energy_model.config.uav_propulsion_power_w
        * env.base_env.simulation_config.slot_length_s
        * 100
    )
    print(
        f"tasks={task_count} completed={completed_count} "
        f"task_energy_j={task_energy_j:.3f} battery_drop_j={battery_drop_j:.3f} "
        f"updates={update_count} buffer={len(system.replay_buffer)}"
    )
    assert task_count > 0, "No task traversed the environment."
    assert task_energy_j > 0.0, "Executed tasks did not consume energy."
    assert battery_drop_j >= propulsion_energy_j, "Propulsion energy was not charged."
    assert update_count > 0, "Actor/critic update did not run."

    for uav in env.base_env.uavs:
        uav.remaining_energy_j = 0.0
    before = env.episode_generated_task_count
    for _ in range(50):
        _, _, _, info = env.step({})
        assert info["records"] == []
        if info["pending_ground_task_count"] > 0:
            break
    assert info["generated_task_count"] >= before
    assert info["pending_ground_task_count"] > 0


if __name__ == "__main__":
    main()
