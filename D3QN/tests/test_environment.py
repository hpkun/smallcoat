from copy import deepcopy
import unittest

import numpy as np

from drl_ra.config import load_config
from drl_ra.environment import Candidate, SAGINEnv, Task, smooth_shortfall


def tiny_config():
    config = deepcopy(load_config())
    config["environment"]["episode_steps"] = 5
    return config


class EnvironmentTests(unittest.TestCase):
    def test_paper_action_and_state_dimensions(self):
        env = SAGINEnv(tiny_config(), seed=1)
        state, info = env.reset(seed=1)
        self.assertEqual(env.action_dim, 20)
        self.assertEqual(env.state_dim, 108)
        self.assertEqual(state.shape, (108,))
        self.assertEqual(info["action_mask"].shape, (20,))
        self.assertTrue(info["action_mask"][0])

    def test_episode_metrics_are_finite(self):
        env = SAGINEnv(tiny_config(), seed=2)
        _, info = env.reset(seed=2)
        done = False
        while not done:
            available = np.flatnonzero(info["action_mask"])
            _, _, terminated, truncated, info = env.step(int(available[0]))
            done = terminated or truncated
        summary = env.summary()
        self.assertEqual(summary["tasks"], 5)
        self.assertTrue(all(np.isfinite(value) for value in summary.values()))
        self.assertTrue(0 <= summary["tcr"] <= 100)
        self.assertTrue(0 <= summary["cvr"] <= 100)

    def test_normalized_smooth_cost_matches_budget_scale(self):
        self.assertLess(smooth_shortfall(-1.0, 0.05), 1e-6)
        self.assertTrue(0.03 < smooth_shortfall(0.0, 0.05) < 0.04)
        self.assertGreater(smooth_shortfall(0.2, 0.05), 0.2)

    def test_seed_reproduces_initial_observation(self):
        env_a = SAGINEnv(tiny_config(), seed=9)
        env_b = SAGINEnv(tiny_config(), seed=9)
        state_a, info_a = env_a.reset(seed=9)
        state_b, info_b = env_b.reset(seed=9)
        np.testing.assert_allclose(state_a, state_b)
        np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])

    def test_device_categories_are_fixed(self):
        env = SAGINEnv(tiny_config(), seed=3)
        expected = {"environmental": 40, "traffic": 25, "meter": 20, "safety": 15}
        self.assertEqual({kind: int(np.sum(env.device_types == kind)) for kind in expected}, expected)
        for _ in range(30):
            task = env._sample_task()
            self.assertEqual(task.kind, env.device_types[task.device])

    def test_local_queues_are_per_device(self):
        env = SAGINEnv(tiny_config(), seed=4)
        env.local_queue_s[:] = 0.0
        env.current_task = Task("environmental", 80_000.0, 10_000_000_000.0, 10.0, 0.85, 0)
        env._observation()
        env.step(0)
        self.assertGreater(env.local_queues[0], 0.0)
        self.assertEqual(env.local_queues[1], 0.0)

    def test_uav_physical_speed_range(self):
        env = SAGINEnv(tiny_config(), seed=5)
        before = env._uav_positions(0.0)
        after = env._uav_positions(1.0)
        speeds = np.linalg.norm(after - before, axis=1) * 1000.0
        np.testing.assert_array_less(speeds, env.uav_speeds_mps * 1.01 + 1e-3)
        np.testing.assert_array_less(env.uav_speeds_mps, np.full(env.num_uavs, 15.001))
        np.testing.assert_array_less(np.full(env.num_uavs, 4.999), env.uav_speeds_mps)

    def test_replica_reservations_are_released(self):
        env = SAGINEnv(tiny_config(), seed=6)
        env.current_task = Task("safety", 800_000.0, 500_000_000.0, 3.0, 0.995, 0)
        env._observation()
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["reserved_after_event"], 0.0)
        self.assertEqual(env.reserved_capacity.sum(), 0.0)

    def test_satellite_visibility_uses_clock(self):
        env = SAGINEnv(tiny_config(), seed=7)
        env.current_time_s = 0.0
        first = env._satellite_state(0)
        env.current_time_s = env.env_cfg["satellite_window_s"] + 1.0
        second = env._satellite_state(0)
        self.assertNotEqual(first[0], second[0])

    def test_capacity_gate_uses_cpu_units(self):
        env = SAGINEnv(tiny_config(), seed=8)
        action = 1
        env.edge_positions[0] = env.device_positions[0]
        env.total_capacity[action] = 1e6
        task = Task("traffic", 1e6, 5e9, 1.0, 0.95, 0)
        candidate = env._edge_candidate(task, env.device_positions[0], 0)
        self.assertGreater(candidate.required_capacity, env.available_capacity[action])
        self.assertFalse(candidate.available)
        env.env_cfg["capacity_gating"] = False
        candidate_without_gate = env._edge_candidate(task, env.device_positions[0], 0)
        self.assertTrue(candidate_without_gate.available)

    def test_used_capacity_lives_until_finish(self):
        env = SAGINEnv(tiny_config(), seed=10)
        candidate = Candidate(1, "edge", True, 1.0, 1.0, 0.9, 0.95, 0.99, 0.0, 10.0, 0.1)
        env._reserve([candidate])
        env._promote_winner(candidate, finish_time=1.0)
        self.assertEqual(env.reserved_capacity[1], 0.0)
        self.assertEqual(env.used_capacity[1], 10.0)
        env.current_time_s = 1.1
        env._release_completed_allocations()
        self.assertEqual(env.used_capacity[1], 0.0)
