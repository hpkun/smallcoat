from copy import deepcopy
import unittest

import numpy as np

from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv, smooth_shortfall


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
