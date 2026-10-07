import unittest

import numpy as np

from drl_ra.baselines import nearest_policy
from drl_ra.environment import Candidate


class BaselineTests(unittest.TestCase):
    def test_greedy_nearest_uses_distance_not_latency(self):
        candidates = [
            Candidate(0, "local", True, 0.01, 1.0, 0.9, 1.0, 1.0, 0.0, 0.0, 0.0),
            Candidate(1, "edge", True, 1.0, 1.0, 0.9, 0.9, 1.0, 0.0, 1.0, 0.2),
            Candidate(2, "edge", True, 0.1, 1.0, 0.9, 0.9, 1.0, 0.0, 1.0, 1.0),
        ]
        action = nearest_policy(np.ones(3, dtype=bool), candidates, None, np.random.default_rng(0))
        self.assertEqual(action, 1)
