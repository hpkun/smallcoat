from copy import deepcopy
import unittest

import numpy as np
import torch

from drl_ra.agent import D3QNAgent
from drl_ra.config import load_config
from drl_ra.models import QNetwork, masked_q_values


def tiny_config():
    config = deepcopy(load_config())
    config["training"]["batch_size"] = 2
    config["training"]["replay_capacity"] = 10
    config["training"]["target_update_steps"] = 2
    config["training"]["lagrange_update_steps"] = 1
    return config


class AgentTests(unittest.TestCase):
    def test_dueling_network_output_shape(self):
        network = QNetwork(7, 4, (8, 6), dueling=True)
        output = network(torch.zeros(3, 7))
        self.assertEqual(output.shape, (3, 4))

    def test_mask_excludes_unavailable_action(self):
        values = torch.tensor([[1.0, 99.0, 2.0]])
        mask = torch.tensor([[True, False, True]])
        self.assertEqual(masked_q_values(values, mask).argmax().item(), 2)

    def test_agent_learns_and_updates_nonnegative_lagrange(self):
        config = tiny_config()
        agent = D3QNAgent(4, 3, config, seed=0)
        state = np.zeros(4, dtype=np.float32)
        next_state = np.ones(4, dtype=np.float32)
        mask = np.ones(3, dtype=bool)
        self.assertIsNone(agent.observe(state, 0, 0.0, 0.5, next_state, False, mask))
        loss = agent.observe(next_state, 1, 1.0, 0.5, state, True, mask)
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(agent.lagrange, 0.0)
