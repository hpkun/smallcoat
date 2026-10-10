from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import numpy as np

from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_learned_replica_agent


class ReplicaBehaviorCloningTests(unittest.TestCase):
    def test_masked_behavior_clone_updates_replica_and_syncs_target(self):
        config = deepcopy(load_config())
        env = SAGINEnv(config, seed=7)
        agent = build_learned_replica_agent(env, config, seed=7, device="cpu")
        states = np.zeros((4, env.learned_redundancy_state_dim), dtype=np.float32)
        masks = np.ones((4, env.learned_redundancy_action_dim), dtype=bool)
        masks[:, 0] = False
        actions = np.full(4, env.action_dim, dtype=np.int64)
        losses = agent.replica.behavior_clone(states, masks, actions, epochs=2, batch_size=2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses).all())
        for online, target in zip(agent.replica.online.parameters(), agent.replica.target.parameters()):
            np.testing.assert_allclose(online.detach().numpy(), target.detach().numpy())

    def test_behavior_clone_rejects_masked_teacher_action(self):
        config = deepcopy(load_config())
        env = SAGINEnv(config, seed=8)
        agent = build_learned_replica_agent(env, config, seed=8, device="cpu")
        states = np.zeros((1, env.learned_redundancy_state_dim), dtype=np.float32)
        masks = np.zeros((1, env.learned_redundancy_action_dim), dtype=bool)
        with self.assertRaisesRegex(ValueError, "masked out"):
            agent.replica.behavior_clone(states, masks, np.asarray([0]), epochs=1)

    def test_bc_checkpoint_is_loaded_into_replica_only(self):
        config = deepcopy(load_config())
        env = SAGINEnv(config, seed=9)
        source = build_learned_replica_agent(env, config, seed=9, device="cpu")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "replica_bc.pt"
            source.replica.save(checkpoint, metadata={"artifact": "replica_behavior_cloning"})
            loaded_config = deepcopy(config)
            loaded_config["learned_replica"]["bc_checkpoint"] = str(checkpoint)
            restored = build_learned_replica_agent(env, loaded_config, seed=10, device="cpu")
        for source_param, target_param in zip(source.replica.online.parameters(), restored.replica.online.parameters()):
            np.testing.assert_allclose(source_param.detach().numpy(), target_param.detach().numpy())


if __name__ == "__main__":
    unittest.main()
