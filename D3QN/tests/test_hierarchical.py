from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from drl_ra.config import load_config
from drl_ra.environment import NTLAction, NTL_AIR, NTL_BOTH, NTL_NONE, NTL_SPACE, SAGINEnv, Task
from drl_ra.experiment import train_hierarchical_agent
from drl_ra.hierarchical import HierarchicalAgent


def tiny_config():
    config = deepcopy(load_config())
    config["environment"]["episode_steps"] = 8
    config["environment"]["reliability_requirement_override"] = 0.999
    config["training"]["episodes"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["replay_capacity"] = 32
    config["ppo_training"]["update_epochs"] = 1
    config["ppo_training"]["minibatch_size"] = 4
    return config


class HierarchicalTests(unittest.TestCase):
    def test_hierarchical_dimensions_do_not_change_legacy_dimensions(self):
        env = SAGINEnv(tiny_config(), seed=11)
        legacy_state, legacy_info = env.reset(seed=11)
        self.assertEqual(env.action_dim, 20)
        self.assertEqual(env.state_dim, 108)
        self.assertEqual(legacy_state.shape, (108,))
        self.assertEqual(legacy_info["action_mask"].shape, (20,))
        self.assertEqual(env.ground_action_dim, 12)
        self.assertEqual(env.ground_observation().shape, (env.ground_state_dim,))
        context = env.prepare_hierarchical(0)
        self.assertEqual(context["critic_state"].shape, (env.critic_state_dim,))

    def test_gate_bypasses_ppo_for_satisfied_ground_action(self):
        env = SAGINEnv(tiny_config(), seed=12)
        env.current_task = Task("environmental", 80_000.0, 1_000_000.0, 1000.0, 0.1, 0)
        env._observation()
        context = env.prepare_hierarchical(0)
        self.assertFalse(context["gate"])
        _, _, _, _, info = env.step_hierarchical(0, NTLAction(NTL_BOTH, 1, 1), context)
        self.assertEqual(info["gate"], 0)
        self.assertEqual(info["replicas"], 1)
        self.assertEqual(info["ntl_mode"], NTL_NONE)

    def test_factorized_policy_emits_consistent_joint_action(self):
        config = tiny_config()
        env = SAGINEnv(config, seed=13)
        agent = HierarchicalAgent(env, config, seed=13)
        context = env.prepare_hierarchical(0)
        context["gate"] = True
        context["air_mask"][:] = True
        context["space_mask"][:] = True
        context["mode_mask"][:] = True
        action, _, _, masks = agent.ppo.select_action(context, deterministic=False, active=True)
        self.assertEqual(action.air != 0, action.mode in (NTL_AIR, NTL_BOTH))
        self.assertEqual(action.space != 0, action.mode in (NTL_SPACE, NTL_BOTH))
        self.assertTrue(masks[0][action.air])
        self.assertTrue(masks[1][action.space])
        self.assertTrue(masks[2][action.mode])

    def test_smoke_training_updates_both_levels(self):
        agent, history = train_hierarchical_agent(tiny_config(), seed=14, progress=False)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["phase"], "joint-training")
        self.assertTrue(np.isfinite(history[0]["ppo_loss"]))
        self.assertEqual(history[0]["ppo_active_steps"], 8)
        self.assertGreater(agent.ground.training_steps, 0)

    def test_ppo_is_one_shared_multi_head_model(self):
        config = tiny_config()
        env = SAGINEnv(config, seed=15)
        agent = HierarchicalAgent(env, config, seed=15)
        self.assertTrue(hasattr(agent.ppo.policy, "encoder"))
        self.assertTrue(hasattr(agent.ppo.policy, "mode_head"))
        self.assertTrue(hasattr(agent.ppo.policy, "air_head"))
        self.assertTrue(hasattr(agent.ppo.policy, "space_head"))

    def test_two_deployment_checkpoints_round_trip(self):
        config = tiny_config()
        env = SAGINEnv(config, seed=16)
        source = HierarchicalAgent(env, config, seed=16)
        with TemporaryDirectory() as directory:
            ground_path, ppo_path = source.save_components(directory, {"method": "d3qn-ppo"})
            restored = HierarchicalAgent(env, config, seed=17)
            restored.load_components(ground_path, ppo_path)
            source_ground = next(source.ground.online.parameters()).detach().numpy()
            restored_ground = next(restored.ground.online.parameters()).detach().numpy()
            source_ppo = next(source.ppo.policy.parameters()).detach().numpy()
            restored_ppo = next(restored.ppo.policy.parameters()).detach().numpy()
            np.testing.assert_allclose(source_ground, restored_ground)
            np.testing.assert_allclose(source_ppo, restored_ppo)


if __name__ == "__main__":
    unittest.main()
