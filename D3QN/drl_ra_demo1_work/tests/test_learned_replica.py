from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import MetricsOnlySAGINEnv, build_learned_replica_agent, train_learned_replica_agent
from drl_ra.redundancy import combined_reliability


def tiny_config() -> dict:
    config = deepcopy(load_config())
    config["environment"]["episode_steps"] = 3
    config["training"]["batch_size"] = 2
    config["learned_replica"]["training"]["batch_size"] = 2
    return config


class LearnedReplicaTests(unittest.TestCase):
    def fixed_snapshot(self, required=0.95, available=3, reliability=0.8, env_type=SAGINEnv):
        env = env_type(tiny_config(), seed=21)
        env.current_task = replace(env.current_task, reliability_required=required)
        env._last_candidates = [
            replace(item, available=item.action < available, reliability=reliability)
            for item in env.candidates
        ]
        return env

    def test_state_and_action_dimensions(self):
        env = SAGINEnv(tiny_config(), seed=21)
        env.reset(seed=21)
        state = env.learned_redundancy_observation([0])
        mask = env.learned_redundancy_action_mask([0])
        self.assertEqual(state.shape, (132,))
        self.assertEqual(mask.shape, (21,))
        self.assertFalse(mask[0])
        selected_reliability = combined_reliability([env.candidates[0]])
        self.assertEqual(bool(mask[20]), selected_reliability >= env.current_task.reliability_required or not mask[:20].any())

    def test_stop_is_masked_below_requirement_when_backup_exists(self):
        env = self.fixed_snapshot()
        mask = env.learned_redundancy_action_mask([0])
        self.assertFalse(mask[-1])
        self.assertEqual(np.flatnonzero(mask).tolist(), [1, 2])

    def test_stop_and_backup_remain_valid_once_requirement_is_met(self):
        env = self.fixed_snapshot()
        mask = env.learned_redundancy_action_mask([0, 1])
        self.assertEqual(np.flatnonzero(mask).tolist(), [2, env.action_dim])

    def test_stop_is_valid_at_exact_reliability_boundary(self):
        env = self.fixed_snapshot(required=0.8)
        self.assertTrue(env.learned_redundancy_action_mask([0])[-1])

    def test_stop_is_only_action_without_backup_even_below_requirement(self):
        env = self.fixed_snapshot(available=1)
        self.assertEqual(np.flatnonzero(env.learned_redundancy_action_mask([0])).tolist(), [env.action_dim])

    def test_greedy_and_exploratory_policies_cannot_stop_unreliably(self):
        for epsilon in (0.0, 1.0):
            env = self.fixed_snapshot()
            agent = build_learned_replica_agent(env, env.config, seed=21, device="cpu")
            q_values = torch.zeros(1, env.learned_redundancy_action_dim)
            q_values[0, env.action_dim] = 1000.0
            with patch.object(agent.primary, "act", return_value=0), \
                 patch.object(agent.replica.online, "forward", return_value=q_values):
                _, selected, transitions = agent.decide(env, replica_epsilon=epsilon)
            self.assertGreaterEqual(len(selected), 2)
            self.assertNotEqual(transitions[0][1], env.action_dim)
            self.assertGreaterEqual(combined_reliability([env.candidates[item] for item in selected]), 0.95)
            if epsilon == 0.0:
                self.assertEqual(len(selected), 2)
                self.assertEqual(agent.last_stop_reason, "active_stop")

    def test_reliable_policy_can_choose_third_replica(self):
        env = self.fixed_snapshot()
        agent = build_learned_replica_agent(env, env.config, seed=21, device="cpu")
        q_values = torch.ones(1, env.learned_redundancy_action_dim)
        q_values[0, env.action_dim] = -1000.0
        with patch.object(agent.primary, "act", return_value=0), \
             patch.object(agent.replica.online, "forward", return_value=q_values):
            _, selected, _ = agent.decide(env)
        self.assertEqual(selected, [0, 1, 2])
        self.assertEqual(agent.last_stop_reason, "max_replica_stop")

    def test_replica_cap_submits_without_an_extra_stop_decision(self):
        env = self.fixed_snapshot(required=0.999)
        agent = build_learned_replica_agent(env, env.config, seed=21, device="cpu")
        with patch.object(agent.primary, "act", return_value=0), \
             patch.object(agent.replica, "act", wraps=agent.replica.act) as replica_act:
            _, selected, transitions = agent.decide(env)
        self.assertEqual(len(selected), 3)
        self.assertEqual(replica_act.call_count, 2)
        self.assertTrue(transitions[-1][3])
        self.assertEqual(agent.last_stop_reason, "max_replica_stop")

    def test_no_backup_stop_records_infeasible_even_without_audits(self):
        for env_type in (SAGINEnv, MetricsOnlySAGINEnv):
            env = self.fixed_snapshot(available=1, env_type=env_type)
            agent = build_learned_replica_agent(env, env.config, seed=21, device="cpu")
            _, selected, transitions = agent.decide(env)
            self.assertEqual(selected, [0])
            self.assertEqual(agent.last_stop_reason, "infeasible")
            self.assertEqual(transitions[-1][1], env.action_dim)
            _, _, _, _, info = env.step_with_replicas(selected, stop_reason=agent.last_stop_reason)
            self.assertEqual(info["stop_reason"], "infeasible")
            self.assertEqual(info["infeasible"], 1)
            self.assertEqual(env.metrics[-1]["infeasible"], 1)

    def test_no_backup_after_requirement_is_met_is_not_infeasible(self):
        env = self.fixed_snapshot(required=0.8, available=1)
        agent = build_learned_replica_agent(env, env.config, seed=21, device="cpu")
        _, selected, _ = agent.decide(env)
        self.assertEqual(agent.last_stop_reason, "no_feasible_candidate_stop")
        _, _, _, _, info = env.step_with_replicas(selected, stop_reason=agent.last_stop_reason)
        self.assertEqual(info["infeasible"], 0)

    def test_selected_and_unavailable_actions_are_masked(self):
        env = SAGINEnv(tiny_config(), seed=22)
        _, info = env.reset(seed=22)
        primary = int(np.flatnonzero(info["action_mask"])[0])
        mask = env.learned_redundancy_action_mask([primary])
        self.assertFalse(mask[primary])
        for action, candidate in enumerate(env.candidates):
            if not candidate.available:
                self.assertFalse(mask[action])

    def test_maximum_replica_set_only_allows_stop(self):
        env = SAGINEnv(tiny_config(), seed=23)
        _, info = env.reset(seed=23)
        available = np.flatnonzero(info["action_mask"]).tolist()
        while len(available) < 3:
            env.reset(seed=env.seed + 1)
            available = [item.action for item in env.candidates if item.available]
        mask = env.learned_redundancy_action_mask(available[:3])
        self.assertFalse(mask[:-1].any())
        self.assertTrue(mask[-1])

    def test_internal_selection_does_not_advance_or_resample(self):
        env = SAGINEnv(tiny_config(), seed=24)
        env.reset(seed=24)
        before_time = env.current_time_s
        before_candidates = env.candidates
        env.learned_redundancy_observation([0])
        env.learned_redundancy_action_mask([0])
        self.assertEqual(env.current_time_s, before_time)
        self.assertEqual(env.candidates, before_candidates)

    def test_replica_set_executes_exactly_one_environment_step(self):
        env = SAGINEnv(tiny_config(), seed=25)
        _, info = env.reset(seed=25)
        selected = np.flatnonzero(info["action_mask"]).tolist()[:2]
        before_step = env.step_count
        before_time = env.current_time_s
        _, _, _, _, result = env.step_with_replicas(selected)
        self.assertEqual(env.step_count, before_step + 1)
        self.assertGreater(env.current_time_s, before_time)
        self.assertEqual(result["replica_actions"], selected)
        self.assertEqual(result["reserved_after_event"], 0.0)

    def test_invalid_sets_are_rejected_without_advancing_time(self):
        env = SAGINEnv(tiny_config(), seed=26)
        env.reset(seed=26)
        before = (env.step_count, env.current_time_s)
        with self.assertRaises(ValueError):
            env.step_with_replicas([0, 0])
        self.assertEqual((env.step_count, env.current_time_s), before)

    def test_terminal_internal_transition_does_not_bootstrap(self):
        env = SAGINEnv(tiny_config(), seed=27)
        env.reset(seed=27)
        agent = build_learned_replica_agent(env, tiny_config(), seed=27, device="cpu")
        _, selected, transitions = agent.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
        self.assertTrue(transitions[-1][3])
        self.assertTrue(1 <= len(selected) <= 3)
        agent.replica.observe_sequence(transitions, final_reward=1.0)
        stored = list(agent.replica.replay._data)
        self.assertEqual(stored[-1][2], 1.0)
        self.assertTrue(stored[-1][4])
        self.assertTrue(all(row[2] == 0.0 for row in stored[:-1]))
        self.assertEqual(stored[-1][6], 0.99)

    def test_replica_task_transition_bootstraps_from_next_task(self):
        config = tiny_config()
        env = SAGINEnv(config, seed=31)
        agent = build_learned_replica_agent(env, config, seed=31, device="cpu")
        replica = agent.replica
        state = np.zeros(env.learned_redundancy_state_dim, dtype=np.float32)
        intra_state = np.ones_like(state)
        bootstrap_state = np.full_like(state, 2.0)
        intra_mask = np.ones(env.learned_redundancy_action_dim, dtype=bool)
        bootstrap_mask = np.zeros_like(intra_mask)
        bootstrap_mask[-1] = True
        transitions = [
            (state, 0, intra_state, False, intra_mask),
            (intra_state, env.action_dim, intra_state, True, intra_mask),
        ]

        replica.observe_sequence(
            transitions,
            final_reward=1.0,
            episode_done=False,
            bootstrap_state=bootstrap_state,
            bootstrap_mask=bootstrap_mask,
        )

        first, last = list(replica.replay._data)
        self.assertEqual(first[2], 0.0)
        self.assertFalse(first[4])
        self.assertEqual(first[6], 1.0)
        np.testing.assert_array_equal(first[3], intra_state)
        self.assertEqual(last[2], 1.0)
        self.assertFalse(last[4])
        self.assertEqual(last[6], 0.99)
        np.testing.assert_array_equal(last[3], bootstrap_state)
        np.testing.assert_array_equal(last[5], bootstrap_mask)

    def test_replica_continuation_requires_next_task_state(self):
        env = SAGINEnv(tiny_config(), seed=32)
        agent = build_learned_replica_agent(env, tiny_config(), seed=32, device="cpu")
        _, _, transitions = agent.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
        with self.assertRaises(ValueError):
            agent.replica.observe_sequence(transitions, final_reward=0.5, episode_done=False)

    def test_joint_checkpoint_round_trip(self):
        config = tiny_config()
        env = SAGINEnv(config, seed=28)
        agent = build_learned_replica_agent(env, config, seed=28, device="cpu")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            agent.save(checkpoint, metadata={"method": "drl-ra-learned-replica"})
            restored = build_learned_replica_agent(env, config, seed=29, device="cpu")
            metadata = restored.load(checkpoint)
        self.assertEqual(metadata["method"], "drl-ra-learned-replica")
        for source, target in zip(agent.primary.online.parameters(), restored.primary.online.parameters()):
            np.testing.assert_allclose(source.detach().numpy(), target.detach().numpy())

    def test_training_entrypoint_runs_one_task(self):
        config = tiny_config()
        config["training"]["episodes"] = 1
        config["environment"]["episode_steps"] = 1
        _, history = train_learned_replica_agent(config, seed=30, device="cpu", progress=False)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["tasks"], 1.0)

    def test_training_bootstraps_between_tasks_and_stops_at_episode_end(self):
        config = tiny_config()
        config["training"]["episodes"] = 1
        config["environment"]["episode_steps"] = 2
        agent, history = train_learned_replica_agent(config, seed=33, device="cpu", progress=False)

        stored = list(agent.replica.replay._data)
        self.assertEqual(history[0]["tasks"], 2.0)
        self.assertEqual(sum(bool(row[4]) for row in stored), 1)
        self.assertEqual(sum(row[6] == 0.99 for row in stored), 2)


if __name__ == "__main__":
    unittest.main()
