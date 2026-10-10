from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_agent, build_learned_replica_agent, method_options, train_agent, train_learned_replica_agent
from drl_ra.trace import TraceEnv, evaluation_config, generate_trace, load_trace, save_trace, trace_digest, validate_trace
from replay_trace import METHODS, METRICS_ONLY_FIELDS, STOP_FIELDS, run_replay, verify_shared_inputs


def tiny_config():
    config = load_config()
    config["environment"].update(num_devices=8, num_edges=2, num_uavs=2, num_satellites=1, area_km=2.0, edge_coverage_km=100.0, capacity_gating=False)
    config["training"]["hidden_sizes"] = [16]
    config["learned_replica"]["training"]["hidden_sizes"] = [12]
    return config


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.trace = generate_trace(self.config, seed=42, steps=4)

    def env(self, trace=None):
        trace = self.trace if trace is None else trace
        return TraceEnv(evaluation_config(self.config, trace), trace)

    def test_generation_is_reproducible_and_saved_inputs_are_validated(self):
        self.assertEqual(self.trace, generate_trace(self.config, seed=42, steps=4))
        self.assertNotEqual(self.trace["sha256"], generate_trace(self.config, seed=43, steps=4)["sha256"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            save_trace(self.trace, path)
            self.assertEqual(load_trace(path), self.trace)
            with self.assertRaises(FileExistsError):
                save_trace(self.trace, path)
        corrupt = deepcopy(self.trace)
        corrupt["entries"][0]["task"]["cycles"] *= 2
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_trace(corrupt)
        corrupt["sha256"] = trace_digest(corrupt)
        corrupt["entries"][1]["arrival_time_s"] += 1
        corrupt["sha256"] = trace_digest(corrupt)
        with self.assertRaisesRegex(ValueError, "arrival time"):
            validate_trace(corrupt)

    def test_different_replica_counts_keep_tasks_arrivals_and_channels_identical(self):
        single, replicated = self.env(), self.env()
        selected_counts = []
        for entry in self.trace["entries"]:
            for env in (single, replicated):
                self.assertEqual(asdict(env.current_task), entry["task"])
                self.assertEqual(env.current_time_s, entry["arrival_time_s"])
                self.assertEqual(env._channel_uniform(0, "rain", 0, 6), single._channel_uniform(0, "rain", 0, 6))
            # Only resources vary, not the input timeline, when a policy adds replicas.
            actions = [item.action for item in replicated.candidates if item.available][:3]
            selected_counts.append(len(actions))
            single.step_with_replicas([0])
            replicated.step_with_replicas(actions)
        self.assertTrue(all(count == 3 for count in selected_counts))
        self.assertTrue(all(row["replicas"] == 1 for row in single.metrics))
        self.assertTrue(all(row["replicas"] == 3 for row in replicated.metrics))
        self.assertTrue(all(b["energy_mj"] > a["energy_mj"] for a, b in zip(single.metrics, replicated.metrics)))
        self.assertTrue(all(row["replica_capacity_overhead"] > 0 for row in replicated.metrics))
        self.assertEqual(single.current_time_s, replicated.current_time_s)
        self.assertEqual(len(single.metrics), 4)
        self.assertIsNone(single.current_task)

    def test_repeated_observations_and_skipped_relay_do_not_resample_channels(self):
        env = self.env()
        state, mask = env._observation()
        before = env.candidates
        again, again_mask = env._observation()
        np.testing.assert_array_equal(state, again)
        np.testing.assert_array_equal(mask, again_mask)
        self.assertEqual(before, env.candidates)
        position = env.device_positions[env.current_task.device]
        with patch.object(env, "_uav_link", side_effect=lambda pos, index, data: (100.0 if index == 0 else 0.02, 0.94, 1.0, True)):
            first = env._sat_candidate(env.current_task, position, 0)
            self.assertEqual(first.relay_uav, 1)
            env.uav_battery[0] = 0.0
            after_skip = env._sat_candidate(env.current_task, position, 0)
        self.assertEqual(first, after_skip)

    def test_outcomes_are_node_indexed_and_independent_of_selection_order(self):
        trace = deepcopy(self.trace)
        trace["entries"][0]["success_uniforms"][0] = 0.0
        trace["entries"][0]["success_uniforms"][1] = 0.999999
        trace["sha256"] = trace_digest(trace)
        first, second = self.env(trace), self.env(trace)
        self.assertTrue(first._replica_succeeds(first.candidates[0]))
        self.assertFalse(first._replica_succeeds(first.candidates[1]))
        _, reward_a, _, _, info_a = first.step_with_replicas([0, 1])
        _, reward_b, _, _, info_b = second.step_with_replicas([1, 0])
        self.assertEqual(reward_a, reward_b)
        self.assertEqual(info_a["completed"], info_b["completed"])
        self.assertEqual(info_a["latency_s"], info_b["latency_s"])
        np.testing.assert_array_equal(first.local_queues, second.local_queues)
        np.testing.assert_array_equal(first.available_capacity, second.available_capacity)

    def test_reset_restores_initial_state_and_exhaustion_is_explicit(self):
        env = self.env()
        state, info = env.reset()
        original = env.candidates
        for _ in range(4):
            next_state, _, done, _, next_info = env.step_with_replicas([0])
        self.assertTrue(done)
        self.assertFalse(next_info["action_mask"].any())
        self.assertEqual(next_state.shape, state.shape)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            env.step(0)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            env.step_with_replicas([0])
        restored_state, restored_info = env.reset()
        np.testing.assert_array_equal(state, restored_state)
        np.testing.assert_array_equal(info["action_mask"], restored_info["action_mask"])
        self.assertEqual(original, env.candidates)
        with self.assertRaises(ValueError):
            env.reset(seed=99)

    def test_common_evaluation_config_preserves_checkpoint_network_settings(self):
        training = deepcopy(self.config)
        training["environment"]["arrival_rate"] = 2.0
        training["reward"]["energy"] = 5.0
        training["learned_replica"]["primary_checkpoint"] = "missing_warm_start.pt"
        config = evaluation_config(training, self.trace)
        self.assertEqual(config["training"], training["training"])
        self.assertEqual(config["learned_replica"]["training"]["hidden_sizes"], [12])
        self.assertEqual(config["environment"], self.trace["evaluation_config"]["environment"])
        self.assertEqual(config["reward"], self.trace["evaluation_config"]["reward"])
        self.assertIsNone(config["learned_replica"]["primary_checkpoint"])
        self.assertEqual(training["reward"]["energy"], 5.0)
        training["environment"]["num_edges"] += 1
        with self.assertRaisesRegex(ValueError, "topology"):
            evaluation_config(training, self.trace)

    def test_all_three_checkpoint_replays_share_inputs_and_emit_complete_audits(self):
        torch.set_num_threads(1)
        results = []
        with TemporaryDirectory() as directory:
            for method in METHODS:
                config = deepcopy(self.config)
                env = SAGINEnv(config, seed=1)
                policy = (build_learned_replica_agent(env, config, 1, "cpu") if method == "drl-ra-learned-replica"
                          else build_agent(method, env, config, 1, "cpu"))
                path = Path(directory) / f"{method}.pt"
                policy.save(path, metadata={"method": method, "seed": 1, "config": config})
                result = run_replay(self.trace, path, expected_method=method)
                repeated = run_replay(self.trace, path, expected_method=method)
                self.assertEqual(result["records"], repeated["records"])
                self.assertEqual(len(result["records"]), 4)
                self.assertEqual(result["policy_enable_redundancy"], method_options(method)["redundancy"])
                learned = method == "drl-ra-learned-replica"
                for row in result["records"]:
                    self.assertEqual(sum(row[field] for field in STOP_FIELDS), int(learned))
                if not learned:
                    self.assertTrue(all(value is None for value in result["summary"]["audit"]["stop_reasons"]["rates_pct"].values()))
                if method == "d3qn":
                    self.assertTrue(all(row["replicas"] == 1 for row in result["records"]))
                results.append(result)
        verify_shared_inputs(results, self.trace)

    def test_metrics_only_replay_skips_oracle_and_preserves_results(self):
        torch.set_num_threads(1)
        results = []
        with TemporaryDirectory() as directory:
            for method in METHODS:
                config = deepcopy(self.config)
                env = SAGINEnv(config, seed=1)
                policy = (build_learned_replica_agent(env, config, 1, "cpu") if method == "drl-ra-learned-replica"
                          else build_agent(method, env, config, 1, "cpu"))
                path = Path(directory) / f"{method}.pt"
                policy.save(path, metadata={"method": method, "seed": 1, "config": config})
                audited = run_replay(self.trace, path, expected_method=method)
                with patch.object(SAGINEnv, "replica_audit", side_effect=AssertionError("unexpected audit")), \
                     patch.object(SAGINEnv, "_replica_set_reward", side_effect=AssertionError("unexpected reward oracle")):
                    result = run_replay(self.trace, path, expected_method=method, metrics_only=True,
                                        violation_breakdown=method == "drl-ra-learned-replica")
                self.assertEqual(set(result["summary"]), set(METRICS_ONLY_FIELDS))
                self.assertEqual(result["summary"], {key: audited["summary"][key] for key in METRICS_ONLY_FIELDS})
                self.assertEqual(len(result["records"]), self.trace["steps"])
                if method == "drl-ra-learned-replica":
                    self.assertEqual(result["violation_breakdown"]["violation_tasks"], sum(row["violation"] for row in audited["records"]))
                    self.assertAlmostEqual(result["violation_breakdown"]["cvr_pct"], result["summary"]["cvr"])
                    self.assertEqual(sum(result["stop_reasons"]["counts"].values()), self.trace["steps"])
                    self.assertEqual(result["infeasible_task_ids"], [row["task_id"] for row in audited["records"] if row["infeasible"]])
                    self.assertEqual([row["stop_reason"] for row in result["records"]], [row["stop_reason"] for row in audited["records"]])
                results.append(result)
        verify_shared_inputs(results, self.trace)

    def test_metrics_only_training_skips_oracle_for_all_three_methods(self):
        torch.set_num_threads(1)
        with patch.object(SAGINEnv, "replica_audit", side_effect=AssertionError("unexpected audit")), \
             patch.object(SAGINEnv, "_replica_set_reward", side_effect=AssertionError("unexpected reward oracle")):
            for method in METHODS:
                config = deepcopy(self.config)
                config["training"]["episodes"] = 1
                config["environment"]["episode_steps"] = 2
                config["training"]["batch_size"] = 2
                config["learned_replica"]["training"]["batch_size"] = 2
                if method == "drl-ra-learned-replica":
                    _, history = train_learned_replica_agent(config, 42, progress=False, metrics_only=True)
                else:
                    _, history = train_agent(config, method, 42, progress=False, metrics_only=True)
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["tasks"], 2.0)
                self.assertTrue(all(key in history[0] for key in METRICS_ONLY_FIELDS))


if __name__ == "__main__":
    unittest.main()
