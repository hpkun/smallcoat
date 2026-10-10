from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import train
from drl_ra.config import load_config
from drl_ra.environment import SAGINEnv
from drl_ra.experiment import build_learned_replica_agent, train_learned_replica_agent


class BCFineTuneTests(unittest.TestCase):
    def test_training_auto_device_selects_cuda_or_cpu(self):
        for cuda_available, expected in ((True, "cuda"), (False, "cpu")):
            with self.subTest(cuda_available=cuda_available), \
                 patch("sys.argv", ["train.py", "--config", "configs/replica_bc_finetune.yaml",
                                    "--method", "drl-ra-learned-replica", "--device", "auto"]), \
                 patch("torch.cuda.is_available", return_value=cuda_available), \
                 patch("train.train_learned_replica_agent") as trainer, \
                 patch("train.write_json"), patch("builtins.print"):
                trainer.return_value = (unittest.mock.MagicMock(), [])
                train.main()
                self.assertEqual(trainer.call_args.kwargs["device"], expected)

    def test_finetuning_preserves_greedy_primary_and_updates_replica(self):
        config = load_config("configs/replica_bc_finetune.yaml")
        self.assertGreaterEqual(config["learned_replica"]["freeze_primary_episodes"], config["training"]["episodes"])
        config["training"].update(episodes=3, batch_size=2, hidden_sizes=[16])
        config["environment"]["episode_steps"] = 4
        config["learned_replica"]["training"].update(batch_size=2, hidden_sizes=[16])
        initial_config = deepcopy(config)
        initial_config["learned_replica"]["primary_checkpoint"] = None
        initial_config["learned_replica"]["bc_checkpoint"] = None
        env = SAGINEnv(initial_config, seed=42)
        source = build_learned_replica_agent(env, initial_config, 42, "cpu")
        source.primary.lagrange = 2.5
        with TemporaryDirectory() as directory:
            primary_path = Path(directory) / "primary.pt"
            bc_path = Path(directory) / "bc.pt"
            source.primary.save(primary_path)
            source.replica.save(bc_path, {"artifact": "replica_behavior_cloning"})
            config["learned_replica"]["primary_checkpoint"] = str(primary_path)
            config["learned_replica"]["bc_checkpoint"] = str(bc_path)
            agent = build_learned_replica_agent(env, config, 42, "cpu")
            replica_before = deepcopy(agent.replica.online.state_dict())
            with patch("drl_ra.experiment.build_learned_replica_agent", return_value=agent), \
                 patch.object(agent.primary, "act", wraps=agent.primary.act) as primary_act:
                _, history = train_learned_replica_agent(config, 42, progress=False, metrics_only=True)

        self.assertEqual(primary_act.call_count, 12)
        self.assertTrue(all(call.kwargs["epsilon"] == 0.0 for call in primary_act.call_args_list))
        self.assertEqual([row["epsilon"] for row in history], [0.0] * 3)
        for index, row in enumerate(history):
            self.assertAlmostEqual(row["replica_epsilon"], 0.05 * 0.98 ** index)
        for name in ("online", "target"):
            expected = getattr(source.primary, name).state_dict()
            actual = getattr(agent.primary, name).state_dict()
            self.assertTrue(all(torch.equal(expected[key], actual[key]) for key in expected))
        self.assertEqual(agent.primary.lagrange, 2.5)
        self.assertEqual(agent.primary.training_steps, 0)
        self.assertEqual(agent.replica.optimizer.param_groups[0]["lr"], 1e-5)
        self.assertGreater(agent.replica.training_steps, 0)
        self.assertTrue(any(not torch.equal(value, agent.replica.online.state_dict()[key])
                            for key, value in replica_before.items()))


if __name__ == "__main__":
    unittest.main()
