import copy
import json
from pathlib import Path
import unittest

from task_repr import ResourceLevels, ValidationError, represent_task


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos" / "voice_tx_task.json"
V2_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos" / "isac_joint_task.json"


class TaskRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_preserves_scheduler_inputs_and_excludes_raw_artifacts(self):
        rep = represent_task(self.raw)
        encoded = rep.to_dict()
        self.assertEqual(encoded["S"]["base_priority"], 5)
        self.assertEqual(encoded["Q"]["deadline_ms"], 100)
        self.assertEqual(encoded["Q"]["quality_constraints"][0]["metric"], "ber")
        self.assertEqual(encoded["R"]["dsp_units"], 1)
        self.assertEqual(len(encoded["G"]["nodes"]), 3)
        self.assertEqual(len(encoded["G"]["edges"]), 2)
        self.assertEqual(encoded["G"]["nodes"][1]["parameters"]["codec_profile"], "example_profile")
        self.assertEqual(encoded["X"]["remaining_deadline_ms"], 90)
        self.assertNotIn("raw_artifacts", rep.to_json())

    def test_rejects_unknown_field_instead_of_silently_discarding_it(self):
        self.raw["qos"]["max_ber"] = 1e-5
        with self.assertRaisesRegex(ValidationError, "max_ber"):
            represent_task(self.raw)

    def test_rejects_incompatible_port_formats(self):
        self.raw["graph"]["nodes"][1]["inputs"]["audio"] = "iq/samples"
        with self.assertRaisesRegex(ValidationError, "incompatible"):
            represent_task(self.raw)

    def test_rejects_nonfinite_quality_constraint(self):
        self.raw["qos"]["quality_constraints"][0]["value"] = float("nan")
        with self.assertRaisesRegex(ValidationError, "finite"):
            represent_task(self.raw)

    def test_rejects_dependency_cycle(self):
        graph = self.raw["graph"]
        graph["nodes"][0]["inputs"]["feedback"] = "coded/symbols"
        graph["edges"].append({
            "source": "channel", "source_port": "symbols",
            "target": "source", "target_port": "feedback"
        })
        with self.assertRaisesRegex(ValidationError, "cycle"):
            represent_task(self.raw)

    def test_rejects_invalid_deadline_and_priority(self):
        late = copy.deepcopy(self.raw)
        late["qos"]["deadline_ms"] = 0
        with self.assertRaises(ValidationError):
            represent_task(late)
        self.raw["state"]["current_priority"] = -1
        with self.assertRaises(ValidationError):
            represent_task(self.raw)

    def test_accepts_first_version_simulation_calibration(self):
        raw = json.loads(V2_EXAMPLE.read_text(encoding="utf-8"))
        raw["graph"]["nodes"][0]["simulation_profile"] = {
            "resource_levels": {"cpu": 2, "mem": 3, "dsp": 4, "fpga": 5, "bw": 4},
            "latency_level": "L1",
            "sharing_class": "P",
            "preemption_class": "C",
        }
        node = represent_task(raw).G.nodes[0]
        self.assertEqual(node.simulation_profile.resource_levels.cpu, 2)
        self.assertEqual(node.simulation_profile.resource_levels.to_normalized_units()["cpu"], 10)
        self.assertEqual(node.simulation_profile.sharing_class, "P")

    def test_rejects_invalid_simulation_calibration(self):
        raw = json.loads(V2_EXAMPLE.read_text(encoding="utf-8"))
        raw["graph"]["nodes"][0]["simulation_profile"] = {
            "resource_levels": {"cpu": 6, "mem": 3, "dsp": 4, "fpga": 5, "bw": 4},
            "latency_level": "L1",
            "sharing_class": "P",
            "preemption_class": "C",
        }
        with self.assertRaisesRegex(ValidationError, "must be <= 5"):
            represent_task(raw)


if __name__ == "__main__":
    unittest.main()
