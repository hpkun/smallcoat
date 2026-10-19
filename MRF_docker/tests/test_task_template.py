import copy
import json
from pathlib import Path
import unittest

from task_repr import ValidationError, build_template, represent_task, restore_task


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "isac_joint_task.json"
EXTENSIONS = {
    "CNI": {"data_rate_mbps": 10, "modulation_coding": "example_mcs",
            "link_quality": 0.8, "payload_mib": 1},
    "Radar": {"pri_us": 100, "pulse_bandwidth_mhz": 10,
              "fft_size": 1024, "mode": "search"},
    "EW": {"threat_level": 3, "response_limit_ms": 20,
           "intercept_bandwidth_mhz": 20, "jamming_mode": "example_mode"},
    "ISAC": {"communication_rate_mbps": 10, "sensing_resolution_m": 2,
             "shared_component_ratio": 0.5},
}


def example_for(family):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["task_id"] = f"example-{family.lower()}"
    raw["semantics"]["family"] = family
    raw["semantics"]["extension"] = EXTENSIONS[family]
    keep = {
        "CNI": {"rf-1", "ddc-1", "demod-1"},
        "Radar": {"rf-1", "ddc-1", "detect-1"},
        "EW": {"rf-1", "ddc-1", "detect-1"},
        "ISAC": {"rf-1", "ddc-1", "detect-1", "demod-1"},
    }[family]
    raw["graph"]["nodes"] = [node for node in raw["graph"]["nodes"] if node["id"] in keep]
    raw["graph"]["edges"] = [edge for edge in raw["graph"]["edges"]
                              if edge["source"] in keep and edge["target"] in keep]
    if family == "EW":
        threat = raw["graph"]["nodes"][-1]
        threat["role"] = "threat_detect"
        threat["capability"] = "threat_detection"
    if family == "Radar":
        raw["graph"]["nodes"][1]["execution_profiles"]["DSP"]["runtime_ms"] = 2
    return raw


class TaskTemplateTests(unittest.TestCase):
    def test_four_extensions_and_per_device_costs(self):
        for family in EXTENSIONS:
            with self.subTest(family=family):
                task = represent_task(example_for(family))
                self.assertEqual(task.schema_version, 2)
                self.assertEqual(task.S.extension, EXTENSIONS[family])
                self.assertEqual(task.Q.scheduling.reconfiguration_ms, 3)
                self.assertEqual(task.R.rf_channels, 1)
                self.assertEqual(task.X.remaining_work, 100)
                ddc = task.G.nodes[1]
                self.assertGreater(ddc.execution_profiles["CPU"].runtime_ms,
                                   ddc.execution_profiles["DSP"].runtime_ms)

    def test_extracts_shared_dag_and_restores_every_variant(self):
        tasks = [represent_task(example_for(family)) for family in EXTENSIONS]
        template = build_template(tasks)
        self.assertEqual([node.role for node in template.common_nodes], ["ddc", "rf_receive"])
        self.assertEqual(len(template.common_edges), 1)
        self.assertEqual(template.common_edges[0].source_role, "rf_receive")
        self.assertEqual(len(template.variants["example-isac"].branch_nodes), 2)
        self.assertEqual(len(template.variants["example-isac"].branch_edges), 2)
        self.assertEqual(len(template.variants["example-radar"].branch_nodes), 1)
        self.assertIn("communication_rate_mbps", template.to_json())
        for original in tasks:
            with self.subTest(task=original.task_id):
                restored = restore_task(template, original.task_id)
                self.assertEqual((restored.S, restored.Q, restored.R, restored.X),
                                 (original.S, original.Q, original.R, original.X))
                self.assertEqual({node.node_id: node for node in restored.G.nodes},
                                 {node.node_id: node for node in original.G.nodes})
                self.assertEqual(set(restored.G.edges), set(original.G.edges))
                self.assertEqual(restored.to_dict(), original.to_dict())

    def test_rejects_missing_and_unknown_type_fields(self):
        missing = example_for("Radar")
        del missing["semantics"]["extension"]["fft_size"]
        with self.assertRaisesRegex(ValidationError, "fft_size"):
            represent_task(missing)
        unknown = example_for("EW")
        unknown["semantics"]["extension"]["radar_only"] = 1
        with self.assertRaisesRegex(ValidationError, "radar_only"):
            represent_task(unknown)

    def test_rejects_invalid_profiles_and_roles(self):
        raw = example_for("CNI")
        raw["graph"]["nodes"][1]["execution_profiles"]["DSP"]["resources"]["dsp_units"] = 0
        with self.assertRaisesRegex(ValidationError, "mapped device"):
            represent_task(raw)
        raw = example_for("CNI")
        raw["graph"]["nodes"][1]["role"] = "rf_receive"
        with self.assertRaisesRegex(ValidationError, "roles must be unique"):
            represent_task(raw)

    def test_rejects_conflicting_role_contract(self):
        first = represent_task(example_for("CNI"))
        raw = copy.deepcopy(example_for("Radar"))
        raw["graph"]["nodes"][1]["capability"] = "different_ddc"
        second = represent_task(raw)
        with self.assertRaisesRegex(ValidationError, "conflicting"):
            build_template([first, second])

    def test_v2_rejects_new_unknown_fields(self):
        raw = example_for("ISAC")
        raw["qos"]["unmodelled_metric"] = 1
        with self.assertRaisesRegex(ValidationError, "unmodelled_metric"):
            represent_task(raw)


if __name__ == "__main__":
    unittest.main()
