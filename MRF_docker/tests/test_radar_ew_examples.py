import json
from pathlib import Path
import unittest
import copy

from task_repr import represent_task


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"


class RadarAndElectronicWarfareExampleTests(unittest.TestCase):
    """验证 Radar/EW 示例保留统一调度模板中的关键属性。"""

    def load(self, filename):
        return json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))

    def test_radar_search_carries_pri_and_accelerator_requirements(self):
        task = represent_task(self.load("radar_search_task.json"))

        self.assertEqual(task.S.family, "Radar")
        self.assertEqual(task.S.extension["pri_us"], 1000)
        self.assertEqual(task.Q.period_ms, 1)
        self.assertEqual(task.Q.max_response_ms, 1)
        self.assertFalse(task.Q.scheduling.preemptible)
        self.assertEqual(task.R.fpga_units, 1)
        self.assertEqual(task.R.dsp_units, 2)
        self.assertEqual(
            [node.role for node in task.G.nodes],
            ["rf_receive", "ddc", "pulse_compression", "doppler_processing", "target_detection"],
        )
        self.assertEqual(task.G.nodes[3].parameters["coherent_pulses"], 64)
        self.assertTrue(any("FPGA" in node.execution_profiles for node in task.G.nodes))
        self.assertTrue(any("DSP" in node.execution_profiles for node in task.G.nodes))
        self.assertEqual(len(task.G.edges), len(task.G.nodes) - 1)

    def test_ew_response_carries_sub_ms_preemptive_hot_standby_path(self):
        task = represent_task(self.load("ew_threat_response_task.json"))

        self.assertEqual(task.S.family, "EW")
        self.assertEqual(task.S.base_priority, 10)
        self.assertEqual(task.S.extension["threat_level"], 5)
        self.assertEqual(task.Q.max_response_ms, 0.5)
        self.assertEqual(task.Q.quality_constraints[0].value, 0.5)
        self.assertFalse(task.Q.scheduling.preemptible)
        self.assertTrue(task.Q.scheduling.can_preempt)
        self.assertEqual(task.G.nodes[6].parameters["container_mode"], "hot_standby")
        self.assertTrue(task.G.nodes[7].parameters["waveform_preloaded"])
        self.assertEqual(task.R.bandwidth_mbps, 500)
        self.assertEqual(task.R.rf_bandwidth_mhz, 500)

        # 示例节点按拓扑序排列；每条边都应指向后续节点。
        node_index = {node.node_id: index for index, node in enumerate(task.G.nodes)}
        self.assertTrue(all(node_index[edge.source] < node_index[edge.target]
                            for edge in task.G.edges))

    def test_can_preempt_defaults_false_for_existing_v2_records(self):
        raw = self.load("radar_search_task.json")
        del raw["qos"]["scheduling"]["can_preempt"]
        self.assertFalse(represent_task(raw).Q.scheduling.can_preempt)

        invalid = copy.deepcopy(raw)
        invalid["qos"]["scheduling"]["can_preempt"] = "yes"
        with self.assertRaisesRegex(ValueError, "can_preempt must be a boolean"):
            represent_task(invalid)


if __name__ == "__main__":
    unittest.main()
