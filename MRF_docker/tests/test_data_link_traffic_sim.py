from pathlib import Path
import unittest

from examples.data_link_traffic_sim import parse_scenario, simulate
from examples.data_link_representation_demo import adapt_scenario
from task_repr import represent_task


SCENARIO = Path(__file__).resolve().parents[1] / "examples" / "TrafficScenarios" / "data_link_2s.mgn"


class DataLinkTrafficSimulationTests(unittest.TestCase):
    def test_two_second_flow_and_finite_burst(self):
        report = simulate(SCENARIO.read_text(encoding="utf-8"))
        self.assertEqual(report["simulation"], "packet_schedule_only")
        self.assertEqual([flow["packets_planned"] for flow in report["flows"]], [100, 8])
        self.assertEqual([flow["bytes_planned"] for flow in report["flows"]], [25600, 4096])
        self.assertEqual([flow["last_packet_s"] for flow in report["flows"]], [1.98, 0.85])
        self.assertEqual([flow["end_reason"] for flow in report["flows"]],
                         ["off_event", "count_limit"])
        self.assertEqual(report["totals"], {"packets_planned": 108, "bytes_planned": 29696})
        self.assertEqual(report["packets"][0]["time_s"], 0)
        self.assertEqual(report["packets"][0]["sequence"], 1)
        self.assertEqual(report["packets"][-1]["time_s"], 1.98)
        burst = [packet for packet in report["packets"] if packet["flow_id"] == 2]
        self.assertEqual([packet["time_s"] for packet in burst],
                         [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
        self.assertEqual(report["flow_events"][0],
                         {"time_s": 0.0, "event": "LISTEN", "port": 5000})
        self.assertEqual(report["flow_events"][-1],
                         {"time_s": 2.0, "event": "IGNORE", "port": 5000})

    def test_invalid_event_order_or_unclosed_flow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "time order"):
            parse_scenario("1.0 LISTEN UDP 5000\n0.0 IGNORE UDP 5000")
        with self.assertRaisesRegex(ValueError, "close every flow"):
            parse_scenario("0.0 ON 1 UDP SRC 5001 DST 127.0.0.1/5000 PERIODIC [1 256]")
        with self.assertRaisesRegex(ValueError, "unsupported MGEN event"):
            parse_scenario("0.0 JOIN 224.225.1.2")

    def test_external_flow_facts_survive_task_representation(self):
        raw = adapt_scenario(SCENARIO.read_text(encoding="utf-8"))
        task = represent_task(raw)
        nodes = {node.node_id: node for node in task.G.nodes}
        self.assertEqual(task.schema_version, 1)
        self.assertEqual(task.Q.deadline_ms, 2000)
        self.assertIsNone(task.Q.max_latency_ms)
        self.assertEqual(len(task.G.edges), 2)
        self.assertEqual(nodes["packet_source_1"].parameters["packets_per_s"], 50)
        self.assertEqual(nodes["packet_source_1"].parameters["packet_bytes"], 256)
        self.assertEqual(nodes["packet_source_2"].parameters["start_ms"], 500)
        self.assertEqual(nodes["packet_source_2"].parameters["count_limit"], 8)
        self.assertEqual(nodes["udp_transmit_2"].parameters["destination_port"], 5000)


if __name__ == "__main__":
    unittest.main()
