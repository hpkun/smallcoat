"""验证业务细化档案的节点、接口和静态匹配边界。"""

import json
from pathlib import Path
import unittest

from examples.business_refinement_demo import configured_data_link_receive, configured_request
from matching import match_task
from task_repr import parse_external_task, refine_business_task


INPUTS = Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests" / "External"


def source(name):
    return json.loads((INPUTS / name).read_text(encoding="utf-8"))


class BusinessRefinementProfileTests(unittest.TestCase):
    def test_six_profiles_have_complete_specification_graphs(self):
        expected = {
            "01_voice_communication.json": ("codec2_ofdm_qpsk_conv", "PARTIAL"),
            "02_data_link_transfer.json": ("ofdm_qpsk_conv_tx", "PARTIAL"),
            "03_target_search.json": ("lfm_pulse_doppler_ca_cfar", "PARTIAL"),
            "04_target_tracking.json": ("radar_echo_gnn_ekf", "UNSUPPORTED"),
            "05_signal_interception.json": ("wideband_channelized_psd_intercept", "PARTIAL"),
            "06_signal_identification.json": ("iq_features_classifier", "PARTIAL"),
        }
        for name, (profile_id, status) in expected.items():
            with self.subTest(task=name):
                raw = configured_request(name)
                refinement = refine_business_task(parse_external_task(raw))
                self.assertIsNotNone(refinement.task, refinement.issues)
                result = match_task(refinement)
                self.assertEqual(result.status, status, result.to_dict())
                graph = result.to_dict()["refined_graph"]
                self.assertEqual(graph["profile_id"], profile_id)
                self.assertEqual(graph["profile_version"], "1.0")
                self.assertTrue(graph["nodes"] and graph["boundaries"])
                self.assertTrue(all(node["business_node_id"] for node in graph["nodes"]))
                self.assertNotEqual(result.status, "MATCHED")

    def test_business_outputs_are_not_silently_treated_as_matched(self):
        result = match_task(configured_request("02_data_link_transfer.json"))
        graph = result.to_dict()["refined_graph"]
        self.assertIn("tx-02:fragment", {node["id"] for node in graph["nodes"]})
        self.assertEqual(graph["covered_outputs"], [])
        self.assertEqual(graph["uncovered_outputs"],
                         ["delivered_message_records", "link_delivery_statistics"])
        self.assertEqual({issue.detail for issue in result.unresolved
                          if issue.code == "OUTPUT_NOT_COVERED"},
                         {"refinement profile does not produce delivered_message_records",
                          "refinement profile does not produce link_delivery_statistics"})
        self.assertEqual(match_task(source("07_spectrum_analysis.json")).status, "MATCHED")

    def test_frame_capacity_and_fragment_size_are_checked_before_matching(self):
        raw = configured_request("02_data_link_transfer.json")
        raw["request"]["processing"]["fragment_payload_bytes"] = 17
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.unresolved[0].code, "INTERFACE_MISMATCH")
        self.assertFalse(result.bindings)
        raw = configured_request("01_voice_communication.json")
        raw["request"]["processing"]["frame_header_bits"] = 8
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.unresolved[0].code, "INTERFACE_MISMATCH")

    def test_voice_traffic_payload_must_match_selected_codec_frame(self):
        raw = configured_request("01_voice_communication.json")
        raw["request"]["traffic"]["encoded_payload_bytes_per_frame"] = 60
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertIn("encoded payload bytes", result.unresolved[0].detail)
        self.assertFalse(result.bindings)

    def test_data_link_receive_validates_c15_c14_c17_subchain(self):
        result = match_task(configured_data_link_receive())
        self.assertEqual(result.status, "PARTIAL", result.to_dict())
        self.assertEqual({node: binding.cid for node, binding in result.bindings.items()},
                         {"rx-02:ofdm_rx": "C15", "rx-02:qpsk_demod": "C14",
                          "rx-02:fec_decode": "C17"})
        self.assertIn(("rx-02:ofdm_rx", "rx-02:qpsk_demod"), result.edges)
        self.assertIn(("rx-02:qpsk_demod", "rx-02:fec_decode"), result.edges)
        self.assertTrue(any(issue.node_id == "rx-02:frame_verify" and
                            issue.code == "UNSUPPORTED" for issue in result.unresolved))
        self.assertTrue(any(issue.node_id == "rx-02:reassemble" and
                            issue.code == "UNSUPPORTED" for issue in result.unresolved))
        self.assertEqual(len(result.to_dict()["refined_graph"]["prerequisites"]), 3)

    def test_missing_profile_and_parameters_are_distinct_from_unsupported(self):
        raw = source("03_target_search.json")
        self.assertEqual(match_task(raw).status, "NEEDS_REFINEMENT")
        raw = configured_request("03_target_search.json")
        del raw["request"]["processing"]["reference_waveform"]
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertFalse(result.bindings)
        self.assertTrue(any("reference_waveform" in issue.detail for issue in result.unresolved))

    def test_radar_profile_checks_requested_range_and_velocity(self):
        raw = configured_request("03_target_search.json")
        raw["request"]["processing"]["pulse_bandwidth_mhz"] = 5
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertIn("range resolution", result.unresolved[0].detail)
        raw = configured_request("03_target_search.json")
        raw["request"]["processing"]["carrier_frequency_mhz"] = 1000
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertIn("velocity span", result.unresolved[0].detail)

    def test_one_symbol_frame_contract_rejects_incompatible_bit_count(self):
        raw = configured_request("02_data_link_transfer.json")
        raw["request"]["processing"]["block_length"] = 63
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.unresolved[0].code, "INTERFACE_MISMATCH")
        self.assertFalse(result.bindings)

    def test_interception_keeps_multichannel_selection_gap(self):
        result = match_task(configured_request("05_signal_interception.json"))
        graph = result.to_dict()["refined_graph"]
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(by_id["main-02:channelize"]["outputs"]["out"], "ew/channelized_iq")
        self.assertEqual(by_id["main-02:select_channel"]["inputs"]["in"], "ew/channelized_iq")
        self.assertEqual(by_id["main-02:select_channel"]["outputs"]["out"], "iq/complex")
        self.assertIn("main-02:channelize", result.bindings)
        self.assertIn("main-02:psd", result.bindings)
        self.assertNotIn("main-02:select_channel", result.bindings)


if __name__ == "__main__":
    unittest.main()
