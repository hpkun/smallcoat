import copy
import json
from pathlib import Path
import unittest

from task_repr import match_task
from task_repr.catalog import CapabilityMap, CapabilityMapping, default_capability_map


TASKS = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"


def sample():
    return json.loads((TASKS / "signal_analysis_task.json").read_text(encoding="utf-8"))


class StaticMatcherTests(unittest.TestCase):
    def test_complete_signal_chain(self):
        result = match_task(TASKS / "signal_analysis_task.json")
        self.assertEqual(result.status, "MATCHED")
        self.assertEqual({node: binding.cid for node, binding in result.bindings.items()},
                         {"ddc-node": "C03", "filter-node": "C05", "sensing-node": "C27"})
        self.assertEqual(result.bindings["ddc-node"].parameters["decimation"], 2)
        self.assertEqual(result.edges, (("ddc-node", "filter-node"),
                                        ("filter-node", "sensing-node")))
        self.assertEqual(result.to_dict()["unresolved"], [])

    def test_existing_tasks_explain_partial_matches(self):
        radar = match_task(TASKS / "radar_search_task.json")
        self.assertEqual(radar.status, "PARTIAL")
        issues = {(issue.node_id, issue.cid, issue.code) for issue in radar.unresolved}
        self.assertIn(("radar-ddc", "C03", "NEEDS_PARAMETERS"), issues)
        self.assertIn(("pulse-compress", "C21", "NEEDS_PARAMETERS"), issues)
        self.assertIn(("doppler-mtd", None, "UNSUPPORTED"), issues)
        self.assertIn(("cfar-detect", "C23", "INTERFACE_MISMATCH"), issues)

        voice = match_task(TASKS / "cni_tactical_voice_task.json")
        self.assertEqual(voice.bindings["voice-demod"].cid, "C14")
        self.assertEqual(voice.bindings["voice-demod"].parameters["decision_mode"], "soft")
        self.assertTrue(any(issue.node_id == "voice-sync" and issue.code == "NEEDS_PARAMETERS"
                            for issue in voice.unresolved))

        video = match_task(TASKS / "cni_video_backhaul_task.json")
        self.assertEqual(video.status, "PARTIAL")
        self.assertNotIn("video-fft", video.bindings)
        self.assertTrue(any(issue.node_id == "video-ldpc" and issue.code == "UNSUPPORTED"
                            for issue in video.unresolved))

    def test_parameter_and_unit_checks(self):
        raw = sample()
        raw["graph"]["nodes"][0]["parameters"]["nco_frequency"] = 2_000_000
        result = match_task(raw)
        self.assertTrue(any(issue.node_id == "ddc-node" and "Nyquist" in issue.detail
                            for issue in result.unresolved))

        raw = sample()
        del raw["graph"]["nodes"][0]["parameters"]["nco_frequency"]
        result = match_task(raw)
        self.assertTrue(any(issue.node_id == "ddc-node" and issue.code == "NEEDS_PARAMETERS"
                            for issue in result.unresolved))

        raw = sample()
        parameters = raw["graph"]["nodes"][0]["parameters"]
        del parameters["nco_frequency"]
        parameters["nco_frequency_mhz"] = 0.1
        result = match_task(raw)
        self.assertEqual(result.status, "MATCHED")
        self.assertEqual(result.bindings["ddc-node"].parameters["nco_frequency"], 100000)

    def test_cfar_alias_and_invalid_probability(self):
        raw = sample()
        raw["graph"]["nodes"] = [{
            "id": "detect", "capability": "cfar_detection",
            "inputs": {"power": "radar/power_map"},
            "outputs": {"detections": "radar/detections"},
            "estimated_runtime_ms": 1,
            "parameters": {"method": "CA-CFAR", "false_alarm_probability": 1e-6},
        }]
        raw["graph"]["edges"] = []
        result = match_task(raw)
        self.assertEqual(result.status, "MATCHED")
        self.assertEqual(result.bindings["detect"].parameters["pfa"], 1e-6)
        raw["graph"]["nodes"][0]["parameters"]["false_alarm_probability"] = 1.5
        result = match_task(raw)
        self.assertEqual(result.unresolved[0].code, "INVALID_PARAMETERS")

    def test_edge_rejects_rate_and_length_mismatch(self):
        raw = sample()
        raw["graph"]["nodes"][1]["parameters"]["sample_rate_mhz"] = 3
        result = match_task(raw)
        self.assertEqual(result.status, "PARTIAL")
        self.assertNotIn(("ddc-node", "filter-node"), result.edges)
        self.assertTrue(any(issue.edge == ("ddc-node", "filter-node") and "sample rate" in issue.detail
                            for issue in result.unresolved))

        raw = sample()
        raw["graph"]["nodes"][1]["parameters"]["input_block_length"] = 1024
        raw["graph"]["nodes"][1]["parameters"]["output_block_length"] = 1024
        result = match_task(raw)
        self.assertTrue(any(issue.edge == ("ddc-node", "filter-node") and "block length" in issue.detail
                            for issue in result.unresolved))

    def test_candidate_search_skips_infeasible_first_entry(self):
        raw = sample()
        base = default_capability_map()
        mappings = (CapabilityMapping("C05", ("digital_down_conversion",), "test alternative"),) + base.mappings
        result = match_task(raw, CapabilityMap(mappings))
        self.assertEqual(result.status, "MATCHED")
        self.assertEqual(result.bindings["ddc-node"].cid, "C03")

    def test_no_ability_is_unsupported(self):
        raw = sample()
        raw["graph"]["nodes"] = [copy.deepcopy(raw["graph"]["nodes"][0])]
        raw["graph"]["nodes"][0]["capability"] = "nonexistent_capability"
        raw["graph"]["edges"] = []
        self.assertEqual(match_task(raw).status, "UNSUPPORTED")

    def test_each_runtime_component_has_a_valid_static_contract(self):
        cases = [
            ("digital_down_conversion", "C03", "iq/complex", "iq/baseband",
             {"decimation": 2, "nco_frequency": 1000}),
            ("fir_filter", "C05", "iq/baseband", "iq/baseband", {"cutoff": 0.25}),
            ("known_preamble_frame_synchronization", "C07", "iq/complex", "iq/baseband",
             {"preamble": "[1,1,-1,1]", "input_block_length": 16}),
            ("fft", "C08", "iq/complex", "frequency/complex",
             {"fft_size": 16, "input_block_length": 16}),
            ("wideband_digital_channelizer", "C09", "iq/wideband", "ew/channelized_iq",
             {"channel_count": 4, "decimation": 4}),
            ("qpsk_demodulation", "C14", "cni/symbols", "data/soft_bits",
             {"modulation": "QPSK"}),
            ("ofdm_tx", "C15", "cni/symbols", "iq/complex",
             {"fft_length": 8, "cp_length": 2, "subcarrier_allocation": "[1,2,3,4,5,6,7]",
              "pilot_pattern": "[]", "guard_carriers": "[0]"}),
            ("convolutional_fec_decoding", "C17", "data/soft_bits", "data/decoded_bits",
             {"block_length": 8, "constraint_length": 3, "generators": "[7,5]"}),
            ("lfm_pulse_compression", "C21", "iq/baseband", "radar/range_profile",
             {"reference_waveform": "[\"1+0j\",\"0+1j\",\"-1+0j\",\"0-1j\"]",
              "input_block_length": 8}),
            ("cfar_detection", "C23", "radar/power_map", "radar/detections",
             {"method": "CA-CFAR", "false_alarm_probability": 1e-3}),
            ("fast_spectrum_sensing", "C27", "iq/complex", "spectrum/psd",
             {"fft_length": 16, "overlap": 0.5, "input_block_length": 32}),
            ("iq_delay_replay", "C32", "iq/complex", "iq/baseband",
             {"delay_samples": 2}),
        ]
        for capability, cid, input_format, output_format, parameters in cases:
            with self.subTest(cid=cid):
                raw = sample()
                raw["graph"]["nodes"] = [{
                    "id": "unit", "capability": capability,
                    "inputs": {"in": input_format}, "outputs": {"out": output_format},
                    "estimated_runtime_ms": 1, "parameters": parameters,
                }]
                raw["graph"]["edges"] = []
                result = match_task(raw)
                self.assertEqual(result.status, "MATCHED", result.to_dict())
                self.assertEqual(result.bindings["unit"].cid, cid)


if __name__ == "__main__":
    unittest.main()
