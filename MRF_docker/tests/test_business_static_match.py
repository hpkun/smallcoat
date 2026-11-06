import json
from pathlib import Path
import unittest

from matching import match_task
from task_repr import build_template, parse_external_task, restore_task


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests" / "External"


def example(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class BusinessStaticMatchTests(unittest.TestCase):
    def test_external_request_expands_boundaries_and_composite_role(self):
        result = match_task(example("07_spectrum_analysis.json"))
        self.assertEqual(result.status, "MATCHED", result.to_dict())
        self.assertEqual({node: binding.cid for node, binding in result.bindings.items()},
                         {"main-02:ddc": "C03", "main-03:filter": "C05",
                          "main-03:sensing": "C27"})
        self.assertEqual(result.edges, (("main-03:filter", "main-03:sensing"),
                                        ("main-02:ddc", "main-03:filter")))
        graph = result.to_dict()["refined_graph"]
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(graph["nodes"][1]["business_node_id"], "main-03")
        self.assertEqual(len(graph["boundaries"]), 2)
        self.assertEqual(result.to_dict()["match_scope"], "component_dag")
        self.assertEqual(result.unresolved, ())

    def test_normalized_and_restored_task_use_same_matcher(self):
        normalized = parse_external_task(example("07_spectrum_analysis.json"))
        template = build_template([normalized])
        restored = restore_task(template, normalized["task_id"])
        self.assertEqual(match_task(normalized).to_dict(), match_task(restored).to_dict())

    def test_missing_configuration_does_not_bind_components(self):
        raw = example("07_spectrum_analysis.json")
        del raw["request"]["processing"]["fft_length"]
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.bindings, {})
        self.assertTrue(any("processing.fft_length" in issue.detail for issue in result.unresolved))

    def test_unknown_processing_profile_requires_refinement(self):
        raw = example("07_spectrum_analysis.json")
        raw["request"]["processing"]["profile"] = "unknown"
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.bindings, {})
        self.assertIn("processing.profile", result.unresolved[0].detail)

    def test_input_format_mismatch_does_not_bind_components(self):
        raw = example("07_spectrum_analysis.json")
        raw["data_input"]["format"] = "iq/baseband"
        result = match_task(raw)
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertTrue(any(issue.code == "INTERFACE_MISMATCH" for issue in result.unresolved))

    def test_uncovered_voice_request_is_not_reported_as_matched(self):
        result = match_task(example("01_voice_communication.json"))
        self.assertEqual(result.status, "NEEDS_REFINEMENT")
        self.assertEqual(result.bindings, {})
        self.assertIn("voice_codec_and_payload_format", result.unresolved[0].detail)

    def test_modified_business_graph_is_rejected(self):
        raw = parse_external_task(example("07_spectrum_analysis.json"))
        raw["G"]["edges"].pop()
        with self.assertRaisesRegex(ValueError, "business graph differs"):
            match_task(raw)

    def test_http_match_uses_existing_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app

        client = TestClient(create_app())
        response = client.post("/api/v1/matches", json=example("07_spectrum_analysis.json"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "MATCHED")
        self.assertEqual(response.json()["match_scope"], "component_dag")


if __name__ == "__main__":
    unittest.main()
