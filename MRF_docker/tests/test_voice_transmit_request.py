import copy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from task_repr import (ValidationError, build_template, parse_external_task,
                       represent_external_task, restore_task)


ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "examples" / "BusinessRequests" / "voice_transmit_request.json"
DATA_LINK = ROOT / "examples" / "BusinessRequests" / "External" / "02_data_link_transfer.json"
COLLECTION = ROOT / "examples" / "BusinessRequests" / "External" / "all_tasks.json"


class VoiceTransmitRequestTests(TestCase):
    def setUp(self):
        self.request = json.loads(REQUEST.read_text(encoding="utf-8"))

    def test_voice_transmit_uses_external_abstract_graph(self):
        result = represent_external_task(self.request)
        task = result["task_representation"]
        self.assertEqual(task["schema_version"], "rf_task_representation.v1")
        self.assertEqual(result["status"], "abstract_only_requires_refinement")
        self.assertEqual(task["S"]["task_type"], "voice_communication")
        self.assertEqual(task["S"]["business_parameters"]["audio_source"]["reference"],
                         "demo-voice-buffer-001")
        self.assertEqual(task["Q"]["deadline_ms"], 100)
        self.assertEqual(task["X"]["remaining_deadline_ms"], 100)
        self.assertEqual([node["role"] for node in task["G"]["nodes"]],
                         ["accept_voice_service_data", "prepare_voice_payload",
                          "send_voice_over_radio"])
        self.assertEqual(len(task["G"]["edges"]), 2)
        self.assertTrue(all(value is None for value in
                            task["R"]["required_resources"].values()))
        self.assertEqual(task["G"]["component_bindings"], [])
        self.assertTrue(all("capability" not in node for node in task["G"]["nodes"]))

    def test_voice_and_data_link_restore_from_one_common_template(self):
        data_link = json.loads(DATA_LINK.read_text(encoding="utf-8"))
        tasks = [parse_external_task(raw) for raw in (self.request, data_link)]
        template = build_template(tasks)
        self.assertEqual(len(template.common_nodes), 3)
        self.assertEqual(len(template.common_edges), 2)
        for task in tasks:
            self.assertEqual(restore_task(template, task["task_id"]), task)
        collection = json.loads(COLLECTION.read_text(encoding="utf-8"))
        self.assertEqual([task for task in collection["tasks"]
                          if task["task_id"] == self.request["task_id"]], [self.request])

    def test_invalid_external_voice_request(self):
        bad = copy.deepcopy(self.request)
        bad["direction"] = "invalid"
        with self.assertRaises(ValidationError):
            represent_external_task(bad)
        bad = copy.deepcopy(self.request)
        bad["deadline_ms"] = bad["arrival_ms"]
        with self.assertRaises(ValidationError):
            represent_external_task(bad)
        bad = copy.deepcopy(self.request)
        bad["request"]["audio_source"]["sample_rate_hz"] = 0
        with self.assertRaisesRegex(ValidationError, "sample_rate_hz"):
            represent_external_task(bad)
        bad = copy.deepcopy(self.request)
        bad["request"]["communication"]["link_quality"] = 1.5
        with self.assertRaisesRegex(ValidationError, "link_quality"):
            represent_external_task(bad)

    def test_http_uses_unified_external_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app

        client = TestClient(create_app())
        with patch("backend.services.tasks.match_task", side_effect=AssertionError("matched")):
            response = client.post("/api/v1/task-representations/external",
                                   json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_representation"]["task_id"],
                         "voice-tx-demo-001")
        self.assertEqual(client.post("/api/v1/task-representations/voice-transmit",
                                     json=self.request).status_code, 404)
