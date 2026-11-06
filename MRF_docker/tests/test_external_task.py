import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from task_repr import (ValidationError, build_template, parse_external_task,
                       represent_external_task, represent_external_tasks,
                       restore_task)


INPUTS = Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests" / "External"


def example(name):
    return json.loads((INPUTS / name).read_text(encoding="utf-8"))


class ExternalTaskTests(unittest.TestCase):
    def test_six_external_tasks_keep_business_constraints(self):
        for source in sorted(INPUTS.glob("0*.json")):
            with self.subTest(task=source.name):
                raw = json.loads(source.read_text(encoding="utf-8"))
                normalized = parse_external_task(raw)
                result = represent_external_task(raw)
                self.assertEqual(result["task_representation"], normalized)
                self.assertEqual(result["status"], "abstract_only_requires_refinement")
                self.assertEqual(result["task_template"]["abstraction_level"],
                                 "business_function")
                self.assertEqual(normalized["S"]["task_type"], raw["task_type"])
                self.assertEqual(normalized["S"]["base_priority"], raw["priority"])
                self.assertEqual(normalized["Q"]["duration_ms"], raw["duration_ms"])
                self.assertEqual(normalized["Q"]["deadline_ms"], None)
                self.assertTrue(all(value is None for value in
                                    normalized["R"]["required_resources"].values()))
                self.assertEqual(normalized["G"]["component_bindings"], [])
                self.assertTrue(all(node["canonical_role"] for node in
                                    normalized["G"]["nodes"]))

    def test_template_shares_tx_roles_and_restores_bidirectional_variant(self):
        two_way = example("01_voice_communication.json")
        tx_only = copy.deepcopy(two_way)
        tx_only["task_id"] = "RF-COM-VOICE-TX"
        tx_only["direction"] = "tx"
        tx_only["performance"]["max_mouth_to_ear_latency_ms"] = 250
        tasks = [parse_external_task(item) for item in (two_way, tx_only)]
        template = build_template(tasks)
        self.assertEqual(len(template.common_nodes), 3)
        self.assertEqual(len(template.common_edges), 2)
        self.assertEqual(len(template.variants[two_way["task_id"]].branch_nodes), 3)
        for task in tasks:
            self.assertEqual(restore_task(template, task["task_id"]), task)
        batch = represent_external_tasks([two_way, tx_only])
        self.assertEqual(len(batch["task_templates"]), 1)
        self.assertEqual(len(batch["task_templates"]["CNI"]["common_nodes"]), 3)
        self.assertNotEqual(represent_external_task(two_way)["template_id"],
                            represent_external_task(tx_only)["template_id"])

    def test_cross_type_common_dag_restores_original_functions(self):
        voice = parse_external_task(example("01_voice_communication.json"))
        data_link = parse_external_task(example("02_data_link_transfer.json"))
        template = build_template([voice, data_link])
        self.assertEqual(template.template_id,
                         build_template([data_link, voice]).template_id)
        self.assertEqual([node["canonical_role"] for node in template.common_nodes],
                         ["accept_outbound_service_data", "prepare_outbound_payload",
                          "send_outbound_radio_payload"])
        self.assertEqual([(edge["source_role"], edge["target_role"])
                          for edge in template.common_edges],
                         [("accept_outbound_service_data", "prepare_outbound_payload"),
                          ("prepare_outbound_payload", "send_outbound_radio_payload")])
        self.assertEqual(len(template.variants[voice["task_id"]].branch_nodes), 3)
        for original in (voice, data_link):
            self.assertEqual(restore_task(template, original["task_id"]), original)

    def test_existing_normalized_nodes_without_canonical_role(self):
        task = parse_external_task(example("03_target_search.json"))
        for node in task["G"]["nodes"]:
            del node["canonical_role"]
        template = build_template([task])
        self.assertEqual(restore_task(template, task["task_id"]), task)

    def test_batch_collection_and_distinct_time_meanings(self):
        result = represent_external_tasks(example("all_tasks.json"))
        self.assertEqual(len(result["task_representations"]), 7)
        self.assertEqual(set(result["task_templates"]), {"CNI", "Radar", "EW"})
        self.assertEqual(len(result["task_templates"]["CNI"]["common_nodes"]), 3)
        self.assertEqual(result["task_templates"]["Radar"]["common_nodes"], ())
        for task in result["task_representations"]:
            family = task["S"]["family"]
            template = build_template([item for item in result["task_representations"]
                                       if item["S"]["family"] == family])
            self.assertEqual(restore_task(template, task["task_id"]), task)
        self.assertEqual(result["source_collection"]["collection_version"],
                         "external_rf_tasks_collection.v1")
        link = result["task_representations"][1]
        self.assertEqual(link["Q"]["duration_ms"], 30000)
        self.assertIsNone(link["Q"]["deadline_ms"])
        self.assertEqual(link["Q"]["quality_constraints"][0]["value"], 50)
        self.assertEqual(link["Q"]["quality_constraints"][0]["scope"], "message")
        self.assertEqual(link["Q"]["periodic_requirements"][0]["interval_ms"], 50)
        self.assertIsNone(link["X"]["remaining_deadline_ms"])

        raw = example("03_target_search.json")
        raw["deadline_ms"] = 1500
        task = parse_external_task(raw)
        self.assertEqual(task["Q"]["deadline_ms"], 1500)
        self.assertEqual(task["X"]["remaining_deadline_ms"], 500)
        self.assertEqual(task["Q"]["quality_constraints"][2]["scope"], "first_report")

    def test_rejects_invalid_requests_and_mixed_template_stages(self):
        raw = example("02_data_link_transfer.json")
        raw["performance"]["max_one_way_delivery_latency_ms"] = -1
        with self.assertRaises(ValidationError):
            parse_external_task(raw)
        with self.assertRaises(ValidationError):
            represent_external_tasks([])
        with self.assertRaises(ValidationError):
            build_template([parse_external_task(example("01_voice_communication.json")),
                            parse_external_task(example("03_target_search.json"))])

    def test_http_endpoint_returns_abstract_template_without_matching(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app

        client = TestClient(create_app())
        with patch("backend.services.tasks.match_task", side_effect=AssertionError("matched")):
            response = client.post("/api/v1/task-representations/external",
                                   json=example("02_data_link_transfer.json"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_representation"]["task_id"],
                         "RF-COM-DATALINK-001")
        match_response = client.post("/api/v1/matches",
                                     json=response.json()["task_representation"])
        self.assertEqual(match_response.status_code, 200)
        self.assertEqual(match_response.json()["status"], "NEEDS_REFINEMENT")
        collection = client.post("/api/v1/task-representations/external",
                                 json=example("all_tasks.json"))
        self.assertEqual(collection.status_code, 200)
        self.assertEqual(len(collection.json()["task_representations"]), 7)
        bad = example("02_data_link_transfer.json")
        bad["duration_ms"] = 0
        self.assertEqual(client.post("/api/v1/task-representations/external",
                                     json=bad).status_code, 422)


if __name__ == "__main__":
    unittest.main()
