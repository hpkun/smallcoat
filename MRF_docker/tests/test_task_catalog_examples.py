import json
from pathlib import Path
import unittest

from task_repr import represent_task


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
V2_EXAMPLES = (
    "cni_tactical_voice_task.json",
    "cni_video_backhaul_task.json",
    "radar_search_task.json",
    "radar_iff_task.json",
    "radar_sar_task.json",
    "ew_esm_task.json",
    "ew_ecm_task.json",
    "ew_threat_response_task.json",
    "isac_joint_task.json",
)


class TaskCatalogExampleTests(unittest.TestCase):
    def test_all_v2_catalog_examples_validate(self):
        for filename in V2_EXAMPLES:
            with self.subTest(filename=filename):
                raw = json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))
                task = represent_task(raw)
                self.assertEqual(task.task_id, raw["task_id"])

    def test_catalog_has_all_four_families_and_ecm_can_preempt(self):
        families = set()
        for filename in V2_EXAMPLES:
            raw = json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))
            families.add(raw["semantics"]["family"])
        self.assertEqual(families, {"CNI", "Radar", "EW", "ISAC"})

        ecm = json.loads((EXAMPLES / "ew_ecm_task.json").read_text(encoding="utf-8"))
        self.assertTrue(ecm["qos"]["scheduling"]["can_preempt"])
        self.assertFalse(ecm["qos"]["scheduling"]["preemptible"])


if __name__ == "__main__":
    unittest.main()
