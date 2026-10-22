import json
from pathlib import Path
import tempfile
import unittest

from task_repr import ValidationError, default_capability_map, load_capability_map
from task_repr.component_models.runtime_registry import RUNTIME_COMPONENT_REGISTRY


class CapabilityMapTests(unittest.TestCase):
    def test_builtin_map_covers_the_twelve_runtime_components(self):
        mapping = default_capability_map()
        self.assertEqual({entry.cid for entry in mapping.mappings},
                         set(RUNTIME_COMPONENT_REGISTRY))

    def test_existing_task_capabilities_are_candidates_only(self):
        mapping = default_capability_map()
        self.assertEqual([entry.cid for entry in mapping.candidates_for("digital_down_conversion")],
                         ["C03"])
        self.assertEqual([entry.cid for entry in mapping.candidates_for("ofdm_fft_transform")],
                         ["C08"])
        self.assertIn("另行验证", mapping.by_component("C08").match_condition)
        self.assertEqual(mapping.candidates_for("mti_mtd_fft"), ())
        self.assertEqual(mapping.candidates_for("ofdm_timing_frequency_sync"), ())

    def test_rejects_unknown_cid(self):
        raw = {"schema_version": 1, "mappings": [
            {"cid": "C99", "capabilities": ["new_capability"], "match_condition": "条件"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capability_map.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unknown component"):
                load_capability_map(path)


if __name__ == "__main__":
    unittest.main()
