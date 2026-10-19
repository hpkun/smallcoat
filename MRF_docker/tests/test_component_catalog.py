import copy
import json
from pathlib import Path
import tempfile
import unittest

from task_repr import ValidationError, default_component_catalog, load_component_catalog


CATALOG_PATH = Path(__file__).resolve().parents[1] / "task_repr" / "data" / "components.json"


class ComponentCatalogTests(unittest.TestCase):
    def test_builtin_catalog_contains_all_32_components(self):
        catalog = default_component_catalog()
        self.assertEqual(len(catalog), 32)
        self.assertEqual([item.cid for item in catalog.components],
                         [f"C{index:02d}" for index in range(1, 33)])

    def test_component_keeps_function_and_scheduler_calibration(self):
        component = default_component_catalog().get("C08")
        self.assertEqual(component.short_name, "FFT/IFFT")
        self.assertIn("IQ数据块", component.input_formats)
        self.assertEqual(component.simulation_profile.resource_levels.fpga, 5)
        self.assertEqual(component.simulation_profile.latency_level, "L1")
        self.assertIsNone(component.estimated_runtime_ms)

    def test_common_components_are_available_to_each_family(self):
        radar_components = default_component_catalog().by_family("Radar")
        self.assertIn("C01", {item.cid for item in radar_components})
        self.assertIn("C21", {item.cid for item in radar_components})

    def test_latency_range_is_preserved(self):
        component = default_component_catalog().get("C05")
        profile = component.simulation_profile
        self.assertEqual(profile.latency_level, "L0-L1")
        self.assertEqual(profile.latency_bounds, (0, 1))

    def test_rejects_reverse_latency_range(self):
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        raw["components"][0]["simulation_profile"]["latency_level"] = "L2-L1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "components.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "ascending range"):
                load_component_catalog(path)

    def test_rejects_resource_level_outside_baseline(self):
        raw = copy.deepcopy(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
        raw["components"][31]["simulation_profile"]["resource_levels"]["fpga"] = 6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "components.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "integer in 0..5"):
                load_component_catalog(path)


if __name__ == "__main__":
    unittest.main()
