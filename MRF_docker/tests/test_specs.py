import inspect
import json
from pathlib import Path
import tempfile
import unittest

from task_repr import ValidationError, load_component_specs, load_protocol_specs, match_task
from task_repr.component_models.runtime_registry import RUNTIME_COMPONENT_REGISTRY


DATA = Path(__file__).resolve().parents[1] / "task_repr" / "data"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"


class SpecificationTests(unittest.TestCase):
    def test_runtime_interfaces_cover_constructor_properties(self):
        interfaces = load_component_specs()
        self.assertEqual(set(interfaces), set(RUNTIME_COMPONENT_REGISTRY))
        for cid, component in RUNTIME_COMPONENT_REGISTRY.items():
            with self.subTest(cid=cid):
                parameters = set(inspect.signature(component).parameters)
                self.assertEqual(set(interfaces[cid].properties), parameters)
                required = {name for name, parameter in inspect.signature(component).parameters.items()
                            if parameter.default is inspect.Parameter.empty}
                self.assertEqual(set(interfaces[cid].required_properties), required)

    def test_unknown_protocol_reference_is_rejected(self):
        raw = json.loads((DATA / "ocs.json").read_text(encoding="utf-8"))
        raw["components"][0]["ports"]["in"]["protocols"].append("missing/protocol")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocs.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unknown OPS"):
                load_component_specs(path)

    def test_duplicate_protocol_is_rejected(self):
        raw = json.loads((DATA / "ops.json").read_text(encoding="utf-8"))
        raw["protocols"].append(raw["protocols"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ops.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "duplicate OPS"):
                load_protocol_specs(path)

    def test_ocs_rejects_wrong_property_type(self):
        error = load_component_specs()["C08"].check_property("fft_size", "1024")
        self.assertIn("must be integer", error)

    def test_matching_uses_protocol_registry(self):
        raw = json.loads((EXAMPLES / "signal_analysis_task.json").read_text(encoding="utf-8"))
        inputs = raw["graph"]["nodes"][0]["inputs"]
        inputs[next(iter(inputs))] = "unknown/format"
        result = match_task(raw)
        self.assertTrue(any("OPS" in issue.detail for issue in result.unresolved))


if __name__ == "__main__":
    unittest.main()
