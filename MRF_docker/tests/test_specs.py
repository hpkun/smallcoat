import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from task_repr import ValidationError, load_component_specs, load_protocol_specs, match_task
from task_repr.component_models.runtime_registry import RUNTIME_COMPONENT_REGISTRY
from task_repr.runtime import DataPacket, RuntimePort
from task_repr.matching.contracts import validate_runtime_packet


DATA = Path(__file__).resolve().parents[1] / "task_repr" / "data"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"


class SpecificationTests(unittest.TestCase):
    def test_ops_validates_actual_packet_fields(self):
        specs = load_protocol_specs()
        packet = DataPacket(np.ones(16, dtype=complex), 4_000_000,
                            100_000_000, 0, "frequency")
        self.assertIsNone(specs["frequency/complex"].validate_packet(packet))
        self.assertIn("domain", validate_runtime_packet("iq/complex", packet))
        self.assertIn("dtype", specs["frequency/complex"].validate_packet(
            DataPacket(np.ones(16), 4_000_000, 0, 0, "frequency")))
        self.assertIn("dimensions", specs["frequency/complex"].validate_packet(
            DataPacket(np.ones((2, 16), dtype=complex), 4_000_000, 0, 0, "frequency")))
        self.assertIn("DataPacket", specs["frequency/complex"].validate_packet(object()))
        self.assertIn("finite samples", specs["frequency/complex"].validate_packet(
            DataPacket(np.array([complex(float("nan"), 0)]), 4_000_000, 0, 0, "frequency")))

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

    def test_fft_ocs_ports_follow_mode_even_with_permissive_runtime_ports(self):
        interface = load_component_specs()["C08"]
        input_port = RuntimePort("in", "input", ("time", "frequency"))
        output_port = RuntimePort("out", "output", ("time", "frequency"))
        for mode, input_format, output_format, wrong_input, wrong_output in (
            ("fft", "iq/complex", "frequency/complex", "frequency/complex", "iq/complex"),
            ("ifft", "frequency/complex", "iq/complex", "iq/complex", "frequency/complex"),
        ):
            with self.subTest(mode=mode):
                config = {"mode": mode}
                self.assertIsNone(interface.check_port("in", input_format, input_port, config))
                self.assertIsNone(interface.check_port("out", output_format, output_port, config))
                self.assertIn("does not declare", interface.check_port("in", wrong_input, input_port, config))
                self.assertIn("does not declare", interface.check_port("out", wrong_output, output_port, config))
        self.assertIn("no OCS port profile", interface.check_port("in", "iq/complex", input_port))

    def test_fft_ocs_rejects_inconsistent_mode_profiles(self):
        raw = json.loads((DATA / "ocs.json").read_text(encoding="utf-8"))
        del raw["components"][3]["ports"]["out"]["protocols_by_mode"]["ifft"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocs.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "mode profiles must agree"):
                load_component_specs(path)

    def test_fft_and_ifft_match_their_declared_ports(self):
        raw = json.loads((EXAMPLES / "signal_analysis_task.json").read_text(encoding="utf-8"))
        raw["graph"]["edges"] = []
        for capability, input_format, output_format in (
            ("fft", "iq/complex", "frequency/complex"),
            ("ifft", "frequency/complex", "iq/complex"),
        ):
            with self.subTest(capability=capability):
                raw["graph"]["nodes"] = [{
                    "id": "transform", "capability": capability,
                    "inputs": {"in": input_format}, "outputs": {"out": output_format},
                    "estimated_runtime_ms": 1,
                    "parameters": {"fft_size": 16, "input_block_length": 16},
                }]
                result = match_task(raw)
                self.assertEqual(result.status, "MATCHED", result.to_dict())
                self.assertEqual(result.bindings["transform"].parameters["mode"], capability)

    def test_matching_uses_protocol_registry(self):
        raw = json.loads((EXAMPLES / "signal_analysis_task.json").read_text(encoding="utf-8"))
        inputs = raw["graph"]["nodes"][0]["inputs"]
        inputs[next(iter(inputs))] = "unknown/format"
        result = match_task(raw)
        self.assertTrue(any("OPS" in issue.detail for issue in result.unresolved))


if __name__ == "__main__":
    unittest.main()
