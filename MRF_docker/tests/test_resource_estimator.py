import unittest
from unittest.mock import patch

from task_repr import HardwareProfile, ResourceEstimator, ResourceProfile, default_component_catalog
from task_repr.component_models import COMPONENT_MODEL_REGISTRY, ComponentModel


class RecordingModel(ComponentModel):
    def estimate(self, definition, parameters, hardware=None):
        return ResourceProfile(parameters["cpu_units"], 10, 0, 0, 5, "L2", 2.5)


class ResourceEstimatorTests(unittest.TestCase):
    def test_unmodeled_components_have_static_fallback(self):
        estimator = ResourceEstimator()
        for definition in default_component_catalog().components:
            if definition.cid == "C08":
                continue
            with self.subTest(cid=definition.cid):
                profile = estimator.estimate(definition, {})
                units = definition.simulation_profile.resource_levels.to_normalized_units()
                self.assertEqual(
                    (profile.cpu, profile.memory, profile.dsp, profile.fpga, profile.bandwidth),
                    (units["cpu"], units["mem"], units["dsp"], units["fpga"], units["bw"]),
                )
                self.assertEqual(profile.latency_level, definition.simulation_profile.latency_level)
                self.assertIsNone(profile.runtime_ms)

    def test_registered_model_receives_parameters_and_can_override_fallback(self):
        definition = default_component_catalog().get("C08")
        hardware = HardwareProfile()
        with patch.dict(COMPONENT_MODEL_REGISTRY, {"C08": RecordingModel}):
            profile = ResourceEstimator().estimate(definition, {"cpu_units": 42}, hardware)
        self.assertEqual(profile.cpu, 42)
        self.assertEqual(profile.runtime_ms, 2.5)


if __name__ == "__main__":
    unittest.main()
