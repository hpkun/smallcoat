import unittest

from task_repr import ComponentInstance, ResourceProfile, default_component_catalog
from task_repr.component_models import COMPONENT_MODEL_REGISTRY, get_component_model


class ComponentModelBoundaryTests(unittest.TestCase):
    def test_definition_and_instances_are_separate(self):
        definition = default_component_catalog().get("C08")
        small = ComponentInstance("radar-1-fft", definition.cid, "radar-1", {"fft_size": 1024})
        large = ComponentInstance("radar-2-fft", definition.cid, "radar-2", {"fft_size": 8192})

        self.assertEqual(small.cid, large.cid)
        self.assertNotEqual(small.parameters, large.parameters)
        self.assertIsNone(small.resource_profile)
        self.assertIsNone(definition.estimated_runtime_ms)

    def test_only_fft_model_is_registered(self):
        self.assertEqual(set(COMPONENT_MODEL_REGISTRY), {"C08"})
        self.assertEqual(get_component_model("C08").cid, "C08")
        self.assertIsNone(get_component_model("C03"))

    def test_instance_can_hold_estimated_profile_without_changing_definition(self):
        definition = default_component_catalog().get("C08")
        profile = ResourceProfile(10, 20, 35, 50, 35, "L1", None)
        instance = ComponentInstance("radar-1-fft", definition.cid, "radar-1",
                                     {"fft_size": 1024}, profile)
        self.assertIs(instance.resource_profile, profile)
        self.assertIsNone(definition.estimated_runtime_ms)


if __name__ == "__main__":
    unittest.main()
