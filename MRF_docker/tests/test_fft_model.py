import math
import unittest
from dataclasses import replace

from task_repr import (ComponentInstance, HardwareProfile, ResourceEstimator,
                       default_component_catalog)


class FFTModelTests(unittest.TestCase):
    def setUp(self):
        self.definition = default_component_catalog().get("C08")
        self.estimator = ResourceEstimator()

    def test_catalog_to_instance_to_resource_profile(self):
        instance = ComponentInstance(
            "radar-search-fft", self.definition.cid, "radar-search",
            {"fft_size": 1024, "batch_size": 2, "sample_rate": 1_000_000,
             "bytes_per_sample": 8},
        )
        profile = self.estimator.estimate(
            default_component_catalog().get(instance.cid), instance.parameters, HardwareProfile(),
        )
        instance = replace(instance, resource_profile=profile)

        self.assertIs(instance.resource_profile, profile)
        self.assertEqual(profile.workload, 2 * 1024 * math.log2(1024))
        self.assertEqual(profile.memory_bytes, 2 * 1024 * 8)
        self.assertEqual(profile.bandwidth_bytes_per_s, 8_000_000)
        self.assertEqual(
            (profile.cpu, profile.memory, profile.dsp, profile.fpga, profile.bandwidth),
            (10, 20, 35, 50, 35),
        )
        self.assertEqual(profile.latency_level, "L1")
        self.assertIsNone(profile.runtime_ms)
        self.assertIsNone(self.definition.estimated_runtime_ms)

    def test_fft_size_and_batch_change_dynamic_profile(self):
        small = self.estimator.estimate(self.definition, {"fft_size": 1024})
        large = self.estimator.estimate(self.definition, {"fft_size": 8192})
        batched = self.estimator.estimate(self.definition, {"fft_size": 1024, "batch_size": 2})

        self.assertGreater(large.workload, small.workload)
        self.assertGreater(large.memory_bytes, small.memory_bytes)
        self.assertEqual(batched.workload, 2 * small.workload)
        self.assertEqual(batched.memory_bytes, 2 * small.memory_bytes)
        self.assertIsNone(small.bandwidth_bytes_per_s)
        self.assertEqual(small.memory, large.memory)

    def test_invalid_parameters_are_rejected(self):
        for parameters in (
            {}, {"fft_size": 1}, {"fft_size": True}, {"fft_size": 1024.0},
            {"fft_size": 1024, "batch_size": 0},
            {"fft_size": 1024, "bytes_per_sample": -1},
            {"fft_size": 1024, "sample_rate": float("nan")},
            {"fft_size": 1024, "sample_rate": 0},
            {"fft_size": 1024, "sample_rate": "1000000"},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.estimator.estimate(self.definition, parameters)


if __name__ == "__main__":
    unittest.main()
