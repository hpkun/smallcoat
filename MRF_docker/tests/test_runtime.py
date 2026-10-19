"""公共数据契约和测量分母测试，不依赖机器运行速度。"""

import unittest
from unittest.mock import patch

import numpy as np

from task_repr.runtime import DataPacket, RuntimeComponent, RuntimePort


class EchoComponent(RuntimeComponent):
    """只用于验证公共框架的透明组件。"""

    def __init__(self):
        super().__init__()
        self.input_port = RuntimePort("in", "input", ("time",))
        self.output_port = RuntimePort("out", "output", ("time",))

    def configure(self, **parameters):
        pass

    def _execute(self, samples):
        return samples.copy()

    def process(self, packet):
        def action():
            self.input_port.validate(packet)
            return DataPacket(self._run_frame(packet.samples), packet.sample_rate,
                              packet.center_frequency, packet.timestamp, "time")
        return self._measure_call("process", packet, action)

    def push(self, packet):
        raise NotImplementedError


class RuntimeContractTests(unittest.TestCase):
    def test_packet_owns_samples_and_metadata(self):
        samples = np.ones(4, dtype=np.complex64)
        metadata = {"source": {"channel": 2}}
        packet = DataPacket(samples, 8, 100, 0.5, "time", metadata)
        samples[0] = 99
        metadata["source"]["channel"] = 9
        self.assertEqual(packet.samples[0], 1)
        self.assertEqual(packet.metadata["source"]["channel"], 2)
        self.assertEqual((packet.item_count, packet.nbytes), (4, 32))
        with self.assertRaises(ValueError):
            packet.samples[0] = 3
        with self.assertRaises(TypeError):
            packet.metadata["source"] = "changed"

    def test_packet_rejects_invalid_units_shapes_and_metadata(self):
        defaults = dict(samples=np.ones(4, dtype=complex), sample_rate=8,
                        center_frequency=100, timestamp=0, domain="time")
        for changes in (
            {"sample_rate": 0}, {"sample_rate": True}, {"timestamp": float("nan")},
            {"center_frequency": float("inf")}, {"domain": " "},
            {"samples": np.ones((2, 2, 2))}, {"samples": np.empty((0, 4))},
            {"samples": ["IQ"]}, {"metadata": {"array": np.ones(2)}},
            {"metadata": {"invalid": float("nan")}}, {"metadata": {1: "value"}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DataPacket(**(defaults | changes))

    def test_port_requires_compatible_domain_dtype_dimensions_and_units(self):
        output = RuntimePort("fft", "output", ("frequency",))
        compatible = RuntimePort("ifft", "input", ("time", "frequency"))
        self.assertTrue(output.can_connect_to(compatible))
        for target in (
            RuntimePort("in", "input", ("time",)),
            RuntimePort("in", "input", ("frequency",), dtype_kinds=("f",)),
            RuntimePort("in", "input", ("frequency",), dimensions=(1,)),
            RuntimePort("in", "input", ("frequency",), item_unit="bits"),
            RuntimePort("out", "output", ("frequency",)),
        ):
            with self.subTest(target=target):
                self.assertFalse(output.can_connect_to(target))
        with self.assertRaises(TypeError):
            compatible.validate(np.ones(4))
        with self.assertRaises(ValueError):
            compatible.validate(DataPacket(np.ones(4), 8, 100, 0, "time"))

    def test_call_and_frame_timing_have_explicit_denominators(self):
        component = EchoComponent()
        packet = DataPacket(np.ones(4, dtype=complex), 8, 100, 0, "time")
        # 整体调用 100 ns，内部算法 40 ns；不用不稳定的耗时阈值做断言。
        with patch("task_repr.runtime.base.perf_counter_ns", side_effect=[100, 120, 160, 200]):
            component.process(packet)
        metrics = component.metrics
        self.assertEqual(metrics.processing_time_ns, 100)
        self.assertEqual(metrics.latency_ms, 0.0001)
        self.assertEqual((metrics.input_samples, metrics.output_samples), (4, 4))
        self.assertEqual((metrics.input_bytes, metrics.output_bytes), (64, 64))
        self.assertEqual(metrics.throughput_items_per_s, 40_000_000)
        self.assertEqual(metrics.throughput_bytes_per_s, 1_280_000_000)
        self.assertEqual(metrics.component_specific_metrics["frame_processing_times_ns"], [40])
        self.assertEqual(metrics.component_specific_metrics["frames_per_s"], 10_000_000)
        self.assertEqual(metrics.total_frames, 1)
        self.assertEqual(metrics.queue_depth, 0)

    def test_failed_calls_are_counted_without_success_throughput(self):
        component = EchoComponent()
        valid = DataPacket(np.ones(4, dtype=complex), 8, 100, 0, "time")
        with patch("task_repr.runtime.base.perf_counter_ns", side_effect=[100, 120, 160, 200]):
            component.process(valid)
        invalid = DataPacket(valid.samples, 8, 100, 0, "frequency")
        with patch("task_repr.runtime.base.perf_counter_ns", side_effect=[220, 270]):
            with self.assertRaises(ValueError):
                component.process(invalid)
        metrics = component.metrics
        self.assertFalse(metrics.last_call_succeeded)
        self.assertEqual((metrics.call_count, metrics.error_count), (2, 1))
        self.assertEqual(metrics.total_processing_time_ns, 150)
        self.assertEqual(metrics.total_success_time_ns, 100)
        self.assertEqual(metrics.average_latency_ms, 0.000075)
        self.assertEqual(metrics.total_input_samples, 4)
        self.assertEqual(metrics.total_output_samples, 4)
        self.assertEqual(metrics.input_samples, 4)  # 本次尝试输入仍可追踪。
        self.assertEqual(metrics.output_samples, 0)
        self.assertEqual(metrics.throughput_bytes_per_s, 0)
        snapshot = component.metrics
        snapshot.component_specific_metrics["frame_count"] = 99
        self.assertEqual(component.metrics.component_specific_metrics["frame_count"], 0)
        component.reset_metrics()
        self.assertEqual(component.metrics.call_count, 0)
        self.assertEqual(component.metrics.throughput_items_per_s, 0)


if __name__ == "__main__":
    unittest.main()
