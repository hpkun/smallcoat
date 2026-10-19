"""C08 的真实信号、流状态和公共运行测量测试。"""

import unittest
from unittest.mock import patch

import numpy as np

from task_repr import ComponentInstance, ResourceEstimator, default_component_catalog
from task_repr.component_models.common.fft_component import FFTComponent
from task_repr.runtime import DataPacket, RuntimeComponent


def packet(samples, timestamp=0, domain="time", **changes):
    values = dict(samples=samples, sample_rate=8, center_frequency=100,
                  timestamp=timestamp, domain=domain, metadata={"channel": 1})
    return DataPacket(**(values | changes))


def direct_dft(samples):
    """小规模独立 DFT 参考，避免仅拿同一 FFT 函数做对照。"""
    n = len(samples)
    return np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) @ samples


class FFTComponentTests(unittest.TestCase):
    def test_definition_instance_estimator_and_execution_chain(self):
        definition = default_component_catalog().get("C08")
        instance = ComponentInstance("voice-fft", definition.cid, "voice",
                                     {"fft_size": 8, "batch_size": 2})
        profile = ResourceEstimator().estimate(definition, instance.parameters)
        component = FFTComponent(**instance.parameters)
        result = component.process(packet(np.ones((2, 8), dtype=complex)))
        self.assertEqual(component.cid, instance.cid)
        self.assertEqual(profile.workload, 48)
        self.assertIsNone(profile.runtime_ms)
        self.assertEqual(result.samples.shape, (2, 8))
        self.assertEqual(component.metrics.total_frames, 2)
        self.assertGreater(component.metrics.latency_ms, 0)

    def test_tone_peak_metadata_and_full_runtime_chain(self):
        component = FFTComponent(8)
        self.assertIsInstance(component, RuntimeComponent)
        data = np.exp(2j * np.pi * 2 * np.arange(8) / 8)
        result = component.process(packet(data, timestamp=1.5))
        self.assertEqual(result.domain, "frequency")
        self.assertEqual(np.argmax(np.abs(result.samples)), 2)
        self.assertAlmostEqual(abs(result.samples[2]), 8)
        np.testing.assert_allclose(result.samples, direct_dft(data), atol=1e-13)
        self.assertEqual((result.sample_rate, result.center_frequency, result.timestamp), (8, 100, 1.5))
        self.assertEqual(result.metadata["channel"], 1)
        self.assertEqual(result.metadata["fft"]["frame_timestamps"], [1.5])
        self.assertEqual(result.metadata["fft"]["bin_width_hz"], 1)
        metrics = component.metrics
        self.assertTrue(metrics.last_call_succeeded)
        self.assertGreater(metrics.processing_time_ns, 0)
        self.assertEqual((metrics.input_bytes, metrics.output_bytes), (data.nbytes, result.nbytes))
        self.assertEqual(metrics.total_frames, 1)

    def test_forward_inverse_roundtrip_and_ports_for_odd_and_even_lengths(self):
        for n in (7, 8):
            for shift in (False, True):
                with self.subTest(n=n, shift=shift):
                    data = np.arange(n) + 1j * np.arange(n)[::-1]
                    forward = FFTComponent(n, shift=shift)
                    inverse = FFTComponent(n, mode="ifft", shift=shift)
                    self.assertTrue(forward.output_port.can_connect_to(inverse.input_port))
                    spectrum = forward.process(packet(data))
                    recovered = inverse.process(spectrum)
                    np.testing.assert_allclose(recovered.samples, data, atol=1e-12)
                    self.assertEqual(recovered.domain, "time")

    def test_windows_against_independent_dft(self):
        data = np.arange(8).astype(complex)
        for name, window in ((None, np.ones(8)), ("rect", np.ones(8)),
                             ("hann", np.hanning(8)), ("hamming", np.hamming(8)),
                             ("blackman", np.blackman(8))):
            with self.subTest(window=name):
                actual = FFTComponent(8, window=name).process(packet(data))
                np.testing.assert_allclose(actual.samples, direct_dft(data * window), atol=1e-12)

    def test_complete_batch_is_processed_with_frame_measurements(self):
        component = FFTComponent(8, batch_size=3)
        data = np.array([np.ones(8), np.eye(1, 8)[0]], dtype=complex)
        result = component.process(packet(data, timestamp=2))
        np.testing.assert_allclose(result.samples[0], [8, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(result.samples[1], np.ones(8))
        self.assertEqual(result.samples.shape, (2, 8))
        self.assertEqual(result.metadata["fft"]["frame_timestamps"], [2, 3])
        self.assertEqual(result.metadata["fft"]["batch_size"], 2)
        metrics = component.metrics
        self.assertEqual(metrics.total_frames, 2)
        self.assertEqual(len(metrics.component_specific_metrics["frame_processing_times_ns"]), 2)
        self.assertEqual(metrics.component_specific_metrics["batch_sizes"], [2])

    def test_split_stream_overlap_timestamps_and_buffer_measurement(self):
        component = FFTComponent(8, overlap=0.5)
        data = np.arange(16).astype(complex)
        self.assertEqual(component.push(packet(data[:3])), [])
        outputs = component.push(packet(data[3:10], timestamp=3 / 8))
        outputs += component.push(packet(data[10:], timestamp=10 / 8))
        self.assertEqual(len(outputs), 3)
        for index, output in enumerate(outputs):
            np.testing.assert_allclose(output.samples, direct_dft(data[index * 4:index * 4 + 8]), atol=1e-12)
            self.assertEqual(output.timestamp, index * 0.5)
        metrics = component.metrics
        self.assertEqual(metrics.buffer_current_bytes, 4 * 16)
        self.assertEqual(metrics.buffer_peak_bytes, 12 * 16)
        self.assertEqual((metrics.total_input_samples, metrics.total_output_samples), (16, 24))
        self.assertEqual(metrics.total_frames, 3)
        self.assertEqual(metrics.component_specific_metrics["frame_count"], 2)
        self.assertEqual(component.flush(), [])
        self.assertEqual(component.reset_buffer(), 4)
        self.assertEqual(component.metrics.buffer_current_bytes, 0)
        self.assertEqual(component.metrics.buffer_peak_bytes, 12 * 16)

    def test_batch_wait_flush_and_tail_continuation(self):
        component = FFTComponent(4, batch_size=2)
        data = np.arange(8).astype(complex)
        self.assertEqual(component.push(packet(data[:7])), [])
        self.assertEqual(component.metrics.buffer_current_bytes, 7 * 16)
        outputs = component.flush()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].samples.shape, (1, 4))
        np.testing.assert_allclose(outputs[0].samples[0], direct_dft(data[:4]), atol=1e-13)
        self.assertEqual(component.metrics.buffer_current_bytes, 3 * 16)
        self.assertEqual(component.metrics.input_bytes, 0)
        self.assertEqual(component.push(packet(data[7:], timestamp=7 / 8)), [])
        output = component.flush()[0]
        np.testing.assert_allclose(output.samples[0], direct_dft(data[4:]), atol=1e-13)
        self.assertEqual(output.timestamp, 0.5)
        self.assertEqual(component.metrics.total_frames, 2)
        self.assertEqual(component.reset_buffer(), 0)

    def test_complete_stream_batches_and_ifft_stream(self):
        forward = FFTComponent(4, batch_size=2)
        source = np.arange(12).astype(complex)
        batches = forward.push(packet(source)) + forward.flush()
        self.assertEqual([item.samples.shape for item in batches], [(2, 4), (1, 4)])
        inverse = FFTComponent(4, mode="ifft")
        spectra = np.concatenate([item.samples.reshape(-1) for item in batches])
        restored = inverse.push(packet(spectra, domain="frequency"))
        np.testing.assert_allclose(np.concatenate([item.samples for item in restored]), source, atol=1e-12)

    def test_rejects_invalid_configurations(self):
        for changes in (
            {"fft_size": 1}, {"fft_size": True}, {"fft_size": 4.0},
            {"mode": "FFT"}, {"window": "unknown"}, {"shift": 1},
            {"batch_size": 0}, {"batch_size": True}, {"overlap": -0.1},
            {"overlap": 1}, {"overlap": True}, {"overlap": float("nan")},
            {"overlap": 0.3}, {"hop_size": 0}, {"hop_size": 9},
            {"hop_size": 4.0}, {"overlap": 0.5, "hop_size": 4.0},
            {"overlap": 0.5, "hop_size": True},
            {"overlap": 0.5, "hop_size": 3}, {"mode": "ifft", "overlap": 0.5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FFTComponent(**({"fft_size": 8} | changes))

    def test_rejects_invalid_packets_and_counts_errors(self):
        component = FFTComponent(8)
        cases = [packet(np.ones(8)), packet(np.ones(7, dtype=complex)),
                 packet(np.ones((2, 8), dtype=complex)),
                 packet(np.ones(8, dtype=complex), domain="frequency"),
                 packet(np.full(8, complex("nan")))]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                component.process(invalid)
        with self.assertRaises(TypeError):
            component.process(np.ones(8, dtype=complex))
        with self.assertRaises(ValueError):
            component.push(packet(np.ones((1, 8), dtype=complex)))
        self.assertEqual(component.metrics.error_count, 7)
        self.assertEqual(component.metrics.total_frames, 0)
        result = component.process(packet(np.ones(8, dtype=complex)))
        self.assertEqual(result.samples[0], 8)

    def test_stream_rejects_discontinuities_without_losing_pending_data(self):
        component = FFTComponent(8)
        component.push(packet(np.ones(3, dtype=complex)))
        for changes in ({"timestamp": 0}, {"sample_rate": 16},
                        {"center_frequency": 101}, {"metadata": {"channel": 2}},
                        {"samples": np.ones(5, dtype=np.complex64)}):
            invalid = dict(samples=np.ones(5, dtype=complex), timestamp=3 / 8) | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                component.push(packet(**invalid))
            self.assertEqual(component.metrics.buffer_current_bytes, 3 * 16)
        result = component.push(packet(np.ones(5, dtype=complex), timestamp=3 / 8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].samples[0], 8)
        self.assertEqual(component.metrics.total_input_samples, 8)

    def test_algorithm_failure_preserves_buffer_for_retry(self):
        component = FFTComponent(8)
        component.push(packet(np.ones(3, dtype=complex)))
        final_packet = packet(np.ones(5, dtype=complex), timestamp=3 / 8)
        with patch.object(component, "_execute", side_effect=RuntimeError("simulated failure")):
            with self.assertRaises(RuntimeError):
                component.push(final_packet)
        self.assertEqual(component.metrics.buffer_current_bytes, 3 * 16)
        self.assertEqual(component.metrics.error_count, 1)
        result = component.push(final_packet)
        self.assertEqual(result[0].samples[0], 8)

    def test_config_is_validated_atomically_and_requires_empty_buffer(self):
        component = FFTComponent(8, overlap=0.5)
        initial = component.config
        with self.assertRaises(ValueError):
            component.configure(fft_size=1)
        with self.assertRaises(ValueError):
            component.configure(unrecognized=True)
        self.assertEqual(component.config, initial)
        component.push(packet(np.ones(3, dtype=complex)))
        with self.assertRaises(RuntimeError):
            component.configure(mode="ifft", hop_size=8)
        with self.assertRaises(RuntimeError):
            component.process(packet(np.ones(8, dtype=complex)))
        self.assertEqual(component.reset_buffer(), 3)
        component.configure(fft_size=4, mode="ifft", shift=True)
        self.assertEqual(component.config.hop_size, 4)
        self.assertEqual(component.input_port.domains, ("frequency",))
        component.configure(mode="fft", overlap=0.5)
        self.assertEqual(component.config.hop_size, 2)

    def test_metric_reset_preserves_buffer_and_empty_stream_is_noop(self):
        component = FFTComponent(8)
        component.push(packet(np.ones(3, dtype=complex)))
        component.reset_metrics()
        self.assertEqual(component.metrics.call_count, 0)
        self.assertEqual(component.metrics.buffer_current_bytes, 3 * 16)
        self.assertEqual(component.metrics.buffer_peak_bytes, 3 * 16)
        self.assertEqual(component.push(packet(np.empty(0, dtype=complex), timestamp=99)), [])
        result = component.push(packet(np.ones(5, dtype=complex), timestamp=3 / 8))
        self.assertEqual(len(result), 1)
        self.assertEqual(component.metrics.total_input_samples, 5)
        self.assertEqual(component.metrics.total_output_samples, 8)


if __name__ == "__main__":
    unittest.main()
