"""11 个核心运行组件的统一接口和基本算法回归测试。"""

import unittest

import numpy as np

from task_repr.component_models.runtime_registry import (
    RUNTIME_COMPONENT_REGISTRY,
    create_runtime_component,
)
from task_repr.runtime import DataPacket


def packet(samples, domain="time", sample_rate=1_000_000.0, timestamp=0.0):
    """构造测试数据包，显式保留运行时所需的信号上下文。"""
    return DataPacket(np.asarray(samples), sample_rate, 100e6, timestamp, domain, {})


class RuntimeComponentTests(unittest.TestCase):
    def test_registry_contains_all_twelve_runtime_components(self):
        self.assertEqual(
            set(RUNTIME_COMPONENT_REGISTRY),
            {"C03", "C05", "C07", "C08", "C09", "C14", "C15", "C17",
             "C21", "C23", "C27", "C32"},
        )

    def test_ddc_mixes_a_tone_to_baseband(self):
        fs = 1_000_000.0
        tone = np.exp(2j * np.pi * 100_000 * np.arange(256) / fs)
        component = create_runtime_component("C03", nco_frequency=100_000,
                                             filter_taps=9)
        output = component.process(packet(tone, sample_rate=fs))
        baseband = output.samples[32:]
        self.assertGreater(abs(np.mean(baseband)), 0.8)
        self.assertLess(np.std(baseband / np.mean(baseband)), 0.15)
        self.assertEqual(component.metrics.call_count, 1)

    def test_filter_streaming_matches_one_block(self):
        samples = np.exp(1j * 0.07 * np.arange(37))
        coefficients = np.array([0.25, 0.5, 0.25])
        whole = create_runtime_component("C05", coefficients=coefficients)
        streamed = create_runtime_component("C05", coefficients=coefficients)
        expected = whole.process(packet(samples)).samples
        first = streamed.push(packet(samples[:13]))
        second = streamed.push(packet(samples[13:], timestamp=13 / 1_000_000))
        actual = np.concatenate([x.samples for x in first + second])
        np.testing.assert_allclose(actual, expected)

    def test_synchronizer_finds_known_preamble(self):
        preamble = np.array([1 + 0j, 1 + 1j, 1 - 1j, -1 + 0j])
        samples = np.r_[np.zeros(3, complex), preamble, [0.2 + 0.1j, -0.1j]]
        component = create_runtime_component("C07", preamble=preamble,
                                             frame_length=len(samples), threshold=0.8)
        output = component.process(packet(samples))
        self.assertTrue(output.metadata["synchronizer"]["locked"])
        self.assertEqual(output.metadata["synchronizer"]["timing_offset_samples"], 3)

    def test_channelizer_exposes_channel_metadata(self):
        component = create_runtime_component("C09", channel_count=4)
        output = component.process(packet(np.ones(64, complex)))
        self.assertEqual(output.samples.shape, (4, 16))
        self.assertEqual(output.metadata["channelizer"]["channel_count"], 4)
        self.assertEqual(len(output.metadata["channelizer"]["channels"]), 4)

    def test_psk_qam_modulate_and_demodulate_round_trip(self):
        source = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        modem = create_runtime_component("C14", mode="modulate", modulation="qpsk")
        demod = create_runtime_component("C14", mode="demodulate", modulation="qpsk")
        symbols = modem.process(packet(source, domain="bits"))
        recovered = demod.process(symbols)
        np.testing.assert_array_equal(recovered.samples, source)

    def test_ofdm_tx_rx_round_trip(self):
        allocation = [1, 2, 3, 5, 6]
        source = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j, 0.5 + 0.2j])
        tx = create_runtime_component("C15", mode="tx", fft_length=8,
                                      cp_length=2, subcarrier_allocation=allocation)
        rx = create_runtime_component("C15", mode="rx", fft_length=8,
                                      cp_length=2, subcarrier_allocation=allocation)
        waveform = tx.process(packet(source, domain="symbols"))
        recovered = rx.process(waveform)
        np.testing.assert_allclose(recovered.samples, source, atol=1e-12)

    def test_fec_convolutional_encode_decode_round_trip(self):
        source = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)
        encoder = create_runtime_component("C17", mode="encode", block_length=len(source))
        decoder = create_runtime_component("C17", mode="decode", block_length=len(source))
        encoded = encoder.process(packet(source, domain="bits"))
        decoded = decoder.process(encoded)
        np.testing.assert_array_equal(decoded.samples, source)

    def test_pulse_compression_peak_is_at_echo_delay(self):
        reference = np.ones(4, complex)
        delay = 5
        echo = np.r_[np.zeros(delay, complex), reference, np.zeros(2, complex)]
        component = create_runtime_component("C21", reference_waveform=reference,
                                             method="time")
        output = component.process(packet(echo))
        self.assertEqual(int(np.argmax(abs(output.samples))), delay + len(reference) - 1)

    def test_cfar_detects_an_inserted_target(self):
        power = np.ones(31)
        power[15] = 100.0
        component = create_runtime_component("C23", guard_cells=1,
                                             reference_cells=4, pfa=0.1)
        output = component.process(packet(power, domain="power"))
        self.assertTrue(output.samples[15])
        self.assertGreater(output.metadata["cfar"]["detection_count"], 0)

    def test_spectrum_sensing_detects_a_tone_band(self):
        n = 128
        tone = np.exp(2j * np.pi * 8 * np.arange(n) / 64)
        component = create_runtime_component("C27", fft_length=64,
                                             averaging=1, threshold_ratio=4)
        output = component.process(packet(tone))
        self.assertEqual(output.domain, "psd")
        self.assertGreater(output.metadata["spectrum"]["event_count"], 0)

    def test_drfm_push_and_flush_produce_a_delayed_stream(self):
        samples = np.arange(9, dtype=complex)
        component = create_runtime_component("C32", delay_samples=2)
        outputs = component.push(packet(samples[:4]))
        outputs += component.push(packet(samples[4:], timestamp=4 / 1_000_000))
        outputs += component.flush()
        result = np.concatenate([item.samples for item in outputs])
        np.testing.assert_allclose(result, np.r_[0, 0, samples])
        replay = component.replay()
        self.assertEqual(replay.samples.size, samples.size)

    def test_failed_call_is_counted_by_common_metrics(self):
        component = create_runtime_component("C05")
        with self.assertRaises(ValueError):
            component.process(packet(np.array([], dtype=complex)))
        self.assertEqual(component.metrics.call_count, 1)
        self.assertEqual(component.metrics.error_count, 1)


if __name__ == "__main__":
    unittest.main()
