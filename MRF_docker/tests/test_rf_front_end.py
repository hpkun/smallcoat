import json
from pathlib import Path
import unittest

import numpy as np

from components import default_capability_map, default_component_catalog, load_component_specs
from components.implementations.runtime_registry import create_runtime_component
from components.runtime import DataPacket
from matching import match_task


ROOT = Path(__file__).resolve().parents[1]


def packet(samples, timestamp=0, rate=1000, center=100_000_000):
    return DataPacket(np.asarray(samples, dtype=complex), rate, center, timestamp, "time")


class RFFrontEndTests(unittest.TestCase):
    def test_agc_converges_and_streaming_keeps_gain_state(self):
        samples = np.full(100, 0.2 + 0j)
        config = dict(target_amplitude=1, attack_rate=0.05, release_rate=0.1,
                      max_gain=10)
        whole = create_runtime_component("C04", **config).process(packet(samples))
        self.assertGreater(abs(whole.samples[-1]), 0.85)
        self.assertLessEqual(abs(whole.samples[-1]), 1.1)
        streamed = create_runtime_component("C04", **config)
        output = streamed.push(packet(samples[:43])) + streamed.push(
            packet(samples[43:], timestamp=0.043))
        np.testing.assert_allclose(np.concatenate([item.samples for item in output]),
                                   whole.samples)
        self.assertEqual(streamed.metrics.call_count, 2)

    def test_receive_gain_downmix_filter_and_continuity(self):
        n = np.arange(256)
        signal = np.exp(2j * np.pi * 100 * n / 1000)
        config = dict(lo_offset_hz=100, gain_db=6.020599913279624,
                      bandwidth_hz=200, filter_taps=17)
        whole = create_runtime_component("C33", **config).process(packet(signal))
        self.assertAlmostEqual(whole.center_frequency, 100_000_100)
        self.assertGreater(abs(np.mean(whole.samples[50:])), 1.8)
        self.assertEqual(whole.metadata["rf_rx_sim"]["filter_group_delay_samples"], 8)
        streamed = create_runtime_component("C33", **config)
        output = streamed.push(packet(signal[:77])) + streamed.push(
            packet(signal[77:], timestamp=0.077))
        np.testing.assert_allclose(np.concatenate([item.samples for item in output]),
                                   whole.samples, atol=1e-12)

    def test_transmit_upmix_gain_limit_and_noise_seed(self):
        signal = np.ones(80, dtype=complex)
        tx = create_runtime_component("C34", lo_offset_hz=100, gain_db=20,
                                      clip_amplitude=0.7)
        output = tx.process(packet(signal))
        self.assertAlmostEqual(output.center_frequency, 99_999_900)
        self.assertTrue(np.all(np.abs(output.samples) <= 0.7000000001))
        self.assertEqual(output.metadata["rf_tx_sim"]["clipped_samples"], 80)
        self.assertEqual(output.metadata["rf_tx_sim"]["filter_group_delay_samples"], 0)
        filtered = create_runtime_component("C34", bandwidth_hz=200,
                                            filter_taps=17).process(packet(signal))
        self.assertEqual(filtered.metadata["rf_tx_sim"]["filter_group_delay_samples"], 8)

        config = dict(noise_std=0.1, seed=42)
        first = create_runtime_component("C33", **config).process(packet(signal))
        second = create_runtime_component("C33", **config).process(packet(signal))
        np.testing.assert_array_equal(first.samples, second.samples)
        self.assertGreater(np.std(first.samples), 0)
        streamed = create_runtime_component("C33", **config)
        chunks = streamed.push(packet(signal[:30])) + streamed.push(
            packet(signal[30:], timestamp=0.03))
        np.testing.assert_array_equal(np.concatenate([item.samples for item in chunks]),
                                      first.samples)

    def test_receive_front_end_suppresses_out_of_band_tone(self):
        n = np.arange(512)
        low = np.exp(2j * np.pi * 20 * n / 1000)
        high = np.exp(2j * np.pi * 350 * n / 1000)
        rx = create_runtime_component("C33", bandwidth_hz=100, filter_taps=63)
        low_out = rx.process(packet(low)).samples[120:]
        high_out = rx.process(packet(high)).samples[120:]
        self.assertGreater(np.sqrt(np.mean(np.abs(low_out) ** 2)), 0.8)
        self.assertLess(np.sqrt(np.mean(np.abs(high_out) ** 2)), 0.05)

    def test_invalid_parameters_and_signal_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            create_runtime_component("C33", lo_offset_hz=500).process(packet([1]))
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            create_runtime_component("C34", bandwidth_hz=1000).process(packet([1]))
        with self.assertRaisesRegex(ValueError, "gain_db"):
            create_runtime_component("C34", gain_db=150)
        with self.assertRaisesRegex(ValueError, "attack_rate"):
            create_runtime_component("C04", attack_rate=0)
        with self.assertRaisesRegex(ValueError, "release_rate"):
            create_runtime_component("C04", release_rate=2)
        with self.assertRaisesRegex(ValueError, "max_gain"):
            create_runtime_component("C04", max_gain=0)
        with self.assertRaises(ValueError):
            create_runtime_component("C33").process(packet([np.nan]))

    def test_catalog_specs_and_explicit_simulation_mapping(self):
        catalog = default_component_catalog()
        mapping = default_capability_map()
        specs = load_component_specs()
        self.assertEqual([catalog.get(cid).cid for cid in ("C04", "C33", "C34")],
                         ["C04", "C33", "C34"])
        self.assertEqual(mapping.candidates_for("rf_receive"), ())
        self.assertEqual(mapping.candidates_for("rf_transmit"), ())
        self.assertEqual(mapping.candidates_for("rf_iq_receive_simulation")[0].cid, "C33")
        self.assertEqual(mapping.candidates_for("rf_iq_transmit_simulation")[0].cid, "C34")
        self.assertTrue({"C04", "C33", "C34"} <= set(specs))
        raw = json.loads((ROOT / "examples" / "TaskDemos" / "voice_tx_task.json")
                         .read_text(encoding="utf-8"))
        raw["graph"] = {"nodes": [{
            "id": "rx-sim", "capability": "rf_iq_receive_simulation",
            "inputs": {"in": "iq/complex"}, "outputs": {"out": "iq/complex"},
            "estimated_runtime_ms": 1, "parameters": {"gain_db": 6},
        }], "edges": []}
        result = match_task(raw)
        self.assertEqual(result.status, "MATCHED", result.to_dict())
        self.assertEqual(result.bindings["rx-sim"].cid, "C33")
