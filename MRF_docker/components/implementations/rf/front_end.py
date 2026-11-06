"""C33/C34: RF front-end effects represented on complex IQ samples."""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import fir, integer, lowpass, real


class _IQFrontEnd(BlockComponent):
    def _validate_front_end(self, lo_offset_hz, gain_db, bandwidth_hz,
                            filter_taps, clip_amplitude):
        lo_offset_hz = real(lo_offset_hz, "lo_offset_hz")
        gain_db = real(gain_db, "gain_db")
        if not -120 <= gain_db <= 120:
            raise ValueError("gain_db must be between -120 and 120 dB")
        if bandwidth_hz is not None:
            bandwidth_hz = real(bandwidth_hz, "bandwidth_hz", 0)
            if bandwidth_hz == 0:
                raise ValueError("bandwidth_hz must be > 0")
        integer(filter_taps, "filter_taps", 3)
        if filter_taps % 2 == 0:
            raise ValueError("filter_taps must be odd")
        if clip_amplitude is not None:
            clip_amplitude = real(clip_amplitude, "clip_amplitude", 0)
            if clip_amplitude == 0:
                raise ValueError("clip_amplitude must be > 0")
        return dict(lo_offset_hz=lo_offset_hz, gain_db=gain_db,
                    bandwidth_hz=bandwidth_hz, filter_taps=filter_taps,
                    clip_amplitude=clip_amplitude)

    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    def _filter(self, samples):
        p = self._parameters
        if p["bandwidth_hz"] is None:
            return samples
        cutoff = p["bandwidth_hz"] / self._active_packet.sample_rate
        if not 0 < cutoff < 1:
            raise ValueError("bandwidth_hz must be below sample_rate")
        coefficients = lowpass(p["filter_taps"], cutoff)
        output, history = self._run_stage(
            "filter", fir, samples, coefficients, self._state.get("fir"))
        self._state["fir"] = history
        return output

    def _mix(self, samples, sign):
        frequency = self._parameters["lo_offset_hz"]
        rate = self._active_packet.sample_rate
        if abs(frequency) >= rate / 2:
            raise ValueError("lo_offset_hz must lie strictly inside Nyquist")
        count = self._state.get("input_count", 0)
        phase = (count + np.arange(len(samples))) * (sign * 2 * np.pi * frequency / rate)
        self._state["input_count"] = count + len(samples)
        return samples * np.exp(1j * phase)

    def _limit(self, samples):
        limit = self._parameters["clip_amplitude"]
        if limit is None:
            return samples, 0
        amplitude = np.abs(samples)
        clipped = amplitude > limit
        scale = np.ones_like(amplitude)
        scale[clipped] = limit / amplitude[clipped]
        return samples * scale, int(np.count_nonzero(clipped))


class RFReceiveIQComponent(_IQFrontEnd):
    cid = "C33"
    metadata_key = "rf_rx_sim"

    def __init__(self, lo_offset_hz=0.0, gain_db=0.0, bandwidth_hz=None,
                 filter_taps=33, clip_amplitude=None, noise_std=0.0, seed=0):
        super().__init__(lo_offset_hz=lo_offset_hz, gain_db=gain_db,
                         bandwidth_hz=bandwidth_hz, filter_taps=filter_taps,
                         clip_amplitude=clip_amplitude, noise_std=noise_std, seed=seed)

    def _validate_config(self, lo_offset_hz=0.0, gain_db=0.0, bandwidth_hz=None,
                         filter_taps=33, clip_amplitude=None, noise_std=0.0, seed=0):
        values = self._validate_front_end(lo_offset_hz, gain_db, bandwidth_hz,
                                          filter_taps, clip_amplitude)
        values["noise_std"] = real(noise_std, "noise_std", 0)
        values["seed"] = integer(seed, "seed", 0)
        return values

    def _execute(self, samples):
        p = self._parameters
        signal = samples * 10 ** (p["gain_db"] / 20)
        if p["noise_std"]:
            rng = np.random.default_rng()
            rng.bit_generator.state = self._state.get(
                "rng_state", np.random.default_rng(p["seed"]).bit_generator.state)
            draws = rng.standard_normal((len(samples), 2))
            noise = (draws[:, 0] + 1j * draws[:, 1]) * (p["noise_std"] / np.sqrt(2))
            signal = signal + noise
            self._state["rng_state"] = rng.bit_generator.state
        signal, clipped = self._limit(signal)
        signal = self._filter(self._mix(signal, -1))
        self._business = {"gain_db": p["gain_db"], "lo_offset_hz": p["lo_offset_hz"],
                          "noise_std": p["noise_std"], "clipped_samples": clipped,
                          "bandwidth_hz": p["bandwidth_hz"],
                          "filter_group_delay_samples": (p["filter_taps"] - 1) / 2
                          if p["bandwidth_hz"] is not None else 0}
        return signal

    def _output_context(self, packet):
        return (packet.sample_rate,
                packet.center_frequency + self._parameters["lo_offset_hz"],
                packet.timestamp)


class RFTransmitIQComponent(_IQFrontEnd):
    cid = "C34"
    metadata_key = "rf_tx_sim"

    def __init__(self, lo_offset_hz=0.0, gain_db=0.0, bandwidth_hz=None,
                 filter_taps=33, clip_amplitude=None):
        super().__init__(lo_offset_hz=lo_offset_hz, gain_db=gain_db,
                         bandwidth_hz=bandwidth_hz, filter_taps=filter_taps,
                         clip_amplitude=clip_amplitude)

    def _validate_config(self, lo_offset_hz=0.0, gain_db=0.0, bandwidth_hz=None,
                         filter_taps=33, clip_amplitude=None):
        return self._validate_front_end(lo_offset_hz, gain_db, bandwidth_hz,
                                        filter_taps, clip_amplitude)

    def _execute(self, samples):
        p = self._parameters
        signal = self._mix(self._filter(samples), +1) * 10 ** (p["gain_db"] / 20)
        signal, clipped = self._limit(signal)
        self._business = {"gain_db": p["gain_db"], "lo_offset_hz": p["lo_offset_hz"],
                          "clipped_samples": clipped, "bandwidth_hz": p["bandwidth_hz"],
                          "filter_group_delay_samples": (p["filter_taps"] - 1) / 2
                          if p["bandwidth_hz"] is not None else 0}
        return signal

    def _output_context(self, packet):
        return (packet.sample_rate,
                packet.center_frequency - self._parameters["lo_offset_hz"],
                packet.timestamp)
