"""C27：双边功率谱密度、Welch/滑动平均与连续频点事件合并。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, real, window_values


class SpectrumSensingComponent(BlockComponent):
    cid = "C27"
    metadata_key = "spectrum"

    def __init__(self, fft_length=1024, window="hann", overlap=0.5, averaging=8,
                 detection_threshold=None, threshold_ratio=10.0, scan_bandwidth=None):
        super().__init__(fft_length=fft_length, window=window, overlap=overlap,
                         averaging=averaging, detection_threshold=detection_threshold,
                         threshold_ratio=threshold_ratio, scan_bandwidth=scan_bandwidth)

    def _validate_config(self, fft_length=1024, window="hann", overlap=0.5, averaging=8,
                         detection_threshold=None, threshold_ratio=10.0, scan_bandwidth=None):
        integer(fft_length, "fft_length", 2)
        window_values(window, fft_length)
        overlap = real(overlap, "overlap")
        hop = fft_length * (1 - overlap)
        if not 0 <= overlap < 1 or not np.isclose(hop, round(hop), rtol=0, atol=1e-9):
            raise ValueError("overlap must give an integer hop in 1..fft_length")
        integer(averaging, "averaging")
        if detection_threshold is not None:
            detection_threshold = real(detection_threshold, "detection_threshold", 0)
        threshold_ratio = real(threshold_ratio, "threshold_ratio", 1)
        if scan_bandwidth is not None:
            scan_bandwidth = real(scan_bandwidth, "scan_bandwidth", 0)
            if scan_bandwidth == 0:
                raise ValueError("scan_bandwidth must be positive")
        return dict(fft_length=fft_length, window=window, overlap=overlap, averaging=averaging,
                    detection_threshold=detection_threshold, threshold_ratio=threshold_ratio,
                    scan_bandwidth=scan_bandwidth)

    def _setup(self):
        self._window = window_values(self._parameters["window"], self._parameters["fft_length"])
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("psd",), ("f",), (1,), "cells")

    def _frame_size(self):
        return self._parameters["fft_length"]

    def _frame_hop(self):
        return round(self._frame_size() * (1 - self._parameters["overlap"]))

    def _validate_samples(self, samples):
        if len(samples) < self._frame_size():
            raise ValueError("PSD needs at least fft_length samples")

    def _execute(self, samples):
        p, packet = self._parameters, self._active_packet
        n, hop = p["fft_length"], self._frame_hop()
        if p["scan_bandwidth"] is not None and p["scan_bandwidth"] > packet.sample_rate:
            raise ValueError("scan_bandwidth exceeds sample_rate")
        history = list(self._state.get("spectra", []))
        segments = 0
        for offset in range(0, len(samples) - n + 1, hop):
            spectrum = self._run_stage("fft", np.fft.fft, samples[offset:offset + n] * self._window)
            density = np.fft.fftshift(abs(spectrum) ** 2 / (packet.sample_rate * np.sum(self._window ** 2)))
            history.append(density)
            history = history[-p["averaging"]:]
            segments += 1
        psd = self._run_stage("average", lambda: np.mean(history, axis=0))
        self._state["spectra"] = history
        offsets = np.fft.fftshift(np.fft.fftfreq(n, 1 / packet.sample_rate))
        frequencies = offsets + packet.center_frequency
        noise = max(float(np.median(psd)), float(np.max(psd)) * 1e-12, np.finfo(float).tiny)
        threshold = p["detection_threshold"] if p["detection_threshold"] is not None else noise * p["threshold_ratio"]
        active = psd > threshold
        if p["scan_bandwidth"] is not None:
            active &= abs(offsets) <= p["scan_bandwidth"] / 2
        boundaries = np.diff(np.concatenate(([False], active, [False])).astype(int))
        events, df = [], packet.sample_rate / n
        for start, stop in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            peak = int(start + np.argmax(psd[start:stop]))
            events.append(dict(start_bin=int(start), stop_bin=int(stop),
                               lower_frequency_hz=float(frequencies[start] - df / 2),
                               upper_frequency_hz=float(frequencies[stop - 1] + df / 2),
                               peak_frequency_hz=float(frequencies[peak]),
                               integrated_power=float(np.sum(psd[start:stop]) * df)))
        self._business = dict(fft_length=n, segment_count=segments, averaging_count=len(history),
                              frequency_axis_hz=frequencies.tolist(), bin_width_hz=df,
                              psd_units="input_amplitude_squared_per_hz", threshold=threshold,
                              noise_floor=noise, peak_frequency_hz=float(frequencies[np.argmax(psd)]),
                              events=events, event_count=len(events), history_bytes=sum(x.nbytes for x in history))
        return psd

    # process 对块内完整窗口作 Welch，保留最后 averaging 个谱；push 跨块滑窗平均。
