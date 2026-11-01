"""C21：线性匹配滤波，时域/FFT 两种算法共享同一距离延迟约定。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, vector, window_values


class PulseCompressionComponent(BlockComponent):
    cid = "C21"
    metadata_key = "pulse_compression"
    requires_stream_frame = True

    def __init__(self, reference_waveform, method="fft", fft_size=None, window=None,
                 convolution_mode="full", range_gate=None, pulse_length=None):
        super().__init__(reference_waveform=reference_waveform, method=method, fft_size=fft_size,
                         window=window, convolution_mode=convolution_mode,
                         range_gate=range_gate, pulse_length=pulse_length)

    def _validate_config(self, reference_waveform, method="fft", fft_size=None, window=None,
                         convolution_mode="full", range_gate=None, pulse_length=None):
        reference_waveform = vector(reference_waveform, "reference_waveform")
        window_values(window, len(reference_waveform))
        if method not in ("fft", "time") or convolution_mode not in ("full", "same", "valid"):
            raise ValueError("unsupported compression method or convolution_mode")
        if fft_size is not None:
            integer(fft_size, "fft_size", 2)
        if pulse_length is not None:
            integer(pulse_length, "pulse_length")
            if pulse_length < len(reference_waveform):
                raise ValueError("pulse_length must cover reference_waveform")
        if range_gate is not None:
            if len(range_gate) != 2:
                raise ValueError("range_gate is a half-open [start,stop) index pair")
            range_gate = tuple(integer(x, "range_gate", 0) for x in range_gate)
            if range_gate[0] >= range_gate[1]:
                raise ValueError("range_gate requires start < stop")
        return dict(reference_waveform=reference_waveform, method=method, fft_size=fft_size,
                    window=window, convolution_mode=convolution_mode,
                    range_gate=range_gate, pulse_length=pulse_length)

    def _setup(self):
        p = self._parameters
        self._h = (p["reference_waveform"] * window_values(p["window"], len(p["reference_waveform"])))[::-1].conj()
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("range",), dimensions=(1,), item_unit="cells")

    def _frame_size(self):
        return self._parameters["pulse_length"]

    def _validate_samples(self, samples):
        super()._validate_samples(samples)
        if len(samples) < len(self._h):
            raise ValueError("echo length must cover reference waveform")

    def _execute(self, samples):
        p, m = self._parameters, len(self._h)
        count = len(samples) + m - 1
        fft_size = p["fft_size"] or (1 << (count - 1).bit_length())
        if p["method"] == "fft":
            if fft_size < count:
                raise ValueError("fft_size must cover the full linear convolution")
            a = self._run_stage("fft", np.fft.fft, samples, fft_size)
            b = self._run_stage("fft", np.fft.fft, self._h, fft_size)
            output = self._run_stage("ifft", np.fft.ifft, a * b)[:count]
        else:
            output = np.convolve(samples, self._h, "full")
        start, stop = 0, count
        if p["convolution_mode"] == "same":
            start, stop = (m - 1) // 2, (m - 1) // 2 + len(samples)
        elif p["convolution_mode"] == "valid":
            start, stop = m - 1, len(samples)
        output = output[start:stop]
        if p["range_gate"] is not None:
            lo, hi = p["range_gate"]
            if hi > len(output):
                raise ValueError("range_gate exceeds compression output")
            output = output[lo:hi]
            start += lo
        self._business = dict(method=p["method"], pulse_count=1, pulse_samples=len(samples),
                              range_cells=len(output), first_lag_samples=start - (m - 1),
                              range_spacing_m=299792458.0 / (2 * self._active_packet.sample_rate),
                              reference_bytes=self._h.nbytes,
                              fft_workspace_bytes=3 * fft_size * 16 if p["method"] == "fft" else 0,
                              fft_size=fft_size if p["method"] == "fft" else None)
        return output

    # process 一个脉冲；push 需 pulse_length，按脉冲分帧，无循环卷积混叠。
