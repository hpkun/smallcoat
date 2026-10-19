"""C07：已知前导的整数采样定时、数据辅助频偏/相位估计与帧对齐。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, real, vector


class SynchronizerComponent(BlockComponent):
    cid = "C07"
    metadata_key = "synchronizer"
    requires_stream_frame = True

    def __init__(self, preamble, search_window=256, threshold=0.85,
                 samples_per_symbol=1, frame_length=None, sync_algorithm="data_aided"):
        super().__init__(preamble=preamble, search_window=search_window, threshold=threshold,
                         samples_per_symbol=samples_per_symbol, frame_length=frame_length,
                         sync_algorithm=sync_algorithm)

    def _validate_config(self, preamble, search_window=256, threshold=0.85,
                         samples_per_symbol=1, frame_length=None, sync_algorithm="data_aided"):
        preamble = vector(preamble, "preamble")
        if len(preamble) < 4 or np.any(np.abs(preamble) < 1e-12):
            raise ValueError("preamble needs >= 4 nonzero sample-rate reference points")
        integer(search_window, "search_window")
        integer(samples_per_symbol, "samples_per_symbol")
        threshold = real(threshold, "threshold")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must lie in (0,1]")
        if frame_length is not None and integer(frame_length, "frame_length") < len(preamble):
            raise ValueError("frame_length must cover preamble")
        if sync_algorithm != "data_aided":
            raise ValueError("only data_aided synchronization is implemented")
        return dict(preamble=preamble, search_window=search_window, threshold=threshold,
                    samples_per_symbol=samples_per_symbol, frame_length=frame_length,
                    sync_algorithm=sync_algorithm)

    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    def _frame_size(self):
        return self._parameters["frame_length"]

    def _validate_samples(self, samples):
        super()._validate_samples(samples)
        if len(samples) < len(self._parameters["preamble"]):
            raise ValueError("observation is shorter than preamble")

    def _execute(self, samples):
        p, fs = self._parameters, self._active_packet.sample_rate
        ref = p["preamble"]
        candidates = min(p["search_window"], len(samples) - len(ref) + 1)
        n = np.arange(len(ref))
        best = (-1.0, 0, 0.0, 0.0)
        for start in range(candidates):
            segment = samples[start:start + len(ref)]
            ratio = segment / ref
            # 相邻训练采样的相位差，频偏捕获范围约为 (-fs/2,fs/2)。
            omega = float(np.angle(np.sum(ratio[1:] * ratio[:-1].conj())))
            aligned = segment * np.exp(-1j * omega * n)
            correlation = np.vdot(ref, aligned)
            norm = np.linalg.norm(ref) * np.linalg.norm(segment)
            score = min(1.0, float(abs(correlation) / norm)) if norm else 0.0
            if score > best[0]:
                best = score, start, omega, float(np.angle(correlation))
        score, start, omega, phase = best
        locked = score >= p["threshold"]
        failures = self._state.get("sync_failures", 0) + int(not locked)
        self._state["sync_failures"] = failures
        self._business = dict(locked=locked, correlation=score, search_candidates=candidates,
                              timing_offset_samples=start, frame_start_sample=start,
                              frequency_offset_hz=omega * fs / (2 * np.pi), phase_offset_rad=phase,
                              sync_failures=failures, effective_input_samples=len(samples),
                              samples_per_symbol=p["samples_per_symbol"])
        if not locked:
            return np.empty(0, dtype=complex)
        corrected = samples[start:] * np.exp(-1j * (omega * np.arange(len(samples) - start) + phase))
        # 前导保留在输出中；仅实现整数采样相位选择，无分数定时环路。
        return corrected[::p["samples_per_symbol"]]

    def _output_context(self, packet):
        return (packet.sample_rate / self._parameters["samples_per_symbol"], packet.center_frequency,
                packet.timestamp + self._business["timing_offset_samples"] / packet.sample_rate)

    # configure / process / push 由公共块框架提供；push 需显式 observation frame_length。
