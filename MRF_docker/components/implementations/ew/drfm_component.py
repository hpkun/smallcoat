"""C32：离线 IQ 延迟线、有限缓存与重放变换，不连接射频设备。"""

from copy import deepcopy

import numpy as np

from ...runtime import DataPacket, RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, real


class DRFMComponent(BlockComponent):
    cid = "C32"
    metadata_key = "drfm"

    def __init__(self, delay_samples=0, frequency_shift=0.0, phase=0.0,
                 amplitude=1.0, buffer_length=65536):
        super().__init__(delay_samples=delay_samples, frequency_shift=frequency_shift,
                         phase=phase, amplitude=amplitude, buffer_length=buffer_length)

    def _validate_config(self, delay_samples=0, frequency_shift=0.0, phase=0.0,
                         amplitude=1.0, buffer_length=65536):
        integer(delay_samples, "delay_samples", 0)
        integer(buffer_length, "buffer_length")
        if delay_samples > buffer_length:
            raise ValueError("delay_samples exceeds buffer_length")
        return dict(delay_samples=delay_samples, frequency_shift=real(frequency_shift, "frequency_shift"),
                    phase=real(phase, "phase"), amplitude=real(amplitude, "amplitude", 0),
                    buffer_length=buffer_length)

    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    def _transform(self, samples, offset):
        p, fs = self._parameters, self._active_packet.sample_rate
        if abs(p["frequency_shift"]) >= fs / 2:
            raise ValueError("frequency_shift must lie strictly inside Nyquist")
        t = offset + np.arange(len(samples))
        return p["amplitude"] * samples * np.exp(1j * (p["phase"] + 2 * np.pi * p["frequency_shift"] * t / fs))

    def _execute(self, samples):
        p = self._parameters
        delay = self._state.get("delay", np.zeros(p["delay_samples"], dtype=complex))
        joined = np.concatenate((delay, samples))
        count = self._state.get("input_count", 0)
        output = self._transform(joined[:len(samples)], count)
        retained = joined[len(samples):].copy()
        capture = self._state.get("capture", np.empty(0, dtype=complex))
        all_capture = np.concatenate((capture, samples))
        dropped = max(0, len(all_capture) - p["buffer_length"])
        self._state.update(delay=retained, capture=all_capture[-p["buffer_length"]:].copy(),
                           input_count=count + len(samples),
                           overwritten_samples=self._state.get("overwritten_samples", 0) + dropped)
        self._business = dict(delay_samples=p["delay_samples"], frequency_shift_hz=p["frequency_shift"],
                              amplitude=p["amplitude"], phase_rad=p["phase"],
                              replay_buffer_samples=len(self._state["capture"]),
                              delay_buffer_samples=len(retained),
                              retention_seconds=len(retained) / self._active_packet.sample_rate,
                              overwritten_samples=self._state["overwritten_samples"], underrun_count=0)
        return output

    def replay(self, repeat_count=1):
        """重放 push 已截获的最后 buffer_length 点；独立 process 不隐式保留截获数据。"""
        def action():
            integer(repeat_count, "repeat_count")
            if self._context is None or not self._state.get("capture", np.empty(0)).size:
                raise RuntimeError("push a capture before replay")
            samples = np.tile(self._state["capture"], repeat_count)
            packet = self._context
            self._active_packet = packet
            output = self._run_stage("replay_transform", self._transform, samples, 0)
            business = dict(replay=True, repeat_count=repeat_count, replay_samples=len(output),
                            frequency_shift_hz=self._parameters["frequency_shift"],
                            amplitude=self._parameters["amplitude"], phase_rad=self._parameters["phase"])
            self._business = business
            metadata = deepcopy(dict(packet.metadata))
            metadata[self.metadata_key] = business
            start = packet.timestamp + (self._received - len(self._state["capture"])) / packet.sample_rate
            return DataPacket(output, packet.sample_rate, packet.center_frequency, start, "time", metadata)
        return self._measure_call("replay", None, action)

    def flush(self):
        """发出连续流的延迟尾部；重复 flush 无输出，截获缓存继续可重放。"""
        def action():
            tail = self._state.get("delay", np.empty(0))
            if not tail.size or self._context is None:
                return []
            packet = self._context
            self._active_packet = packet
            output = self._run_stage("delay_tail", self._transform, tail, self._received)
            metadata = deepcopy(dict(packet.metadata))
            self._business = dict(flush=True, tail_samples=len(tail))
            metadata[self.metadata_key] = self._business
            result = DataPacket(output, packet.sample_rate, packet.center_frequency,
                                packet.timestamp + self._received / packet.sample_rate, "time", metadata)
            self._state["delay"] = np.empty(0, dtype=complex)
            # flush 结束延迟流，继续 push 需重置；replay 仍可调用。
            self._state["flushed"] = True
            return [result]
        return self._measure_call("flush", None, action)

    def _check_stream(self, packet):
        super()._check_stream(packet)
        if self._state.get("flushed"):
            raise RuntimeError("reset_buffer before push after flush")

    # process 输出与输入等长的零前缀延迟块；push + flush 得到完整线性延迟输出。
