"""同步块处理与定长组帧的公共实现，不在各组件中重复计时和流校验。"""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np

from .base import RuntimeComponent
from .packet import DataPacket


class BlockComponent(RuntimeComponent):
    """process 是独立数据块；push 保持算法状态，并检查一维流的连续性。

    子类通过 _frame_size 指定流式帧长；None 表示逐块连续算法。
    数据包构造及端口校验成功后才提交状态，失败可用相同输入重试。
    """

    metadata_key = "component"
    requires_stream_frame = False

    def __init__(self, **parameters):
        super().__init__()
        self._parameters = {}
        self._state = {}
        self._business = {}
        self._context = None
        self._received = 0
        self._consumed = 0
        self._buffer = np.empty(0, dtype=complex)
        self.configure(**parameters)

    @property
    def config(self):
        return deepcopy(self._parameters)

    def configure(self, **parameters):
        values = deepcopy(self._parameters)
        values.update(parameters)
        values = self._validate_config(**values)
        if self._context is not None:
            raise RuntimeError("reset_buffer before configuring an active stream")
        self._parameters = values
        self._setup()
        self._state = {}
        self._business = {}

    def _validate_config(self, **parameters):
        raise NotImplementedError

    def _setup(self):
        raise NotImplementedError

    def _frame_size(self):
        return None

    def _frame_hop(self):
        return self._frame_size()

    def _validate_samples(self, samples):
        size = self._frame_size()
        if not samples.size:
            raise ValueError("process requires a non-empty block")
        if size is not None and samples.size != size:
            raise ValueError(f"expected {size} input items per frame")

    def _prepare_frame(self, samples):
        return samples

    def _output_context(self, packet):
        return packet.sample_rate, packet.center_frequency, packet.timestamp

    def _make_output(self, packet):
        self._validate_samples(packet.samples)
        self._active_packet = packet
        self._business = {}
        result = self._run_frame(self._prepare_frame(packet.samples))
        rate, center, timestamp = self._output_context(packet)
        metadata = deepcopy(dict(packet.metadata))
        metadata[self.metadata_key] = deepcopy(self._business)
        # DataPacket 不允许空的二维矩阵；无输出统一用一维空载荷表示。
        if result.size == 0:
            result = result.reshape(0)
        output = DataPacket(result, rate, center, timestamp,
                            self.output_port.domains[0], metadata)
        self.output_port.validate(output)
        self._observe_buffer(self._buffer_nbytes())
        return output

    def _check_packet(self, packet):
        self.input_port.validate(packet)
        if not np.all(np.isfinite(packet.samples)):
            raise ValueError("input must contain finite values")

    def process(self, packet):
        def action():
            self._check_packet(packet)
            self._validate_samples(packet.samples)
            if self._context is not None:
                raise RuntimeError("cannot mix process with an active push stream")
            previous = self._state
            self._state = {}
            try:
                return self._make_output(packet)
            finally:
                self._state = previous
        return self._measure_call("process", packet, action)

    def _check_stream(self, packet):
        self._check_packet(packet)
        if packet.samples.ndim != 1:
            raise ValueError("push requires a 1D stream")
        if self._context is None:
            return
        old = self._context
        for field in ("sample_rate", "center_frequency", "domain"):
            if getattr(packet, field) != getattr(old, field):
                raise ValueError(f"stream {field} changed; reset_buffer first")
        if packet.samples.dtype != old.samples.dtype or dict(packet.metadata) != dict(old.metadata):
            raise ValueError("stream dtype or metadata changed; reset_buffer first")
        expected = old.timestamp + self._received / old.sample_rate
        tolerance = max(1e-9, abs(float(np.spacing(expected))) * 2)
        if not math.isclose(packet.timestamp, expected, abs_tol=tolerance, rel_tol=0):
            raise ValueError("stream timestamp is discontinuous; reset_buffer first")

    def push(self, packet):
        def action():
            self._check_stream(packet)
            if self.requires_stream_frame and self._frame_size() is None:
                raise ValueError("configure a frame size before push")
            if not packet.item_count:
                return []
            # 仅保存空载荷上下文，避免持有已经消费的大输入包。
            context = self._context or DataPacket(
                np.empty(0, dtype=packet.samples.dtype), packet.sample_rate,
                packet.center_frequency, packet.timestamp, packet.domain, packet.metadata)
            old_state, old_business = deepcopy(self._state), deepcopy(self._business)
            try:
                if self._frame_size() is None:
                    output = self._make_output(packet)
                    outputs = [output] if output.item_count else []
                    retained, consumed = self._buffer, self._received + packet.item_count
                else:
                    buffer = (np.concatenate((self._buffer, packet.samples)) if self._buffer.size
                              else packet.samples.copy())
                    self._observe_buffer(buffer.nbytes + self._state_nbytes())
                    outputs, offset = [], 0
                    size, hop = self._frame_size(), self._frame_hop()
                    while len(buffer) - offset >= size:
                        frame = DataPacket(buffer[offset:offset + size], context.sample_rate,
                                           context.center_frequency,
                                           context.timestamp + (self._consumed + offset) / context.sample_rate,
                                           context.domain, context.metadata)
                        output = self._make_output(frame)
                        if output.item_count:
                            outputs.append(output)
                        offset += hop
                    retained, consumed = buffer[offset:].copy(), self._consumed + offset
                self._buffer, self._consumed = retained, consumed
                self._received += packet.item_count
                self._context = context
                return outputs
            except Exception:
                self._state, self._business = old_state, old_business
                raise
        return self._measure_call("push", packet, action)

    def flush(self):
        """定长组件在 push 时即输出所有完整帧；此处不补零、不丢弃尾部。"""
        return self._measure_call("flush", None, lambda: [])

    def reset_buffer(self):
        released = int(self._buffer.size)
        self._buffer = np.empty(0, dtype=complex)
        self._state = {}
        self._context = None
        self._received = self._consumed = 0
        self._metrics = replace(self._metrics, buffer_current_bytes=0)
        return released

    def _state_nbytes(self):
        def count(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(item) for item in value)
            return 0
        return count(self._state)

    def _buffer_nbytes(self):
        return self._buffer.nbytes + self._state_nbytes()

    def _specific_metrics(self):
        return deepcopy(self._business)
