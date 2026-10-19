"""统一调用计时、错误统计和显式输入缓存测量。同步实例不支持并发调用。"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from time import perf_counter_ns
from typing import Callable, TypeVar

import numpy as np

from .metrics import RuntimeMetrics
from .packet import DataPacket
from .ports import RuntimePort


_Result = TypeVar("_Result", DataPacket, list[DataPacket])


class RuntimeComponent(ABC):
    """子类提供算法和缓冲逻辑，公共框架负责所有时钟与累计统计。"""

    cid: str
    input_port: RuntimePort
    output_port: RuntimePort

    def __init__(self) -> None:
        self._metrics = RuntimeMetrics()
        self._frame_times_ns: list[int] = []
        self._stage_times_ns: dict[str, int] = {}

    @property
    def metrics(self) -> RuntimeMetrics:
        # 返回快照，调用方对业务指标字典的修改不会破坏累计统计。
        return deepcopy(self._metrics)

    def reset_metrics(self) -> None:
        current = self._buffer_nbytes()
        self._metrics = RuntimeMetrics(buffer_current_bytes=current, buffer_peak_bytes=current)

    def _buffer_nbytes(self) -> int:
        return 0

    def _observe_buffer(self, size: int) -> None:
        self._metrics = replace(self._metrics, buffer_peak_bytes=max(self._metrics.buffer_peak_bytes, size))

    def _specific_metrics(self) -> dict:
        return {}

    def _run_frame(self, samples: np.ndarray) -> np.ndarray:
        """每个实际变换单独计时，不让算法文件重复实现测量逻辑。"""
        start = perf_counter_ns()
        try:
            return self._execute(samples)
        finally:
            self._frame_times_ns.append(perf_counter_ns() - start)

    def _run_stage(self, name: str, action: Callable, *args, **kwargs):
        """统一的可选算法子阶段测量点，例如 FFT、滤波和频谱平均。"""
        start = perf_counter_ns()
        try:
            return action(*args, **kwargs)
        finally:
            self._stage_times_ns[name] = self._stage_times_ns.get(name, 0) + perf_counter_ns() - start

    def _measure_call(
        self, operation: str, packet: DataPacket | None, action: Callable[[], _Result],
    ) -> _Result:
        self._frame_times_ns = []
        self._stage_times_ns = {}
        start = perf_counter_ns()
        outputs: list[DataPacket] = []
        succeeded = False
        try:
            result = action()
            outputs = [result] if isinstance(result, DataPacket) else result
            for output in outputs:
                self.output_port.validate(output)
            succeeded = True
            return result
        finally:
            elapsed = max(1, perf_counter_ns() - start)
            old = self._metrics
            ins = packet.item_count if isinstance(packet, DataPacket) else 0
            in_bytes = packet.nbytes if isinstance(packet, DataPacket) else 0
            outs = sum(output.item_count for output in outputs) if succeeded else 0
            out_bytes = sum(output.nbytes for output in outputs) if succeeded else 0
            frames = len(self._frame_times_ns) if succeeded else 0
            specific = self._specific_metrics()
            specific.update(
                stage_processing_times_ns=dict(self._stage_times_ns),
                frame_count=frames,
                frame_attempt_count=len(self._frame_times_ns),
                frame_processing_times_ns=list(self._frame_times_ns),
                frames_per_s=frames * 1_000_000_000 / elapsed,
            )
            self._metrics = RuntimeMetrics(
                operation=operation, item_unit=self.output_port.item_unit,
                last_call_succeeded=succeeded,
                processing_time_ns=elapsed,
                input_samples=ins, output_samples=outs,
                input_bytes=in_bytes, output_bytes=out_bytes,
                buffer_current_bytes=self._buffer_nbytes(),
                buffer_peak_bytes=max(old.buffer_peak_bytes, self._buffer_nbytes()),
                call_count=old.call_count + 1, error_count=old.error_count + int(not succeeded),
                total_processing_time_ns=old.total_processing_time_ns + elapsed,
                total_success_time_ns=old.total_success_time_ns + (elapsed if succeeded else 0),
                total_input_samples=old.total_input_samples + (ins if succeeded else 0),
                total_output_samples=old.total_output_samples + outs,
                total_input_bytes=old.total_input_bytes + (in_bytes if succeeded else 0),
                total_output_bytes=old.total_output_bytes + out_bytes,
                total_frames=old.total_frames + frames,
                component_specific_metrics=specific,
            )

    @abstractmethod
    def configure(self, **parameters) -> None:
        raise NotImplementedError

    @abstractmethod
    def _execute(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def process(self, packet: DataPacket) -> DataPacket:
        raise NotImplementedError

    @abstractmethod
    def push(self, packet: DataPacket) -> list[DataPacket]:
        raise NotImplementedError
