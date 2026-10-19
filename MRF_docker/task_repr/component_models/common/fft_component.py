"""C08 可执行 FFT/IFFT 模板，资源估算仍在同目录 fft.py 中。"""

from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np

from ...runtime import DataPacket, RuntimeComponent, RuntimePort


@dataclass(frozen=True)
class FFTConfig:
    fft_size: int
    hop_size: int
    mode: str = "fft"
    window: str | None = None
    shift: bool = False
    batch_size: int = 1

    @property
    def overlap(self) -> float:
        return 1 - self.hop_size / self.fft_size


class FFTComponent(RuntimeComponent):
    """NumPy CPU 参考实现。process 处理完整帧，push 缓冲连续一维流。

    shift=True 时 FFT 对输出 fftshift，IFFT 对输入 ifftshift。
    FFT 不缩放，IFFT 按 1/N 缩放。overlap 表示 [0,1) 的重叠比例。
    """

    cid = "C08"

    # 1. 初始化
    def __init__(
        self, fft_size: int, mode: str = "fft", window: str | None = None,
        overlap: float | None = None, hop_size: int | None = None,
        shift: bool = False, batch_size: int = 1,
    ) -> None:
        super().__init__()
        self._buffer = np.empty(0, dtype=np.complex128)
        self._stream_context: dict | None = None
        self._received_samples = 0
        self._next_frame_start = 0
        self._last_batch_sizes: list[int] = []
        self._apply_config(self._validate_config(
            fft_size, mode, window, overlap, hop_size, shift, batch_size,
        ))

    @property
    def config(self) -> FFTConfig:
        return self._config

    # 2. 参数合法性检查
    @staticmethod
    def _validate_config(fft_size, mode, window, overlap, hop_size, shift, batch_size) -> FFTConfig:
        if type(fft_size) is not int or fft_size < 2:
            raise ValueError("fft_size must be an integer >= 2")
        if mode not in ("fft", "ifft"):
            raise ValueError("mode must be fft or ifft")
        if window not in (None, "rect", "hann", "hamming", "blackman"):
            raise ValueError("window must be None, rect, hann, hamming or blackman")
        if type(shift) is not bool:
            raise ValueError("shift must be boolean")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if hop_size is not None and type(hop_size) is not int:
            raise ValueError("hop_size must be an integer in 1..fft_size")
        if overlap is not None:
            if (type(overlap) not in (int, float) or not math.isfinite(overlap) or
                    not 0 <= overlap < 1):
                raise ValueError("overlap must be a finite fraction in [0, 1)")
            calculated_hop = fft_size * (1 - overlap)
            if not math.isclose(calculated_hop, round(calculated_hop), rel_tol=0, abs_tol=1e-9):
                raise ValueError("overlap must produce an integer hop_size")
            if hop_size is not None and hop_size != round(calculated_hop):
                raise ValueError("overlap and hop_size are inconsistent")
            hop_size = round(calculated_hop)
        if hop_size is None:
            hop_size = fft_size
        if type(hop_size) is not int or not 1 <= hop_size <= fft_size:
            raise ValueError("hop_size must be an integer in 1..fft_size")
        if mode == "ifft" and hop_size != fft_size:
            raise ValueError("IFFT stream consumes complete spectra; overlap is only supported for FFT")
        return FFTConfig(fft_size=fft_size, hop_size=hop_size, mode=mode,
                         window=window, shift=shift, batch_size=batch_size)

    def _validate_packet(self, packet: DataPacket, *, streaming: bool) -> None:
        self.input_port.validate(packet)
        if not np.all(np.isfinite(packet.samples)):
            raise ValueError("samples must be finite complex values")
        if streaming:
            if packet.samples.ndim != 1:
                raise ValueError("push requires a 1D continuous stream")
        elif (packet.samples.shape[-1] != self.config.fft_size or
              packet.samples.ndim == 2 and packet.samples.shape[0] > self.config.batch_size):
            raise ValueError("process requires fft_size points per frame and at most batch_size frames")

    # 3. 动态配置（验证通过后一次性切换，不混用已有输入缓存）
    def _apply_config(self, config: FFTConfig) -> None:
        window = ({
            "hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman,
        }[config.window](config.fft_size) if config.window not in (None, "rect") else None)
        source, target = ("time", "frequency") if config.mode == "fft" else ("frequency", "time")
        self._config, self._window = config, window
        self.input_port = RuntimePort("in", "input", (source,))
        self.output_port = RuntimePort("out", "output", (target,))

    def configure(self, **parameters) -> None:
        allowed = {"fft_size", "mode", "window", "overlap", "hop_size", "shift", "batch_size"}
        if parameters.keys() - allowed:
            raise ValueError(f"unknown FFT parameters: {sorted(parameters.keys() - allowed)}")
        values = dict(
            fft_size=self.config.fft_size, mode=self.config.mode, window=self.config.window,
            overlap=None, hop_size=self.config.hop_size,
            shift=self.config.shift, batch_size=self.config.batch_size,
        )
        # 改帧长时默认恢复无重叠；显式给出 overlap 时以该比例重算步长。
        if "fft_size" in parameters or "overlap" in parameters:
            values["hop_size"] = None
        values.update(parameters)
        config = self._validate_config(**values)
        if self._buffer.size:
            raise RuntimeError("reset_buffer before configuring a component with pending samples")
        self._apply_config(config)
        self.reset_buffer()

    # 4. 核心算法（计时在 RuntimeComponent._run_frame 中完成）
    def _execute(self, samples: np.ndarray) -> np.ndarray:
        data = samples * self._window if self._window is not None else samples
        if self.config.mode == "ifft":
            if self.config.shift:
                data = np.fft.ifftshift(data)
            return np.fft.ifft(data)
        spectrum = np.fft.fft(data)
        return np.fft.fftshift(spectrum) if self.config.shift else spectrum

    def _make_output(self, frames: np.ndarray, context: dict, timestamps: list[float],
                     *, batched: bool) -> DataPacket:
        transformed = [self._run_frame(frame) for frame in frames]
        samples = np.stack(transformed) if batched else transformed[0]
        metadata = deepcopy(context["metadata"])
        metadata["fft"] = dict(
            fft_size=self.config.fft_size, mode=self.config.mode, window=self.config.window,
            shift=self.config.shift, hop_size=self.config.hop_size,
            batch_size=len(frames), frame_timestamps=timestamps,
            bin_width_hz=context["sample_rate"] / self.config.fft_size,
            normalization="backward",
        )
        self._last_batch_sizes.append(len(frames))
        return DataPacket(
            samples, context["sample_rate"], context["center_frequency"], timestamps[0],
            self.output_port.domains[0], metadata,
        )

    @staticmethod
    def _context(packet: DataPacket) -> dict:
        return dict(sample_rate=packet.sample_rate, center_frequency=packet.center_frequency,
                    timestamp=packet.timestamp, domain=packet.domain,
                    dtype=packet.samples.dtype, metadata=deepcopy(dict(packet.metadata)))

    # 5. 对外完整帧接口，不截断、不补零、不隐式使用输入缓存
    def process(self, packet: DataPacket) -> DataPacket:
        def action():
            self._last_batch_sizes = []
            self._validate_packet(packet, streaming=False)
            if self._buffer.size:
                raise RuntimeError("cannot mix process with pending push samples")
            frames = packet.samples.reshape(-1, self.config.fft_size)
            timestamps = [packet.timestamp + index * self.config.hop_size / packet.sample_rate
                          for index in range(len(frames))]
            return self._make_output(frames, self._context(packet), timestamps,
                                     batched=packet.samples.ndim == 2)
        return self._measure_call("process", packet, action)

    # 6. 流式处理/缓冲区：只发出完整批次，flush 可发出不足批量的完整帧
    def push(self, packet: DataPacket) -> list[DataPacket]:
        def action():
            self._last_batch_sizes = []
            self._validate_packet(packet, streaming=True)
            if not packet.item_count:
                return []
            context = self._context(packet)
            if self._stream_context is not None:
                previous = self._stream_context
                for field in ("sample_rate", "center_frequency", "domain", "dtype", "metadata"):
                    if context[field] != previous[field]:
                        raise ValueError(f"stream {field} changed; reset_buffer before a new stream")
                expected = previous["timestamp"] + self._received_samples / previous["sample_rate"]
                tolerance = max(1e-9, abs(float(np.spacing(expected))) * 2)
                if not math.isclose(packet.timestamp, expected, rel_tol=0, abs_tol=tolerance):
                    raise ValueError("stream timestamp is discontinuous; reset_buffer before a new stream")
                context = previous
            buffer = (np.concatenate((self._buffer, packet.samples)) if self._buffer.size
                      else packet.samples.copy())
            self._observe_buffer(buffer.nbytes)
            outputs, retained, next_start = self._drain(buffer, context, partial=False)
            # 成功后再提交状态；算法异常时保留旧缓存，调用方可重试。
            self._buffer, self._next_frame_start = retained, next_start
            self._stream_context = context
            self._received_samples += packet.item_count
            return outputs
        return self._measure_call("push", packet, action)

    def _drain(self, buffer: np.ndarray, context: dict, *, partial: bool):
        outputs = []
        offset = 0
        n, hop, batch = self.config.fft_size, self.config.hop_size, self.config.batch_size
        while len(buffer) - offset >= n:
            available = 1 + (len(buffer) - offset - n) // hop
            if available < batch and not partial:
                break
            count = min(batch, available)
            frames = np.stack([buffer[offset + i * hop:offset + i * hop + n]
                               for i in range(count)])
            timestamps = [context["timestamp"] + (self._next_frame_start + offset + i * hop)
                          / context["sample_rate"] for i in range(count)]
            outputs.append(self._make_output(frames, context, timestamps, batched=batch > 1))
            offset += count * hop
        return outputs, buffer[offset:].copy(), self._next_frame_start + offset

    def flush(self) -> list[DataPacket]:
        """发出剩余完整帧；不足一帧的尾部保留，不自动补零或丢弃。"""
        def action():
            self._last_batch_sizes = []
            if self._stream_context is None:
                return []
            outputs, retained, start = self._drain(self._buffer, self._stream_context, partial=True)
            self._buffer, self._next_frame_start = retained, start
            return outputs
        return self._measure_call("flush", None, action)

    def reset_buffer(self) -> int:
        """显式开始新流，返回释放的缓存样点数（含重叠保留样点）。"""
        discarded = len(self._buffer)
        self._buffer = np.empty(0, dtype=np.complex128)
        self._stream_context = None
        self._received_samples = self._next_frame_start = 0
        self._metrics = replace(self._metrics, buffer_current_bytes=0)
        return discarded

    def _buffer_nbytes(self) -> int:
        return self._buffer.nbytes

    def _specific_metrics(self) -> dict:
        return dict(fft_size=self.config.fft_size, mode=self.config.mode,
                    window=self.config.window, shift=self.config.shift,
                    hop_size=self.config.hop_size, configured_batch_size=self.config.batch_size,
                    batch_sizes=list(self._last_batch_sizes))
