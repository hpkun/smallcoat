"""C03：具有连续 NCO 相位和 FIR 状态的数字上下变频参考实现。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import fir, integer, lowpass, real, vector


class DDCComponent(BlockComponent):
    cid = "C03"
    metadata_key = "ddc"

    # 1. 初始化；process/push 的公共计时与事务由 BlockComponent 提供。
    def __init__(self, mode="ddc", nco_frequency=0.0, decimation=1,
                 interpolation=1, bandwidth=None, filter_coefficients=None, filter_taps=63):
        super().__init__(mode=mode, nco_frequency=nco_frequency, decimation=decimation,
                         interpolation=interpolation, bandwidth=bandwidth,
                         filter_coefficients=filter_coefficients, filter_taps=filter_taps)

    # 2. 参数检查；bandwidth 是双边通带总带宽，单位 Hz。
    def _validate_config(self, mode="ddc", nco_frequency=0.0, decimation=1,
                         interpolation=1, bandwidth=None, filter_coefficients=None, filter_taps=63):
        if mode not in ("ddc", "duc"):
            raise ValueError("mode must be ddc or duc")
        integer(decimation, "decimation")
        integer(interpolation, "interpolation")
        if mode == "ddc" and interpolation != 1 or mode == "duc" and decimation != 1:
            raise ValueError("only the rate factor for the selected mode may differ from 1")
        nco_frequency = real(nco_frequency, "nco_frequency")
        if bandwidth is not None:
            bandwidth = real(bandwidth, "bandwidth", 0)
            if bandwidth == 0:
                raise ValueError("bandwidth must be positive")
        integer(filter_taps, "filter_taps", 3)
        if filter_taps % 2 == 0:
            raise ValueError("filter_taps must be odd")
        if filter_coefficients is not None:
            filter_coefficients = vector(filter_coefficients, "filter_coefficients")
        return dict(mode=mode, nco_frequency=nco_frequency, decimation=decimation,
                    interpolation=interpolation, bandwidth=bandwidth,
                    filter_coefficients=filter_coefficients, filter_taps=filter_taps)

    # 3. configure 复用公共实现；有活动流时需先 reset_buffer。
    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    # 4. 核心算法。滤波器群延迟保留在元数据，不伪装成已补偿的时间戳。
    def _execute(self, samples):
        p, fs = self._parameters, self._active_packet.sample_rate
        up, down = p["interpolation"], p["decimation"]
        work_rate = fs * up
        if abs(p["nco_frequency"]) >= work_rate / 2:
            raise ValueError("NCO must lie strictly inside Nyquist")
        cutoff = 0.8 / max(up, down) if p["bandwidth"] is None else p["bandwidth"] / work_rate
        if not 0 < cutoff < 1 or cutoff > 1 / max(up, down):
            raise ValueError("bandwidth exceeds the resampling anti-alias limit")
        h = p["filter_coefficients"]
        if h is None:
            h = lowpass(p["filter_taps"], cutoff)
        count = self._state.get("input_count", 0)
        if p["mode"] == "ddc":
            t = count + np.arange(len(samples))
            mixed = samples * np.exp(-2j * np.pi * p["nco_frequency"] * t / fs)
            filtered, history = self._run_stage("filter", fir, mixed, h, self._state.get("fir"))
            first = (-count) % down
            output = filtered[first::down]
        else:
            expanded = np.zeros(len(samples) * up, dtype=complex)
            expanded[::up] = samples
            filtered, history = self._run_stage("filter", fir, expanded, h * up, self._state.get("fir"))
            t = count * up + np.arange(len(expanded))
            output = filtered * np.exp(2j * np.pi * p["nco_frequency"] * t / work_rate)
            first = 0
        self._state.update(fir=history, input_count=count + len(samples))
        self._business = dict(mode=p["mode"], nco_frequency_hz=p["nco_frequency"],
                              decimation=down, interpolation=up, first_input_offset=first,
                              filter_taps=len(h), filter_group_delay_samples=(len(h) - 1) / 2,
                              input_count=len(samples), output_count=len(output))
        return output

    def _output_context(self, packet):
        p = self._parameters
        center = packet.center_frequency + (1 if p["mode"] == "ddc" else -1) * p["nco_frequency"]
        return (packet.sample_rate * p["interpolation"] / p["decimation"], center,
                packet.timestamp + self._business["first_input_offset"] / packet.sample_rate)

    # 5/6. process 独立块、push 连续流，由公共基类实现。


DDCDUCComponent = DDCComponent
