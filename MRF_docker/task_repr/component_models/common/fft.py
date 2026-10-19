"""C08 FFT/IFFT 的第一版参数化工作量与数据量模型。"""

import math
from typing import Mapping

from ..base import ComponentModel
from ...models.component import ComponentDefinition
from ...models.resource import ResourceProfile
from ...resource.hardware import HardwareProfile


def _positive_int(parameters: Mapping[str, str | int | float | bool], name: str,
                  default: int | None = None) -> int:
    value = parameters.get(name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class FFTModel(ComponentModel):
    """估计归一化工作量和最小输入数据量，不预测未经标定的运行毫秒数。"""

    cid = "C08"

    def estimate(
        self,
        definition: ComponentDefinition,
        parameters: Mapping[str, str | int | float | bool],
        hardware: HardwareProfile | None = None,
    ) -> ResourceProfile:
        if definition.cid != self.cid:
            raise ValueError(f"FFTModel requires {self.cid}, got {definition.cid}")

        n_fft = _positive_int(parameters, "fft_size")
        if n_fft < 2:
            raise ValueError("fft_size must be >= 2")
        batch_size = _positive_int(parameters, "batch_size", 1)
        bytes_per_sample = _positive_int(parameters, "bytes_per_sample", 8)

        sample_rate = parameters.get("sample_rate")
        if sample_rate is not None:
            if (type(sample_rate) not in (int, float) or
                    not math.isfinite(sample_rate) or sample_rate <= 0):
                raise ValueError("sample_rate must be a positive finite number in samples/s")

        # N*log2(N) 是复杂度代理量，不等于设备上的实际运算次数。
        workload = batch_size * n_fft * math.log2(n_fft)
        memory_bytes = batch_size * n_fft * bytes_per_sample
        # Fs 是总输入样点率；输出和内部搬移需在硬件标定时另计。
        bandwidth_bytes_per_s = (sample_rate * bytes_per_sample
                                 if sample_rate is not None else None)
        if bandwidth_bytes_per_s is not None and not math.isfinite(bandwidth_bytes_per_s):
            raise ValueError("sample_rate and bytes_per_sample produce non-finite bandwidth")
        units = definition.simulation_profile.resource_levels.to_normalized_units()
        return ResourceProfile(
            cpu=units["cpu"],
            memory=units["mem"],
            dsp=units["dsp"],
            fpga=units["fpga"],
            bandwidth=units["bw"],
            latency_level=definition.simulation_profile.latency_level,
            runtime_ms=None,
            workload=workload,
            memory_bytes=memory_bytes,
            bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        )
