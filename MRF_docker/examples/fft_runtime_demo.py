"""C08 实测示例：组件定义 → 实例参数 → IQ 数据 → FFT/IFFT → 运行指标。"""

from dataclasses import asdict
import json

import numpy as np

from task_repr import ComponentInstance, ResourceEstimator, default_component_catalog
from task_repr.component_models.common.fft_component import FFTComponent
from task_repr.runtime import DataPacket


def main() -> None:
    definition = default_component_catalog().get("C08")
    instance = ComponentInstance(
        "demo-fft-001", definition.cid, "demo-receiver",
        {"fft_size": 1024, "batch_size": 2, "shift": True},
    )
    component = FFTComponent(**instance.parameters)
    sample_rate = 1_024_000.0
    tone_hz = 128_000.0
    iq = np.exp(2j * np.pi * tone_hz * np.arange(2048) / sample_rate)

    def source(data, start):
        return DataPacket(data, sample_rate, 100_000_000.0,
                          start / sample_rate, "time", {"source": "synthetic-tone"})

    # 第一块不足一帧，第二块凑齐两帧；push 负责缓存与批处理。
    assert component.push(source(iq[:300], 0)) == []
    outputs = component.push(source(iq[300:], 300))
    spectrum = outputs[0]
    bins_hz = np.fft.fftshift(np.fft.fftfreq(instance.parameters["fft_size"], 1 / sample_rate))
    detected_hz = float(bins_hz[np.argmax(np.abs(spectrum.samples[0]))])

    # 直接连接 FFT 输出包和 IFFT 输入，验证端口契约与数据可恢复性。
    inverse = FFTComponent(1024, mode="ifft", shift=True, batch_size=2)
    assert component.output_port.can_connect_to(inverse.input_port)
    recovered = inverse.process(spectrum)
    max_error = float(np.max(np.abs(recovered.samples.reshape(-1) - iq)))
    assert max_error < 1e-10

    # 估算画像与实测分开输出；不会将一次 NumPy 测量写入静态组件库。
    estimated = ResourceEstimator().estimate(
        definition, instance.parameters | {"sample_rate": sample_rate,
                                           "bytes_per_sample": iq.dtype.itemsize},
    )
    metrics = component.metrics
    report = {
        "component": f"{definition.cid} {definition.name}",
        "implementation": "NumPy CPU",
        "expected_baseband_hz": tone_hz,
        "detected_baseband_hz": detected_hz,
        "ifft_max_absolute_error": max_error,
        "estimated_input_block_bytes": estimated.memory_bytes,
        "runtime_metrics": asdict(metrics),
        "latency_ms": metrics.latency_ms,
        "samples_per_s": metrics.throughput_items_per_s,
        "input_plus_output_bytes_per_s": metrics.throughput_bytes_per_s,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
