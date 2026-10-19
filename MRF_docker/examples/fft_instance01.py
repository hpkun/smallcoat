"""A runnable C08 FFT component instance."""

import json

import numpy as np

from task_repr import ComponentInstance, ResourceEstimator, default_component_catalog
from task_repr.component_models.runtime_registry import create_runtime_component
from task_repr.runtime import DataPacket


fft_instance01 = ComponentInstance(
    instance_id="fft_instance01",
    cid="C08",
    task_id="fft_demo",
    parameters={
        "fft_size": 1024,
        "mode": "fft",
        "window": "hann",
        "overlap": 0.5,
        "shift": True,
        "batch_size": 1,
    },
)


def main() -> None:
    component = create_runtime_component(fft_instance01.cid, **fft_instance01.parameters)
    sample_rate = 1_024_000.0
    tone_hz = 128_000.0
    samples = np.exp(
        2j * np.pi * tone_hz * np.arange(fft_instance01.parameters["fft_size"]) / sample_rate
    )
    packet = DataPacket(samples, sample_rate, 100_000_000.0, 0.0, "time")
    spectrum = component.process(packet)
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / sample_rate))
    peak_index = int(np.argmax(np.abs(spectrum.samples)))
    profile = ResourceEstimator().estimate(
        default_component_catalog().get(fft_instance01.cid),
        fft_instance01.parameters | {
            "sample_rate": sample_rate,
            "bytes_per_sample": samples.dtype.itemsize,
        },
    )
    print(json.dumps({
        "instance_id": fft_instance01.instance_id,
        "cid": fft_instance01.cid,
        "parameters": fft_instance01.parameters,
        "output_domain": spectrum.domain,
        "peak_frequency_hz": float(frequencies[peak_index]),
        "estimated_workload": profile.workload,
        "estimated_input_block_bytes": profile.memory_bytes,
        "measured_latency_ms": component.metrics.latency_ms,
    }, indent=2))


if __name__ == "__main__":
    main()
