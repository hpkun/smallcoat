"""模拟业务请求的静态匹配和频谱分析组件链。"""

from graphlib import TopologicalSorter
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from components.implementations.runtime_registry import create_runtime_component
from components.runtime import DataPacket
from matching import match_task
from matching.contracts import validate_runtime_packet
from task_repr import parse_external_task, refine_business_task


ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "examples" / "BusinessRequests" / "External" / "07_spectrum_analysis.json"
OUTPUT = ROOT / "outputs"


def _input_packet(request: dict) -> DataPacket:
    processing = request["request"]["processing"]
    rate = processing["input_sample_rate_mhz"] * 1_000_000
    count = processing["input_block_length"]
    time = np.arange(count) / rate
    rng = np.random.default_rng(7)
    samples = (np.exp(2j * np.pi * 300_000 * time)
               + 0.65 * np.exp(2j * np.pi * 700_000 * time)
               + 0.01 * (rng.standard_normal(count) + 1j * rng.standard_normal(count)))
    return DataPacket(samples, rate, 100_000_000, 0.0, "time")


def _spectrum(packet: DataPacket) -> tuple[np.ndarray, np.ndarray]:
    count = packet.item_count
    window = np.hanning(count)
    values = np.fft.fftshift(np.fft.fft(packet.samples * window))
    frequency = np.fft.fftshift(np.fft.fftfreq(count, 1 / packet.sample_rate)) / 1_000_000
    level = 20 * np.log10(np.maximum(abs(values) / sum(window), 1e-12))
    return frequency, level


def _probe(packet: DataPacket, frequency_hz: float) -> float:
    frequency, level = _spectrum(packet)
    index = np.argmin(abs(frequency - frequency_hz / 1_000_000))
    return float(max(level[max(0, index - 2):index + 3]))


def _plot(packets: dict[str, DataPacket], nodes: list, summary: dict, output: Path) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "Noto Sans SC", "SimHei")
                 if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(4, 1, figsize=(11, 10.5), sharex=False,
                             gridspec_kw={"hspace": 0.43})
    fig.patch.set_facecolor("white")
    fig.suptitle("射频任务模拟：双音 IQ → 下变频 → FIR 滤波 → 功率谱密度",
                 x=0.09, y=0.985, ha="left", fontsize=16, fontweight="bold")
    stages = [("输入 IQ", packets["source"], "#52616b"),
              ("C03  数字下变频 / 抽取 ×2", packets[nodes[0].node_id], "#216f99"),
              ("C05  FIR 低通滤波", packets[nodes[1].node_id], "#147d68")]
    for index, (label, packet, color) in enumerate(stages):
        frequency, level = _spectrum(packet)
        axis = axes[index]
        axis.plot(frequency, level, color=color, linewidth=1.05)
        axis.set_title(f"{label}  ·  {packet.item_count} 点  ·  {packet.sample_rate / 1e6:g} MS/s",
                       loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("幅度 / dB")
        axis.set_xlabel("相对中心频率 / MHz")
        axis.set_xlim(0, 0.9 if index == 0 else 0.8)
        axis.set_ylim(-90, 8)
        axis.axvline(0.3 if index == 0 else 0.2, color="#147d68", linestyle="--", alpha=0.7)
        axis.axvline(0.7 if index == 0 else 0.6, color="#ba6a35", linestyle="--", alpha=0.7)
        axis.grid(color="#e2e8e8", linewidth=0.6)
    psd = packets[nodes[2].node_id]
    frequency = np.fft.fftshift(np.fft.fftfreq(psd.item_count, 1 / psd.sample_rate)) / 1e6
    density = 10 * np.log10(np.maximum(psd.samples, np.finfo(float).tiny))
    axis = axes[3]
    axis.plot(frequency, density, color="#147d68", linewidth=1.2)
    axis.set_title(f"C27  频谱感知 / PSD  ·  {psd.item_count} 频点  ·  峰值 {summary['peak_offset_khz']:.1f} kHz",
                   loc="left", fontsize=11, fontweight="bold")
    axis.set_ylabel("PSD / dB/Hz")
    axis.set_xlabel("相对中心频率 / MHz")
    axis.set_xlim(0, 0.8)
    axis.axvline(0.2, color="#147d68", linestyle="--", alpha=0.7)
    axis.grid(color="#e2e8e8", linewidth=0.6)
    fig.text(0.09, 0.014,
             f"绿色虚线：目标信号；橙色虚线：待抑制干扰。600 kHz 干扰经 FIR 处理降低 "
             f"{summary['interferer_reduction_db']:.1f} dB。此图只展示模拟数据与组件子图。",
             fontsize=9, color="#52616b")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    business = parse_external_task(request)
    refinement = refine_business_task(business)
    match = match_task(refinement)
    if refinement.task is None or match.status != "MATCHED":
        raise RuntimeError(f"static matching failed: {match.to_dict()}")

    graph = refinement.task.G
    by_id = {node.node_id: node for node in graph.nodes}
    incoming = {node_id: set() for node_id in by_id}
    for edge in graph.edges:
        incoming[edge.target].add(edge.source)
    order = list(TopologicalSorter(incoming).static_order())
    packets = {"source": _input_packet(request)}
    stages = []
    for node_id in order:
        node = by_id[node_id]
        parents = incoming[node_id]
        if len(parents) > 1:
            raise RuntimeError(f"demo only supports a single input per node: {node_id}")
        packet = packets[next(iter(parents))] if parents else packets["source"]
        input_error = validate_runtime_packet(node.inputs["in"], packet)
        if input_error:
            raise RuntimeError(f"{node_id} input: {input_error}")
        binding = match.bindings[node_id]
        component = create_runtime_component(binding.cid, **binding.parameters)
        result = component.process(packet)
        output_error = validate_runtime_packet(node.outputs["out"], result)
        if output_error:
            raise RuntimeError(f"{node_id} output: {output_error}")
        packets[node_id] = result
        stages.append({"node_id": node_id, "role": node.business_node_id,
                       "capability": node.capability, "cid": binding.cid,
                       "input_items": packet.item_count, "output_items": result.item_count,
                       "input_rate_msps": packet.sample_rate / 1e6,
                       "output_rate_msps": result.sample_rate / 1e6,
                       "output_domain": result.domain,
                       "measured_processing_ms": round(component.metrics.latency_ms, 4)})

    ddc, filtered, psd = (packets[node_id] for node_id in order)
    peak_frequency = psd.metadata["spectrum"]["peak_frequency_hz"]
    summary = {"task_id": business["task_id"], "match_status": match.status,
               "match_scope": match.to_dict()["match_scope"],
               "input_tones_khz": [300, 700], "ddc_nco_khz": 100,
               "peak_offset_khz": round((peak_frequency - psd.center_frequency) / 1000, 3),
               "interferer_reduction_db": round(_probe(ddc, 600_000) - _probe(filtered, 600_000), 3),
               "psd_event_count": psd.metadata["spectrum"]["event_count"],
               "stages": stages,
               "boundaries": list(graph.boundaries),
               "note": "静态匹配及模拟 IQ 组件链；不代表外部流绑定、资源规划或端到端 QoS 验证"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "spectrum_analysis_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(packets, [by_id[node_id] for node_id in order], summary,
          OUTPUT / "spectrum_analysis_run.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
