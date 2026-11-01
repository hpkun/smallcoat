"""Visualize a real C03 -> C05 -> C27 execution with OPS/OCS validation.

Place this file in the MRF_docker project root and run:
    python visualize_dag_signal.py

Optional:
    python visualize_dag_signal.py --save-only

Requires: numpy, matplotlib, and the project's task_repr/task_executor modules.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from task_executor import DAGExecutor
from matching import match_task
from task_repr import represent_task
from components.runtime import DataPacket


ROOT = Path(__file__).resolve().parent
TASK_PATH = ROOT / "examples" / "TaskDemos" / "signal_analysis_task.json"
OUT_PATH = ROOT / "outputs" / "dag_signal_observation.png"
NODE_IDS = ("ddc-node", "filter-node", "sensing-node")
NAMES = {"ddc-node": "C03  DDC", "filter-node": "C05  FIR Filter", "sensing-node": "C27  PSD"}


def make_input_packet():
    # The example task declares fs=4 MHz, input length 4096, and NCO=100 kHz.
    # 100 kHz is shifted to baseband; a 700 kHz interference tone is filtered.
    fs = 4_000_000.0
    n = 4096
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    desired = np.exp(2j * np.pi * 100_000 * t)
    interferer = 0.40 * np.exp(2j * np.pi * 700_000 * t)
    noise = 0.025 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    iq = desired + interferer + noise
    return DataPacket(
        samples=iq,
        sample_rate=fs,
        center_frequency=100_000_000.0,
        timestamp=0.0,
        domain="time",
    )


def print_execution_report(task, match, result, source):
    nodes = {node.node_id: node for node in task.G.nodes}
    print("\n=== DAG execution report (actual component outputs) ===")
    print("Match:", match.status)
    print("Task:", task.task_id)
    print("Order:", " -> ".join(NAMES[x] for x in result.execution_order))
    print("Input: protocol=iq/complex, items={}, fs={:.3f} MHz, dtype={}, domain={}".format(
        source.item_count, source.sample_rate / 1e6, source.samples.dtype, source.domain))
    print("-" * 110)
    print("{:<17} {:<20} {:>9} {:>10} {:>12} {:>13} {:>10}".format(
        "Component", "OPS output", "Items", "Fs (MHz)", "Domain", "Dtype", "Time (ms)"))
    print("-" * 110)
    for node_id in result.execution_order:
        packet = result.node_outputs[node_id]
        metric = result.metrics[node_id]
        ops = next(iter(nodes[node_id].outputs.values()))
        print("{:<17} {:<20} {:>9} {:>10.3f} {:>12} {:>13} {:>10.4f}".format(
            NAMES[node_id], ops, packet.item_count, packet.sample_rate / 1e6,
            packet.domain, str(packet.samples.dtype), metric.latency_ms))
    print("-" * 110)
    print("Task completed:", result.completed, "| Full task completed:", result.full_task_completed)
    print("Time sum of component process() calls: {:.4f} ms (not end-to-end wall time)".format(
        sum(result.metrics[k].latency_ms for k in NODE_IDS)))
    spectrum = result.node_outputs["sensing-node"]
    meta = spectrum.metadata["spectrum"]
    print("PSD peak: {:.3f} MHz, bin width: {:.3f} Hz, active events: {}".format(
        meta["peak_frequency_hz"] / 1e6, meta["bin_width_hz"], meta["event_count"]))


def draw_dag(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Executed component DAG", loc="left", fontsize=12)
    xs = [0.04, 0.37, 0.70]
    titles = ["C03", "C05", "C27"]
    subtitles = ["DDC", "FIR Filter", "Spectrum PSD"]
    for x, name, label in zip(xs, titles, subtitles):
        patch = FancyBboxPatch((x, 0.35), 0.26, 0.36,
                               boxstyle="round,pad=0.02", fill=False, linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + 0.13, 0.57, name, ha="center", va="center", fontweight="bold")
        ax.text(x + 0.13, 0.43, label, ha="center", va="center", fontsize=9)
    for start, end in ((0.30, 0.37), (0.63, 0.70)):
        ax.annotate("", xy=(end, 0.53), xytext=(start, 0.53),
                    arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.5, 0.13, "IQ 4 MHz  ->  Baseband 2 MHz  ->  PSD",
            ha="center", fontsize=10)


def draw_samples(ax, packet, title, count=170):
    width = min(count, packet.item_count)
    times_ms = np.arange(width) / packet.sample_rate * 1000
    ax.plot(times_ms, packet.samples[:width].real, lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Real(IQ)")
    ax.grid(True, alpha=0.3)


def draw_figure(source, result):
    fig = plt.figure(figsize=(14.4, 9.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.05, 1.05])
    ax_dag = fig.add_subplot(grid[0, 0])
    ax_times = fig.add_subplot(grid[0, 1])
    ax_input = fig.add_subplot(grid[1, 0])
    ax_ddc = fig.add_subplot(grid[1, 1])
    ax_fir = fig.add_subplot(grid[2, 0])
    ax_psd = fig.add_subplot(grid[2, 1])

    draw_dag(ax_dag)

    durations = [result.metrics[k].latency_ms for k in NODE_IDS]
    ax_times.barh([NAMES[k] for k in NODE_IDS], durations)
    ax_times.invert_yaxis()
    ax_times.set_title("Measured processing time by node", loc="left", fontsize=12)
    ax_times.set_xlabel("process() duration (ms)")
    ax_times.grid(axis="x", alpha=0.3)
    for i, value in enumerate(durations):
        ax_times.text(value, i, "  {:.4f}".format(value), va="center", fontsize=9)
    ax_times.set_xlim(0, max(durations) * 1.3 if max(durations) else 1)

    draw_samples(ax_input, source, "Input IQ: 100 kHz + 700 kHz + noise")
    draw_samples(ax_ddc, result.node_outputs["ddc-node"], "After C03: DDC / decimation x2")
    draw_samples(ax_fir, result.node_outputs["filter-node"], "After C05: low-pass FIR filter")

    psd_packet = result.node_outputs["sensing-node"]
    info = psd_packet.metadata["spectrum"]
    freqs_hz = np.asarray(info["frequency_axis_hz"], dtype=float)
    # Frequency axis in the component's output metadata is already fftshifted.
    # Plot relative to the output center frequency to make the baseband visible.
    relative_freq_mhz = (freqs_hz - psd_packet.center_frequency) / 1e6
    power_db_hz = 10 * np.log10(np.maximum(psd_packet.samples, np.finfo(float).tiny))
    ax_psd.plot(relative_freq_mhz, power_db_hz, lw=1)
    ax_psd.set_title("After C27: power spectral density (PSD)", fontsize=11)
    ax_psd.set_xlabel("Frequency offset from center (MHz)")
    ax_psd.set_ylabel("PSD (dB / Hz, relative power units)")
    ax_psd.grid(True, alpha=0.3)

    fig.suptitle("OPS/OCS-checked DAG execution | " + str(result.task_id), fontsize=15)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize real OPS/OCS-checked DAG execution")
    parser.add_argument("--save-only", action="store_true", help="Save PNG without showing a GUI window")
    args = parser.parse_args()

    if not TASK_PATH.is_file():
        raise SystemExit("Task not found: {}. Place this script in the MRF_docker root.".format(TASK_PATH))
    task = represent_task(json.loads(TASK_PATH.read_text(encoding="utf-8")))
    match = match_task(task)
    if match.status != "MATCHED":
        raise SystemExit("Static match failed: {}".format(match.to_dict()))

    executor = DAGExecutor.from_match(task, match)
    source = make_input_packet()
    result = executor.execute({"ddc-node": source})
    print_execution_report(task, match, result, source)
    fig = draw_figure(source, result)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print("\nFigure saved at:", OUT_PATH)
    if not args.save_only:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
