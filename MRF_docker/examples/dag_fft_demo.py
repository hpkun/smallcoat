"""Run the C08 node from the existing video backhaul task DAG."""

import json
from pathlib import Path

import numpy as np

from task_executor import DAGExecutor
from task_repr import represent_task
from task_repr.runtime import DataPacket


def main() -> None:
    path = Path(__file__).parent / "TaskDemos" / "cni_video_backhaul_task.json"
    task = represent_task(json.loads(path.read_text(encoding="utf-8")))
    fft_node = next(node for node in task.G.nodes if node.node_id == "video-fft")
    sample_rate = 2_048_000
    tone_hz = 128_000
    samples = np.exp(2j * np.pi * tone_hz * np.arange(fft_node.parameters["fft_size"])
                     / sample_rate)
    source = DataPacket(samples, sample_rate, 100_000_000, 0, "time")
    executor = DAGExecutor(task, {"video-fft": "C08"},
                           active_nodes={"video-sync", "video-fft"})
    result = executor.execute({"video-sync": source})
    spectrum = result.sink_outputs["video-fft"].samples
    peak_bin = int(np.argmax(np.abs(spectrum)))
    frequency = float(np.fft.fftfreq(len(spectrum), 1 / sample_rate)[peak_bin])
    print(json.dumps({
        "task_id": result.task_id,
        "selected_nodes_completed": result.completed,
        "full_task_completed": result.full_task_completed,
        "execution_order": result.execution_order,
        "peak_frequency_hz": frequency,
    }, indent=2))


if __name__ == "__main__":
    main()
