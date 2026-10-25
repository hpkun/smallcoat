# DAG Executor

`DAGExecutor.from_match()` runs a fully matched task with the normalized parameters returned by the static matcher. It verifies OCS properties and ports, all DAG connections, and each actual packet against OPS, the runtime port, the declared block length, and the expected sample rate. The source map is keyed by root node ID.

```python
import json
from pathlib import Path

from task_executor import DAGExecutor
from task_repr import match_task, represent_task

task = represent_task(json.loads(Path("examples/TaskDemos/signal_analysis_task.json").read_text()))
match = match_task(task)
executor = DAGExecutor.from_match(task, match)  # requires MATCHED
result = executor.execute({"ddc-node": input_packet})
psd = result.sink_outputs["sensing-node"]
```

The example runs C03 DDC, C05 FIR filter and C27 PSD in order. Its declared blocks are 4096, 2048, 2048, then 1024 items; its input is 4 MHz and the outputs retain 2 MHz. PSD uses the input sample rate as the frequency-axis reference.

The explicit constructor remains available for a selected subgraph or an external root boundary. A CID binding in that legacy form uses the node's raw `parameters`; use `from_match()` for OPS/OCS checked execution.

```python
executor = DAGExecutor(task, {"video-fft": "C08"},
                       active_nodes={"video-sync", "video-fft"})
result = executor.execute({"video-sync": packet})
output = result.sink_outputs["video-fft"]
```

The example reads `examples/TaskDemos/cni_video_backhaul_task.json`. `video-sync` is an external boundary; its supplied packet is passed along the existing edge to `video-fft`. Run `python -m examples.dag_fft_demo` from the project root. The result includes execution order, node and sink packets, instances, and per-node metrics. `completed` means the selected subgraph finished; `full_task_completed` remains false when other task nodes were excluded. `executor.status` is `pending`, `running`, `completed`, or `failed`.

The runtime contract is one input and one output `DataPacket` per component. Execution uses `process()` once per node. OPS dimensions are enforced, so the one-dimensional `frequency/complex` protocol excludes C08 batch frames in this DAG path. Streaming, multi-port communication, hardware scheduling, and task QoS enforcement are outside this finite pass.
