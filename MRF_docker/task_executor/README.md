# DAG Executor

`DAGExecutor` runs one finite pass through a task DAG or a selected subgraph, currently with C08 FFT/IFFT processing nodes. Bind each processing node to `"C08"` or a C08 `ComponentInstance`; a CID uses the node's `parameters`. An unbound root is an external boundary whose output `DataPacket` is supplied by the caller.

```python
executor = DAGExecutor(task, {"video-fft": "C08"},
                       active_nodes={"video-sync", "video-fft"})
result = executor.execute({"video-sync": packet})
output = result.sink_outputs["video-fft"]
```

The example reads `examples/TaskDemos/cni_video_backhaul_task.json`. `video-sync` is an external boundary; its supplied packet is passed along the existing edge to `video-fft`. Run `python -m examples.dag_fft_demo` from the project root. The result includes execution order, node and sink packets, instances, and per-node metrics. `completed` means the selected subgraph finished; `full_task_completed` remains false when other task nodes were excluded. `executor.status` is `pending`, `running`, `completed`, or `failed`.

Bindings are explicit because task capabilities describe functional needs and do not uniquely identify runtime CIDs. Other CIDs are currently rejected. The current runtime contract is one input and one output `DataPacket` per component, so a node with multiple graph input/output ports needs a dedicated runtime adapter. Execution uses `process()` once per node; streaming, packet batching across DAG nodes, hardware scheduling, and task QoS enforcement are outside this finite pass.
