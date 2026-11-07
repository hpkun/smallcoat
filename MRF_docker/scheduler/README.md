# 调度仿真

`TaskScheduler` 保留原有的归一化单机接口 `preview()`、`submit()`、`complete()`，管理 `ACTIVE`、`WARM`、`RELEASED` 仿真实例。预览不修改状态；完成后保留内存，后续相同 CID 与参数可复用实例；内存不足时可逐出无关预热实例。

多处理节点物理分配使用 `placement_planner=PlacementPlanner(...)` 构造调度器，并调用 `preview_placement()`、`submit_placement()`、`complete_placement()` 和 `placement_snapshot()`。提交在同一把锁内重新规划并预留节点和链路容量；仅接受 `FEASIBLE_ESTIMATE`。两个任务同时提交时，后提交者会看到前者的预留。完成默认释放全部容量；`keep_warm=True` 只保留实例内存，`evict_placement_warm()` 可释放。基础占用量可用 `placement_baseline=PlacementState(...)` 提供，须与处理节点和链路清单逐项对应。

分配结果中的 `container_unit_id` 是调度仿真的逻辑运行单元 ID，不代表已创建 Docker 容器。物理分配只接受外部提供的实测组件画像、节点容量和链路契约；当前不会启动进程、加载 DSP/FPGA 固件、传输数据或恢复检查点。现有 `task_executor` 在进程内执行经过静态匹配的 CPU DAG，`TaskService.execute_with_schedule()` 只连接旧版归一化预留路径。
