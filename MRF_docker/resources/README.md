# 资源规划

`ResourcePlanner.plan(task, match, capacity, used)` 接收已完整静态匹配的旧版 `TaskRepresentation` 或业务细化后的 `SpecificationTask`，以及同一任务的 `MATCHED` 组件绑定。`capacity` 是 `HardwareProfile`，`used` 是当前已占用的 `ResourceUsage`；规划只读取资源状态，不预留或启动实例。

规划器按绑定的 CID 和参数调用 `ResourceEstimator`，生成各节点资源画像。当前 DAG 执行器在 CPU 上串行运行，因此 CPU 和带宽按最大单节点压力、内存按所有组件实例之和估算。数值仍采用 0–100 归一化仿真等级；DSP/FPGA 画像只供参考，尚不代表实际设备分配。只有 C08 有参数化工作量模型，未标定的组件运行时间保持 `null`。

新规格任务的时限来自 `Q.deadline_ms` 与 `X.observed_at_ms`，或 `X.remaining_deadline_ms`。业务持续时间 `Q.duration_ms` 不是完成时限。规格节点没有 `estimated_runtime_ms` 时，规划器只使用组件资源画像中明确给出的 `runtime_ms`，不会生成默认运行时间。结果状态含义：

| 状态 | 含义 |
| --- | --- |
| `FEASIBLE_ESTIMATE` | 归一化容量与有明确数据的串行时间估计均通过；仍非实测 QoS 保证 |
| `UNVERIFIED_TIME` | 容量未超限，但缺少组件运行时间标定或任务完成时限 |
| `INFEASIBLE` | 容量超限、时限已过，或明确的运行时间估计超过时限 |

```python
from matching import match_task
from resources.planner import ResourcePlanner, ResourceUsage
from task_repr import parse_external_task, refine_business_task

refinement = refine_business_task(parse_external_task(raw_request))
match = match_task(refinement)
if refinement.task is not None and match.status == "MATCHED":
    plan = ResourcePlanner().plan(refinement.task, match, used=ResourceUsage(cpu=5))
```

`TaskService.plan()` 和现有 `/api/v1/plans` 入口也可接收 `external_rf_task.v1` 或已解析的 `rf_task_representation.v1`。未完成细化或静态匹配时返回 `UNMATCHED` 及匹配诊断；匹配完成后使用调度器的当前资源快照生成只读规划。新规格任务返回 `UNVERIFIED_TIME` 时，不进入现有实例提交路径。旧版任务的资源规划和调度接口保持原行为。

## 物理资源画像

`plan_physical(task, match, hardware, measurements, state)` 是独立的物理单位规划入口。它不把上述 0–100 仿真压力换算成硬件占用。`hardware` 描述一台指定设备的容量和特性；必传的 `state` 是 `PhysicalResourceState`，包含同一设备 ID 和当前物理占用，设备 ID 不符时拒绝规划。不能直接传入仿真调度器的 `snapshot()["used"]`。结果中的 `unit_system` 为 `physical`，旧入口为 `normalized_simulation`。

| 字段 | 单位与口径 |
| --- | --- |
| `cpu_millicores` | CPU 可用或占用的毫核；1000 为一个逻辑核的 CPU 时间份额 |
| `memory_bytes` | 常驻实例内存，字节 |
| `bandwidth_bytes_per_s` | 已定义接口的持续吞吐，字节/秒；需由测量方说明统计范围 |
| `dsp_units`、`fpga_units` | 对应设备自定义的可分配处理单元；容量与需求须使用同一设备的同一单位 |
| `runtime_p95_ms`、`setup_p95_ms` | 单次调用运行和准备时间的第 95 百分位，毫秒；两者逐节点相加是保守的串行规划代理，不是整条任务的实测 P95 |
| `input_bytes`、`output_bytes` | 单次调用的输入、输出数据量，字节 |

`ComponentMeasurement` 以 `cid`、`implementation_id`、`processor`、`hardware_id`、`required_features` 和**完整的匹配参数**识别一条标定。`processor` 只能是 `cpu`、`dsp` 或 `fpga`；对应处理器用量必须大于零，其他加速器用量为零。每条记录还包含 `sample_count` 和非空 `source`，用于追溯原始测量。加载器检查字段、单位字段名、数值范围与处理器用量一致性；`source` 的真实性仍需由实验记录和人工审核保证。

内置 [measurements.json](data/measurements.json) 故意为空：项目尚无可复现的硬件实测记录。标定时应分别固定设备版本、实现版本及参数，记录原始运行日志、样本数、CPU/内存/接口吞吐和 DSP/FPGA 占用，再将统计值录入 `profiles`。不同处理器上的同一 CID 应分别标定；`primary_resources` 与 DSP/FPGA 仿真等级不足以证明实现适配，也不能作为物理占用。当前仅 C08 提供参数化**复杂度及最小输入数据量**模型，其结果不是执行时间或设备容量标定。

```python
from resources import (PhysicalResourceState, load_measurements,
                       load_physical_hardware)
from resources.planner import ResourcePlanner

hardware = load_physical_hardware("hardware.json")
measurements = load_measurements("measurements.json")
state = PhysicalResourceState(hardware.hardware_id, current_physical_usage)
plan = ResourcePlanner().plan_physical(refinement.task, match, hardware, measurements, state)
```

这里的 `current_physical_usage` 必须由设备监控提供 `PhysicalUsage`，不能默认视为零占用。

硬件 JSON 使用 `schema_version: 1`、`hardware_id`、`features` 和 `capacity`；`capacity` 必须完整包含表中五个资源字段。测量 JSON 使用 `schema_version: 1` 和 `profiles` 数组；每条记录包含上述身份与统计字段，并以同样的五字段对象填写 `demand`。只有硬件 ID、特性及绑定参数都符合的记录才进入候选集。缺少任一节点标定时返回 `NEEDS_CALIBRATION`；容量或时限超限返回 `INFEASIBLE`；没有完成时限返回 `UNVERIFIED_TIME`；通过检查时返回 `FEASIBLE_MEASURED_ESTIMATE`。物理规划目前只读，不进入现有归一化实例调度器，也不证明 DSP/FPGA 实现已可实际部署。

## 多节点初始分配

`resources.placement.PlacementPlanner` 从完整 `MATCHED` DAG、处理节点容量、组件实测画像、定向链路和每条业务边的 `EdgeTransfer` 契约生成两层映射：功能节点与 CID 对应逻辑 `container_unit_id`，运行单元再对应 `processing_node_id`。指定 `fixed_nodes={功能节点ID: 处理节点ID}` 时检查固定组合；省略时枚举兼容实现和处理节点，按依赖完成时间选出最快可行方案。资源和链路容量在搜索中剪枝。频谱分析样例 C03→C05→C27 的可运行验收见 `tests/test_placement_scheduler.py`；其中测量值是测试夹具，不是项目提供的硬件标定。

跨节点边必须有 `bytes_per_call` 和 `reserved_bytes_per_s`。链路按方向给出 `capacity_bytes_per_s`、`latency_ms`；传输时间为链路时延加字节数除以预留速率，同节点传输按零处理。每条跨节点边还检查数据量不超过上游实测输出和下游实测输入。分支 DAG 先检查全部节点、边及无环性，再计算所有前驱到达后的最早启动时刻；资源预留按实例的五维需求逐节点累加，链路按跨节点边预留速率累加。完成时间是这套保守模型的估计，不等于整条任务实测 P95。

缺少组件标定返回 `NEEDS_CALIBRATION`，缺边契约返回 `NEEDS_TRANSFER_PROFILE`，容量、链路或时限不满足返回 `INFEASIBLE`，未给完成时限返回 `UNVERIFIED_TIME`。`max_assignments` 限制部分分配尝试次数；达到上限后返回 `SEARCH_LIMIT`，不允许调度器将其提交为已验证方案。`match_task(..., max_solutions=N)` 可向分配器传多组经过完整静态连接验证的 CID 绑定；候选枚举不完整时同样返回 `SEARCH_LIMIT`。默认 `match_task()` 只返回一组绑定。

```python
from resources.placement import PlacementPlanner
from scheduler import TaskScheduler

# processing_nodes、network_links、measurements 和 edge_transfers 均来自同一实验环境。
placement = PlacementPlanner(processing_nodes, network_links, measurements, edge_transfers)
scheduler = TaskScheduler(placement_planner=placement)
fixed_nodes = {node.node_id: "processing-node-1" for node in refinement.task.G.nodes}
preview = scheduler.preview_placement(refinement.task, match, fixed_nodes)
if preview.status == "FEASIBLE_ESTIMATE":
    admitted = scheduler.submit_placement(refinement.task, match, fixed_nodes)
    scheduler.complete_placement(admitted.task_id)
```

`preview_placement()` 不占用容量，`submit_placement()` 在锁内重新检查资源状态。`container_unit_id` 是逻辑运行单元编号，实际容器创建和 DSP/FPGA 执行适配仍需部署层实现。
