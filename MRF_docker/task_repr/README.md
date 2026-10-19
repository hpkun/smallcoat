# 波形任务公共模板原型

本模块把 CNI、Radar、EW、ISAC 任务组织成“统一公共字段 + 类型扩展 + 组件 DAG”。任务轻量化表征不保存 IQ 样本或完整协议报文。新增的运行层通过独立的 `DataPacket` 传递真实数据，当前已实现 C08 FFT/IFFT；详见[运行接口说明](runtime/README.md)。当前还包括单任务表征、严格校验、公共图骨架提取和任务实例恢复，**尚未实现训练数据生成与调度算法**。

## 公共字段和口径

输入 JSON 使用 `schema_version: 2`。输出仍为 `T_rep = {S,Q,R,G,X}`，并带任务 ID 和模式版本。旧版 `schema_version: 1` 输入继续可用，但不具备 v2 的公共模板功能。

| 部分 | 公共字段 | 口径 |
| --- | --- | --- |
| `semantics` → `S` | 任务族、类型、场景、方向、基础优先级、场景阶段、子类型、触发方式、类型扩展 | 基础优先级为非负整数；数值越大优先级越高。`family` 为 `CNI`、`Radar`、`EW`、`ISAC` 之一。 |
| `qos` → `Q` | 到达、持续、周期、绝对截止、最大响应时延、吞吐、可靠性、安全等级、旧版可抢占/迁移/降级/共享标志和重构开销 | 时间统一为同一场景时钟的毫秒；`period_ms` 可省略，表示非周期任务。安全等级为非负序数，具体分级仍需项目统一。第一版仿真使用组件级 `simulation_profile` 的 S/P/X 与 Y/C/N。 |
| `resources` → `R` | 兼容旧任务表示中的 CPU、内存、DSP、FPGA、通信带宽、射频通道与射频带宽 | 这些绝对量字段不是第一版仿真的等级标定值；仿真使用组件级 `resource_levels`，不得由旧值自动换算。 |
| `graph` → `G` | 功能节点、规范化 `role`、能力、端口格式、参数、共享/抢占属性、仿真标定档案、异构执行档案、数据依赖边 | `simulation_profile` 保存第一版等级基线；`execution_profiles` 是后续实现映射的细化数据，不作为第一版等级的替代值。 |
| `state` → `X` | 状态、观测时刻、进度、当前优先级、剩余工作量及已占资源 | `remaining_deadline_ms = max(0, deadline_ms - observed_at_ms)` 派生得到；`remaining_work` 的单位由 `work_unit` 给出。 |

第一版仿真用 `simulation_profile.latency_level` 及资源等级做抽象调度，不用 `estimated_runtime_ms` 推导等级。`execution_profiles[设备]` 的 `runtime_ms + setup_ms`、任务级 `reconfiguration_ms` 属于后续实现映射的细化数据；真实系统须确定准备时间和重构时间是否重叠，避免重复计费。`min_reliability` 和归一化链路质量为 `[0,1]`；状态进度为 `[0,1]`。类型扩展中的频率、数据量、距离等单位直接写在字段名中。

## 第一版仿真标定基线

本基线用于任务-组件-资源调度仿真，**替代此前以 millicores、MiB、DSP/FPGA 数量和 Mbps 直接作为仿真等级的口径**。每个组件可带 `simulation_profile`：

```json
{
  "simulation_profile": {
    "resource_levels": {"cpu": 2, "mem": 3, "dsp": 4, "fpga": 5, "bw": 4},
    "latency_level": "L1",
    "sharing_class": "P",
    "preemption_class": "C"
  }
}
```

CPU、MEM、DSP、FPGA、BW 各取 `0..5`，依次表示基本不占用、很低、较低、中等、高、很高。进入算法时可按 `0,1,2,3,4,5 -> 0,5,10,20,35,50` 映射到归一化资源单位；第一版假定每个节点每类资源容量为 `100`。这些等级是相对资源压力/适配属性，不是设备实测占用。特别是 DSP、FPGA 等级表达候选加速适配，不表示一个组件必须同时占满两类加速器；第一版不拆分 `FFT_CPU`、`FFT_DSP`、`FFT_FPGA` 实现。任务资源画像应由调度时活跃/映射的组件决定，不应把整条 DAG 的所有组件需求机械求和。

时延等级按组件处理时延划分，统一用毫秒描述区间边界：`L0=[0.01,0.1)`、`L1=[0.1,1)`、`L2=[1,5)`、`L3=[5,20)`、`L4=[20,100)`、`L5=[100,+∞)` ms。`L5` 可表示 100 ms 以上或批处理。

组件标定中的 `latency_level` 还可写为范围，例如 `L0-L1` 或 `L2-L3`，表示该组件当前只标定到这一组相邻时延等级，不代表已知某个精确执行时间。范围必须从低到高；`L2-L1` 等逆序值会被拒绝。组件记录中的 `estimated_runtime_ms` 目前为 `null`，因为输入基线只提供时延等级，没有给出可复现的实测毫秒值。

共享等级：`S` 表示适合多任务共享/多实例复用；`P` 表示参数兼容或采用时分后可共享；`X` 表示通常应独占。抢占等级：`Y` 表示可在任务/数据块边界直接抢占；`C` 表示只能在帧、脉冲、CPI 或检查点等安全边界抢占；`N` 表示运行期间不宜抢占。

## 32 个核心组件库

机器可读定义保存在 [`data/components.json`](data/components.json)，包含 C01-C32。每条记录把功能定义表与调度属性表合并，字段包括：

| 字段 | 含义 |
| --- | --- |
| `cid`、`name`、`short_name` | 稳定组件编号、组件名与短名 |
| `function` | 组件功能 |
| `input_formats`、`output_formats` | 输入/输出数据描述；当前是可读的数据契约名称，尚非完整的类型系统或序列化协议定义 |
| `configurable_parameters` | 主要可配置参数名称 |
| `families` | 通用、CNI、Radar、EW、ISAC 等适用领域；查询某任务族时会自动包含“通用”组件 |
| `typical_predecessors`、`typical_successors` | 典型处理链的前驱/后继提示；其中既可能是 CID，也可能是外部接口/功能名称，不等于某个任务的实际 DAG |
| `primary_resources` | 主执行资源提示，不是排他的平台兼容清单 |
| `simulation_profile` | CPU/MEM/DSP/FPGA/BW 等级、时延等级范围、共享等级与抢占等级 |
| `estimated_runtime_ms`、`runtime_status` | 可选实测/估算执行时长及标定状态；当前 32 项均为 `null`/“未标定” |

资源等级继续使用第一版 `0..5` 口径。`DSP=5` 与 `FPGA=5` 表示相对适配/压力等级，不能解释成同时必需或已实测占满。`primary_resources` 也仅为主资源提示；下一阶段应为可多平台实现的组件建立 CPU/DSP/FPGA 候选实现档案和各自的执行时间/资源测量。

可通过 API 查询：

```python
from task_repr import default_component_catalog

catalog = default_component_catalog()
fft = catalog.get("C08")
print(fft.simulation_profile.resource_levels.to_normalized_units())
print([item.cid for item in catalog.by_family("Radar")])
```

组件库加载时会检查字段、CID 格式和唯一性、资源等级范围、时延等级/范围、共享/抢占枚举以及主执行资源。`catalog.to_dict()` 可用于导出实验快照。组件目录当前是仿真基线，不应直接视作经硬件验证的资源画像或可部署容器清单。

## 代码结构与依赖

| 路径 | 职责 |
| --- | --- |
| `data/components.json` | 32 个组件的静态定义，不存数学公式 |
| `models/component.py` | `ComponentDefinition` 与 `ComponentInstance`，区分组件种类和任务中的一次配置 |
| `models/task.py` | 原有 `TaskRepresentation`、功能 DAG 及 JSON 校验；保持 v1/v2 样例兼容 |
| `models/resource.py` | 旧任务绝对资源字段、等级标定和统一 `ResourceProfile` |
| `catalog/component_catalog.py` | 加载、校验、查询静态定义 |
| `component_models/base.py`、`registry.py`、`common/fft.py` | 统一 `estimate()` 接口、CID 注册表和 C08 FFT/IFFT 模型 |
| `component_models/common/fft_component.py` | C08 真实 FFT/IFFT、批处理和流式缓冲，不替代资源估算类 |
| `runtime/` | 公共 DataPacket、RuntimePort、RuntimeMetrics 与 RuntimeComponent，统一调用及逐帧测量 |
| `resource/estimator.py`、`hardware.py` | 动态模型调度入口、静态等级回退和默认容量 100 的硬件画像 |
| `template/task_template.py` | 现有公共 DAG 骨架与任务变体提取/恢复 |

`ComponentInstance` 只持有 `cid`、任务/实例 ID、运行参数和可选资源画像；它不会复制或改变静态定义。当前任务样例中的 `FunctionNode` 仍是任务功能需求节点，**尚未自动绑定组件 CID**，因此 `TaskRepresentation` 不会隐式生成组件实例。需要在后续组件匹配阶段确定 CID 和输入输出契约后，才能完成该绑定。

`ResourceEstimator.estimate(definition, parameters, hardware)` 先查注册表。C08 走 `FFTModel`；其余 31 个组件仍将五维等级映射为归一化单位，并保留原时延等级。五维值仅是**相对资源压力画像**，不是五类硬件同时分配的用量；尤其不能把 DSP 和 FPGA 高等级同时计入实际占用。

C08 实例必须传入整数 `fft_size >= 2`；可选 `batch_size`（默认 1）、`bytes_per_sample`（默认 8），以及以**样点/秒**计的总输入 `sample_rate`。返回 `workload = batch_size × fft_size × log2(fft_size)`（无量纲复杂度代理）、`memory_bytes = batch_size × fft_size × bytes_per_sample`（输入块缓存基线）；给定 `sample_rate` 时，`bandwidth_bytes_per_s = sample_rate × bytes_per_sample` 表示单向输入流数据率，未给出时为 `None`。这些动态量不覆盖 `memory` 和 `bandwidth` 的归一化压力值，也不包括 FFT 临时缓存、输出及内部数据搬移。`hardware` 当前尚未标定性能，故 C08 的 `runtime_ms` 仍为 `None`，`latency_level` 仍沿用 L1。不同 FFT 规模会改变动态工作量和缓存量，但不会凭公式虚构精确时延或真实设备占用。

```python
from task_repr import ComponentInstance, ResourceEstimator, default_component_catalog

catalog = default_component_catalog()
instance = ComponentInstance("search-fft", "C08", "search", {"fft_size": 1024})
profile = ResourceEstimator().estimate(catalog.get(instance.cid), instance.parameters)
assert profile.workload == 10240
```

后续模型文件拟按 C03/C05/C07/C09 放入 `component_models/common/`，C14/C15/C17 放入 `cni/`，C21/C23 放入 `radar/`，C27/C32 放入 `ew/`。目前不添加其余模型的占位公式。调度器仍应作为未来的独立模块，不在 `task_repr` 内创建。

当前旧任务 JSON 尚未全部填写 `simulation_profile`；字段缺失表示“未标定”，不能用旧物理量自动换算补齐。节点容量 `100` 是第一版场景假设，资源等级到归一化单位的映射是初始基线，后续应通过真实任务剖面和平台测量校准。任务级原始资源字段和 `execution_profiles` 暂作兼容/细化信息，不应混同为本基线的等级值。

旧版 `qos.scheduling.preemptible` 表示任务是否允许暂停，`can_preempt` 表示是否有权请求抢占低优先级任务；它们仅为兼容字段，不等同于第一版组件级抢占等级。仿真使用 `simulation_profile.preemption_class` 表达 Y/C/N；如需描述“谁抢占谁”，还须提供单独的任务间关系规则。

## 类型扩展

每类扩展是严格字段集，缺失或出现其他字段都会报错。它们是初始建模字段，不代表真实射频任务的完整参数目录。

| 任务族 | `semantics.extension` 必填字段 |
| --- | --- |
| CNI | `data_rate_mbps`、`modulation_coding`、`link_quality`、`payload_mib` |
| Radar | `pri_us`、`pulse_bandwidth_mhz`、`fft_size`、`mode` |
| EW | `threat_level`、`response_limit_ms`、`intercept_bandwidth_mhz`、`jamming_mode` |
| ISAC | `communication_rate_mbps`、`sensing_resolution_m`、`shared_component_ratio` |

其中 `link_quality` 和 `shared_component_ratio` 取 `[0,1]`；`fft_size`、`threat_level` 为正整数。`shared_component_ratio` 当前是输入的估计值，并非从当前任务 DAG 自动计算：跨任务共享比例需要先确定比较集合和分母。真实数据进入训练前，应明确链路质量归一化方法、威胁等级、安全等级、优先级和共享比例的业务定义。

## 公共骨架与任务变体

`build_template()` 用规范化 `role` 对齐多个 v2 任务。只有同一角色的能力、输入端口和输出端口格式完全一致，才允许对齐；不同任务复用同一角色却定义了不同端口契约时会报错。公共节点是**所有输入任务**共有的角色，公共边是这些节点之间**所有任务**共有的依赖。节点 ID、参数、共享/抢占标志、各设备执行档案和非公共分支保留在每条任务的 `variant` 中。

例如四类任务都包含 `rf_receive → ddc`，它可成为公共骨架；ISAC 中从 DDC 分出的目标检测和通信解调仍是其任务变体。`restore_task()` 能在内存中恢复完整任务 DAG。本算法是**给定规范化角色后的严格图交集**，不是任意两个 DAG 的最大公共子图求解；模板结果取决于参与比较的任务集合，也不自动判断两个组件实例是否真的可以并发共享。组件的物理共享还需考虑参数兼容、容量、时间重叠和资源节点位置。

## Radar/EW 示例

Radar 的典型任务包括搜索、跟踪、SAR 成像和 GMTI：搜索/跟踪通常围绕 PRI/CPI 数据流做脉冲压缩、MTI/MTD、CFAR 与状态更新；SAR 更关注长时间相干积累、运动补偿和二维成像吞吐；GMTI 还需考虑杂波抑制及 STAP 的高算力/存储带宽。EW 可拆成 ESM（截获、信道化、参数分选、测向和威胁识别）、RWR（告警链）及 ECM（干扰策略、波形/DRFM 生成和发射）。它们应分别配置任务扩展与组件 DAG，不宜被当作一个固定处理链。

[Radar 目标搜索示例](../examples/radar_search_task.json)表达一条由 PRI 驱动、包含 CPI 相干窗处理的搜索链：

\[
\mathrm{RF接收}\rightarrow\mathrm{DDC}\rightarrow\mathrm{LFM脉冲压缩}
\rightarrow\mathrm{MTD/FFT}\rightarrow\mathrm{CFAR检测}
\]

它把 `pri_us`、脉冲带宽和 FFT 规模放在 Radar 扩展字段，把 FPGA/DSP 映射、执行时间和带宽放在每个组件的 `execution_profiles` 中。调度器可以比较同一组件放在 DSP 或 FPGA 上的代价，而不是把整个 Radar 任务粗略标成“使用 FPGA”。

本例约定 `period_ms=1`、`deadline_ms=1` 表示每个 PRI 到达后，流水处理要在下一个周期前跟上；`coherent_pulses=64` 则表示 MTD 使用 64 脉冲相干窗。PRI 级处理预算与首脉冲到检测输出的端到端时延不是同一个口径，后者还包含相干观测窗。当前模型没有逐节点截止时间/吞吐字段，平台验收时应补充这类口径或拆成 PRI 处理任务与 CPI 检测任务分别调度。

[EW 威胁响应示例](../examples/ew_threat_response_task.json)表达：

\[
\mathrm{宽带接收}\rightarrow\mathrm{数字信道化}\rightarrow\mathrm{频谱感知}
\rightarrow\mathrm{参数/DOA估计}\rightarrow\mathrm{威胁融合}
\rightarrow\mathrm{干扰激活}\rightarrow\mathrm{DRFM}\rightarrow\mathrm{发射}
\]

该示例按假设的 `0-10` 优先级刻度取最高等级 `10`，设置 `can_preempt=true`、`preemptible=false`，并将 `0.5 ms` 写入 `max_response_ms` 和质量约束；`hot_standby`、`waveform_preloaded` 表示容器和波形已热备/预加载，不能把这个数值解释为冷启动容器的保证时延。示例中的 `can_preempt` 表示可请求抢占低优先级任务，并不代表可以无条件夺取正在使用的射频链路；资源释放边界、安全策略和波形切换必须由平台调度器明确。真实系统还需要用硬件链路、DMA、驱动和射频切换实测校准 `runtime_ms`、`setup_ms` 与 `reconfiguration_ms`。

## 运行

完整任务实例目录见 [`examples/README.md`](../examples/README.md)，当前包含战术话音、视频回传、IFF、Radar Search、SAR、ESM、ECM、威胁响应和 ISAC 实例；`voice_tx_task.json` 保留为 v1 兼容性样例。

[ISAC v2 示例](../examples/isac_joint_task.json)展示公共前段加双分支 DAG；[CNI v1 示例](../examples/voice_tx_task.json)展示旧版输入。**所有示例中的时限、射频、带宽和资源数值均为演示值，不代表实测指标。** 在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m examples.fft_runtime_demo
python -m task_repr examples/isac_joint_task.json
python -m task_repr examples/radar_search_task.json
python -m task_repr examples/ew_threat_response_task.json
python -m unittest discover -s tests -v
```

在 Python 中批量解析任务后构建模板：

```python
from task_repr import build_template, represent_task, restore_task

tasks = [represent_task(raw) for raw in raw_task_dicts]
template = build_template(tasks)
print(template.to_json())
original = restore_task(template, tasks[0].task_id)
```

v2 每个任务的节点 `role` 必须唯一，组件 DAG 必须无环。每条边的源/目标端口必须存在且格式一致，一个输入端口最多有一个上游来源；需要汇聚时，应显式建汇聚功能节点。流式处理的内部反馈应封装在功能节点内。除显式标记为 `raw_artifacts` 的原始材料外，未知字段会触发 `ValidationError`，以免新增的调度信息被静默丢弃。

下一步应收集真实 CNI、Radar、EW、ISAC 任务剖面，建立稳定的角色、能力、端口格式及单位目录，校准设备执行时间和资源开销，并检验轻量表示是否保持组件选择与调度结果。之后再定义场景采样、任务满足率和资源利用率的统计窗口与分母，用于数据生成和验收。
