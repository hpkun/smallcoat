# 多功能射频任务与波形组件调度原型

本项目提供任务轻量化表征、公共 DAG 模板、组件静态匹配、归一化资源规划和实例生命周期仿真。原有 32 个软件化波形组件的静态定义、OPS/OCS 规范及 12 个真实数据处理参考实现均保留。运行组件使用 NumPy 在 CPU 上执行；资源规划和实例状态属于仿真，不代表 FPGA/DSP 实测性能或真实 Docker 部署。

- [任务表征](task_repr/README.md)：任务模型、解析与公共模板。
- [波形组件](components/README.md)：组件库、协议规范与运行实现。
- [资源规划](resources/README.md)：归一化容量与时限估算。
- [调度仿真](scheduler/README.md)：实例状态、复用与释放。
- [任务样例](examples/TaskDemos/README.md)：CNI、Radar、EW、ISAC 的任务输入。
- [组件静态库](components/data/components.json)：C01-C32 的功能与资源等级。
- [能力映射表](components/data/capability_map.json)：任务 capability 到 12 个运行组件的候选关系及匹配条件。
- [静态匹配器](matching/matcher.py)：按任务 JSON 生成组件绑定、连接关系及未满足约束。
- [运行接口说明](components/runtime/README.md)：数据包、端口、组件调用方法及测量口径。
- [OPS/OCS 规格](components/SPECIFICATIONS.md)：版本化公共协议、组件接口与现有任务格式的对应关系。
- [C08 运行代码](components/implementations/common/fft_component.py)：`FFTComponent(RuntimeComponent)`，也是公共运行模板。
- [DAG 执行器](task_executor/README.md)：消费静态匹配结果，按 OPS/OCS 校验并串联运行组件。
- [后端框架](backend/README.md)：独立的 HTTP API、应用服务与任务/资源 JSON 数据访问层。

## 技术方案模块对应

| 核心模块 | 代码包 | 当前交付的数据 |
| --- | --- | --- |
| 多功能任务轻量化表征 | `task_repr/` | `TaskRepresentation` 与任务功能 DAG |
| 波形组件匹配与组合 | `matching/`，依赖 `components/` | `StaticMatchResult` 与组件绑定、连接关系 |
| 异构资源优化分配 | `resources/` | `ResourcePlan` 与归一化容量估算 |
| 容器动态调度与任务执行 | `scheduler/`、`task_executor/` | 仿真实例决策与进程内 `ExecutionResult` |

`components/` 保存组件库和 OPS/OCS 等基础支撑数据，`backend/` 提供应用服务与接口。当前调度是单进程仿真；实际容器部署、异构资源测量和并发重调度需要在这些边界内继续实现。

## 已实现的 12 个核心运行组件

| CID | 组件 | 代码位置 |
| --- | --- | --- |
| C03 | DDC/DUC 数字上下变频 | `components/implementations/common/ddc_component.py` |
| C05 | Digital Filter 数字滤波 | `components/implementations/common/digital_filter_component.py` |
| C07 | Synchronizer 同步 | `components/implementations/common/synchronizer_component.py` |
| C08 | FFT/IFFT | `components/implementations/common/fft_component.py` |
| C09 | PFB Channelizer 数字信道化 | `components/implementations/common/channelizer_component.py` |
| C14 | PSK/QAM 调制解调 | `components/implementations/cni/psk_qam_component.py` |
| C15 | OFDM 处理 | `components/implementations/cni/ofdm_component.py` |
| C17 | FEC 编解码（卷积码参考实现） | `components/implementations/cni/fec_component.py` |
| C21 | Pulse Compression 脉冲压缩 | `components/implementations/radar/pulse_compression_component.py` |
| C23 | CA-CFAR 检测 | `components/implementations/radar/cfar_component.py` |
| C27 | Spectrum Sensing 频谱感知 | `components/implementations/ew/spectrum_sensing_component.py` |
| C32 | DRFM/Jamming 数字射频存储与干扰波形仿真 | `components/implementations/ew/drfm_component.py` |

通过运行注册表可以统一创建组件：

```python
from components.implementations.runtime_registry import create_runtime_component

component = create_runtime_component("C08", fft_size=1024)
```

组件统一提供 `process(packet)`、`push(packet)` 和 `flush()` 接口。`process` 用于独立数据块，`push` 用于连续一维数据流，`flush` 用于结束流处理；所有调用都会通过 `RuntimeMetrics` 记录耗时、吞吐、输入输出数量、缓冲区峰值和错误次数。

## 能力映射表

`components/data/capability_map.json` 使用完整的 capability ID 精确查找候选 CID，并保留每个组件的匹配条件。一个能力以后可以对应多个候选组件；找不到明确条目时返回空结果，不根据名称中的 `fft` 等片段猜测。映射表覆盖当前 12 个运行组件，例如 `digital_down_conversion → C03`、`ofdm_fft_transform → C08`。后者仍须确认 OFDM 节点的符号输入和资源栅格输出与 C08 的时域输入和频域输出是否兼容。`mti_mtd_fft` 没有映射到 C08；其功能对应 C22，而 C22 目前没有运行实现。

```python
from components import default_capability_map

candidates = default_capability_map().candidates_for("digital_down_conversion")
for candidate in candidates:
    print(candidate.cid, candidate.match_condition)
```

查询只返回候选；静态匹配还需检查节点参数、输入输出数据域和端口格式，以及运行实现，才能输出可行绑定。资源约束属于后续调度阶段。表中 `fir_filter`、`iir_filter`、`known_preamble_frame_synchronization`、`fft`、`ifft`、`ofdm_tx`、`ofdm_rx`、`iq_delay_replay` 是供后续任务模板使用的明确能力 ID。

## 静态匹配

```powershell
python -m examples.static_match_demo
python -m examples.static_match_demo examples/TaskDemos/radar_search_task.json
python -m unittest tests.test_static_matcher -v
```

也可在代码中调用 `match_task(任务JSON路径)`，并用 `.to_dict()` 获取 `task_id`、`status`、`bindings`、`edges` 和 `unresolved`。`edges` 只列出已通过校验的连接；失败的边及原因记录在 `unresolved`。`status` 为 `MATCHED`、`PARTIAL` 或 `UNSUPPORTED`；`PARTIAL` 表示至少发现一个能力候选，但完整任务仍有未解决约束，可能尚无确认的绑定。每条未解决项含节点或边、候选 CID、错误码和具体原因。示例 `signal_analysis_task.json` 应匹配为 C03 → C05 → C27。参数中的数组以 JSON 数组字符串表示，例如 `"preamble": "[1,1,-1,1]"`；复数参考波形可写成 `"reference_waveform": "[\"1+0j\",\"0+1j\"]"`，由参数规则显式解码。可选的 `sample_rate_mhz`、`input_block_length`、`output_block_length` 描述节点输入采样率与块长度；连接时会检查 C03/C09 后的采样率及上下游块长度。`nco_frequency_mhz`、`bandwidth_mhz` 和 `scan_bandwidth_mhz` 会显式换算为组件使用的 Hz。

该结果只验证当前 12 个 Python 组件的功能、配置和已定义的数据契约。未知任务格式会明确报错；时延、可靠性、设备资源容量和容器运行仍需后续阶段验证。

在项目根目录验证：

```powershell
python -m pip install -r requirements.txt
python -m examples.TaskDemos.fft_runtime_demo
python -m unittest discover -s tests -v
python -m task_repr examples/TaskDemos/radar_search_task.json
```

`task_repr` 负责任务描述，`components` 保存组件库、OPS/OCS 和 CPU 参考实现，`matching` 生成组件绑定，`resources` 估算并预留仿真容量，`scheduler` 管理仿真实例状态，`task_executor` 执行已匹配的 DAG，`backend` 提供 API。资源估算注册表目前只为 C08 提供动态 `FFTModel`，其他组件沿用静态等级；容器部署、异构硬件实测与并发重调度尚需后续实现。
