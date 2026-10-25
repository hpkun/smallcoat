# 多功能射频任务与组件表征原型

本项目提供任务轻量化表征、公共 DAG 模板、32 个软件化波形组件的第一版仿真标定，以及 12 个核心组件的真实数据处理参考实现。12 个运行组件均采用统一的数据包、端口、运行测量和流式处理接口；当前算法使用 NumPy 在 CPU 上执行，用于功能验证和数据模型标定，不代表 FPGA/DSP 的实测性能。

- [模块说明](task_repr/README.md)：数据结构、标定口径和代码边界。
- [任务样例](examples/TaskDemos/README.md)：CNI、Radar、EW、ISAC 的任务输入。
- [组件静态库](task_repr/data/components.json)：C01-C32 的功能与资源等级。
- [能力映射表](task_repr/data/capability_map.json)：任务 capability 到 12 个运行组件的候选关系及匹配条件。
- [静态匹配器](task_repr/matching/matcher.py)：按任务 JSON 生成组件绑定、连接关系及未满足约束。
- [运行接口说明](task_repr/runtime/README.md)：数据包、端口、组件调用方法及测量口径。
- [OPS/OCS 规格](task_repr/SPECIFICATIONS.md)：版本化公共协议、组件接口与现有任务格式的对应关系。
- [C08 运行代码](task_repr/component_models/common/fft_component.py)：`FFTComponent(RuntimeComponent)`，也是公共运行模板。
- [DAG 执行器](task_executor/README.md)：消费静态匹配结果，按 OPS/OCS 校验并串联运行组件。

## 已实现的 12 个核心运行组件

| CID | 组件 | 代码位置 |
| --- | --- | --- |
| C03 | DDC/DUC 数字上下变频 | `task_repr/component_models/common/ddc_component.py` |
| C05 | Digital Filter 数字滤波 | `task_repr/component_models/common/digital_filter_component.py` |
| C07 | Synchronizer 同步 | `task_repr/component_models/common/synchronizer_component.py` |
| C08 | FFT/IFFT | `task_repr/component_models/common/fft_component.py` |
| C09 | PFB Channelizer 数字信道化 | `task_repr/component_models/common/channelizer_component.py` |
| C14 | PSK/QAM 调制解调 | `task_repr/component_models/cni/psk_qam_component.py` |
| C15 | OFDM 处理 | `task_repr/component_models/cni/ofdm_component.py` |
| C17 | FEC 编解码（卷积码参考实现） | `task_repr/component_models/cni/fec_component.py` |
| C21 | Pulse Compression 脉冲压缩 | `task_repr/component_models/radar/pulse_compression_component.py` |
| C23 | CA-CFAR 检测 | `task_repr/component_models/radar/cfar_component.py` |
| C27 | Spectrum Sensing 频谱感知 | `task_repr/component_models/ew/spectrum_sensing_component.py` |
| C32 | DRFM/Jamming 数字射频存储与干扰波形仿真 | `task_repr/component_models/ew/drfm_component.py` |

通过运行注册表可以统一创建组件：

```python
from task_repr.component_models.runtime_registry import create_runtime_component

component = create_runtime_component("C08", fft_size=1024)
```

组件统一提供 `process(packet)`、`push(packet)` 和 `flush()` 接口。`process` 用于独立数据块，`push` 用于连续一维数据流，`flush` 用于结束流处理；所有调用都会通过 `RuntimeMetrics` 记录耗时、吞吐、输入输出数量、缓冲区峰值和错误次数。

## 能力映射表

`task_repr/data/capability_map.json` 使用完整的 capability ID 精确查找候选 CID，并保留每个组件的匹配条件。一个能力以后可以对应多个候选组件；找不到明确条目时返回空结果，不根据名称中的 `fft` 等片段猜测。映射表覆盖当前 12 个运行组件，例如 `digital_down_conversion → C03`、`ofdm_fft_transform → C08`。后者仍须确认 OFDM 节点的符号输入和资源栅格输出与 C08 的时域输入和频域输出是否兼容。`mti_mtd_fft` 没有映射到 C08；其功能对应 C22，而 C22 目前没有运行实现。

```python
from task_repr import default_capability_map

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

`task_repr.models` 存放任务、组件和资源数据结构，`task_repr.catalog` 读取组件定义，`task_repr.template` 处理任务公共骨架，`task_repr.resource` 提供资源估计入口。资源估算模型与运行组件保持分离：当前资源估算注册表仍只注册 C08 的 `FFTModel`，其余组件使用静态等级回退；真实数据处理通过 `runtime_registry.py` 创建 12 个运行组件。当前运行实现对应本机 NumPy CPU 参考模型。
