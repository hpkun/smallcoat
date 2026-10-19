# 多功能射频任务与组件表征原型

本项目提供任务轻量化表征、公共 DAG 模板、32 个软件化波形组件的第一版仿真标定，以及 12 个核心组件的真实数据处理参考实现。12 个运行组件均采用统一的数据包、端口、运行测量和流式处理接口；当前算法使用 NumPy 在 CPU 上执行，用于功能验证和数据模型标定，不代表 FPGA/DSP 的实测性能。

- [模块说明](task_repr/README.md)：数据结构、标定口径和代码边界。
- [任务样例](examples/README.md)：CNI、Radar、EW、ISAC 的任务输入。
- [组件静态库](task_repr/data/components.json)：C01-C32 的功能与资源等级。
- [运行接口说明](task_repr/runtime/README.md)：数据包、端口、组件调用方法及测量口径。
- [C08 运行代码](task_repr/component_models/common/fft_component.py)：`FFTComponent(RuntimeComponent)`，也是公共运行模板。

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

在项目根目录验证：

```powershell
python -m pip install -r requirements.txt
python -m examples.fft_runtime_demo
python -m unittest discover -s tests -v
python -m task_repr examples/radar_search_task.json
```

`task_repr.models` 存放任务、组件和资源数据结构，`task_repr.catalog` 读取组件定义，`task_repr.template` 处理任务公共骨架，`task_repr.resource` 提供资源估计入口。资源估算模型与运行组件保持分离：当前资源估算注册表仍只注册 C08 的 `FFTModel`，其余组件使用静态等级回退；真实数据处理通过 `runtime_registry.py` 创建 12 个运行组件。当前运行实现对应本机 NumPy CPU 参考模型。
