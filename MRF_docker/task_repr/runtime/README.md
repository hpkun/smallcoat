# 12 个核心组件运行接口

C08 的真实算法位于 [`../component_models/common/fft_component.py`](../component_models/common/fft_component.py)，类名为 `FFTComponent(RuntimeComponent)`。它按照初始化、参数校验、动态配置、`_execute()`、`process()`、`push()` 六部分组织，是 12 个运行组件共用接口的参考模板。其余 11 个组件分别位于 `common/`、`cni/`、`radar/` 和 `ew/` 子目录，并通过 `runtime_registry.py` 统一创建。公共框架负责计时和统计，算法部分调用 NumPy 在 CPU 上执行。

同目录的 `fft.py / FFTModel` 仍负责工作量与资源估算。`ResourceEstimator` 的注册表继续指向 `FFTModel`；执行真实数据时显式创建 `FFTComponent`。运行指标暂不自动转换成 CPU、DSP、FPGA 占用，也不改写 32 个组件的仿真等级。

## 运行组件清单

| CID | 类名 | 参考算法范围 |
| --- | --- | --- |
| C03 | `DDCComponent` | NCO 混频、FIR、抽取/插值 |
| C05 | `DigitalFilterComponent` | FIR/IIR、低通/高通/带通/带阻、跨块状态 |
| C07 | `SynchronizerComponent` | 已知前导相关、定时偏移和频偏校正 |
| C08 | `FFTComponent` | FFT/IFFT、窗函数、shift、批处理和重叠分帧 |
| C09 | `ChannelizerComponent` | 原型滤波器组和多路数字信道化 |
| C14 | `PSKQAMComponent` | BPSK/QPSK/QAM 调制、硬判决和软 LLR |
| C15 | `OFDMComponent` | 子载波、导频、循环前缀、FFT/IFFT 收发 |
| C17 | `FECComponent` | 卷积码编码和硬/软 Viterbi 译码 |
| C21 | `PulseCompressionComponent` | 时域/FFT 域匹配滤波和距离压缩 |
| C23 | `CFARComponent` | 一维/二维 CA-CFAR 检测 |
| C27 | `SpectrumSensingComponent` | FFT PSD、Welch/滑动平均和活动频段检测 |
| C32 | `DRFMComponent` | IQ 缓存、延迟、幅相/频移变换和离线重放 |

这些组件是可运行的 NumPy CPU 参考实现，不是硬件驱动、FPGA bitstream 或实时射频闭环。C17 当前用卷积码验证编码/译码链路，C23 当前用 CA-CFAR 验证检测链路，C32 限定为离线/闭环 DRFM 仿真。后续接入硬件时，应在同一 `DataPacket` 和运行指标契约下替换算法后端，并用实测数据重新标定资源需求。

## 统一调用方式

```python
import numpy as np
from task_repr.runtime import DataPacket
from task_repr.component_models.runtime_registry import create_runtime_component

component = create_runtime_component("C03", nco_frequency=100_000)
packet = DataPacket(
    np.exp(2j * np.pi * 100_000 * np.arange(1024) / 1_000_000),
    sample_rate=1_000_000,
    center_frequency=100e6,
    timestamp=0.0,
    domain="time",
)
result = component.process(packet)
print(result.samples.shape, component.metrics.latency_ms)
```

所有运行组件都使用 `DataPacket` 承载数据和上下文，使用 `RuntimePort` 校验输入输出域、数据类型、维度和计数单位。算法主体放在 `_execute()`；公共基类负责 `process`、`push`、`flush` 的调用计时、错误统计、吞吐计算和缓冲观测。组件专属指标放入输出包的组件元数据和 `RuntimeMetrics.component_specific_metrics`。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `packet.py` | `DataPacket`：统一数据载荷和信号上下文 |
| `ports.py` | `RuntimePort`：输入输出的域、类型、维度和计数单位契约 |
| `metrics.py` | `RuntimeMetrics`：最近调用、累计计数与派生吞吐率 |
| `base.py` | `RuntimeComponent`：公共调用测量、逐帧测量和缓冲观测接口 |

## 数据包与端口

```python
DataPacket(
    samples=iq,                 # NumPy 数组；C08 要求复数 IQ/频谱
    sample_rate=1_024_000,       # 样点/秒，必须为正数
    center_frequency=100e6,     # Hz，射频中心频率
    timestamp=0.0,             # 秒，统一场景时钟下首帧首点时刻
    domain="time",            # C08 使用 time 或 frequency
    metadata={"channel": 1},   # 可 JSON 序列化的描述信息
)
```

`samples` 支持一维流 `(M,)` 或二维批帧 `(K,N)`；最后一轴为帧内点数。包会复制数据、设置数组只读，并复制元数据，避免调用方原数组被修改后影响已构建的数据包。元数据顶层只读，嵌套对象也应按只读约定使用。`item_count` 是数组元素总数，`nbytes` 只计数组有效载荷，不含 Python 对象与元数据。包的创建、输入复制发生在 `process/push` 之前，不计入这两个接口的调用耗时。

频域输出仍携带原时域采样率、中心频率和观测时间，不能把采样率理解成频谱包的到达频率。FFT 频点间距为 `sample_rate / fft_size`，频点顺序由 `shift` 决定。`metadata["fft"]` 保存本次配置、实际批量、逐帧时间戳 `frame_timestamps`、频点间距与归一化方式，原始元数据的其他字段保留；`fft` 键为 C08 保留字段。

FFT 输入端口接受 `time` 域，输出 `frequency` 域；IFFT 相反。两者均要求复数数组，拒绝把实数、字符串或错误域的数据静默转换成 IQ。`output_port.can_connect_to(input_port)` 检查域、dtype 类别、维度和计数单位兼容；实际包仍由端口校验，具体帧长由组件校验。当前通过直接调用 `downstream.process(upstream_packet)` 连接，无异步队列或自动 DAG 执行器。

## 参数与处理规则

| 参数 | 含义与限制 |
| --- | --- |
| `fft_size` | 每帧点数，整数且至少为 2；支持非 2 的幂 |
| `mode` | `fft` 或 `ifft` |
| `window` | `None` / `rect` / `hann` / `hamming` / `blackman`；均在变换前逐点相乘 |
| `overlap` | FFT 流式帧重叠比例，范围 `[0,1)`，必须产生整数步长 |
| `hop_size` | 相邻帧起点间隔，整数且在 `1..fft_size`；与 overlap 同时给出时须一致 |
| `shift` | FFT 对输出做 `fftshift`；IFFT 对输入做 `ifftshift` |
| `batch_size` | 一个输出包的最大帧数，正整数；流式 push 等到完整批次再输出 |

默认无窗、无重叠、无 shift，批量为 1。采用 NumPy 的 `backward` 归一化：FFT 不缩放，IFFT 乘 `1/N`。窗函数使用 NumPy 的对称窗；不自动做幅度补偿。IFFT 的窗作用于输入频谱，无窗时才对应普通 FFT/IFFT 的直接互逆。当前 IFFT 流按完整频谱帧消费，禁止重叠，不提供 overlap-add 重建。

`process(packet) -> DataPacket` 接受 `(N,)` 或 `(K,N)`，要求 `1 <= K <= batch_size`。不截断、不补零、不使用流缓存；二维输入应已按所配置的 hop 完成分帧，逐帧时间戳为 `timestamp + k*hop_size/sample_rate`。IFFT 也按自己的 hop 解释输入，不自动继承上游重叠分析配置。

`push(packet) -> list[DataPacket]` 接受任意长度的一维连续流。它缓存样点，以 N 点组帧、按 hop 滑动，按 batch 输出。batch 为 1 时每包为 `(N,)`，大于 1 时每包为 `(K,N)`。暂时没有输出时返回空列表；空输入包经类型校验后为无操作。同一流的采样率、中心频率、域、dtype 和 metadata 必须一致；非空包的时间戳须连续，采用绝对容差 `max(1 ns, 2*时间戳浮点间距)`，新流应先显式重置。

`flush()` 输出剩余完整帧，允许最后一个批次不足 batch_size；不补齐不足一帧的尾部。重叠分析会保留用于下一帧的重叠样点。`reset_buffer()` 明确结束当前流并返回释放的缓存样点数，其中可能包含已参与前一帧处理的重叠样点；不重置累计测量。

`configure(**parameters)` 验证后切换配置；有待处理/重叠缓存时拒绝修改，调用方可先 flush 再决定是否 reset_buffer。修改 fft_size 时默认恢复无重叠；单独修改 overlap 会重算 hop。`reset_metrics()` 只重置统计，保留流状态，以当前缓存字节数作为新统计窗口的初始峰值。配置和缓冲重置不计入处理调用次数。

## 测量口径

公共框架用单调高精度时钟 `perf_counter_ns()`，统计窗口为自初始化或最近一次 `reset_metrics()` 起到当前调用。普通字段描述最近一次 `process/push/flush` 调用；`total_*` 保存窗口内累计值。

| 指标 | 精确含义 |
| --- | --- |
| `processing_time_ns / latency_ms` | 单次公开处理调用的实际耗时，包含校验、分帧、算法和输出包构造；不含调用前等待时间和统计快照整理开销 |
| `input_samples / output_samples` | 最近调用输入/输出元素总数；二维数组按所有元素计数，重叠帧输出允许重复计入样点 |
| `input_bytes / output_bytes` | 最近调用输入/输出数组有效载荷字节数；以实际 dtype 为准，不含包元数据 |
| `throughput_items_per_s` | `output_samples / 调用耗时秒`；C08 的单位为输出 samples/s |
| `throughput_bytes_per_s` | `(input_bytes + output_bytes) / 调用耗时秒`，表示接口双向有效载荷处理速率 |
| `buffer_current_bytes` | 调用后保留的输入流缓存有效载荷字节数；process 不持有跨调用缓存 |
| `buffer_peak_bytes` | 窗口内观测到的最大单个输入缓存有效载荷，包含 push 拼接后、消费前的数组 |
| `queue_depth` | 当前同步实现为 0；预留异步排队接口 |
| `call_count / error_count` | process/push/flush 的累计尝试调用数/异常调用数；无输出的成功 push 也计数 |
| `last_call_succeeded` | 最近调用是否成功；失败时输出量和吞吐率为 0，仍记录尝试输入量和耗时 |
| `total_input_* / total_output_* / total_frames` | 仅成功调用的累计输入、输出和变换帧数 |
| `total_processing_time_ns` | 所有成功及失败调用的累计耗时 |
| `total_success_time_ns` | 仅成功调用的累计耗时，包括成功但尚未产生输出的 push |
| `average_latency_ms` | `total_processing_time_ns / call_count / 1e6`；无调用时为 0 |

`component_specific_metrics` 包含 `frame_count`（本次成功变换帧数）、`frame_attempt_count`（本次尝试帧数）、`frame_processing_times_ns`（各帧算法路径耗时列表）、`frames_per_s`（成功帧数/公开调用耗时秒）、`batch_sizes`（本次各输出包实际批量），以及当前 FFT 配置。逐帧时间包含窗与 shift，但不包含整个公开调用的校验和组包。算法异常时仍保留已尝试帧的计时，失败调用的成功帧数为 0。

flush 没有新输入，其输入量为 0，但可产生前面已缓存的数据；无输出的 push 的 samples/s 为 0，字节吞吐仍可反映接收数据。上述吞吐是调用执行期间的处理速率，不是包括等批、采集与空闲时间的持续流速率；调用时延也不包含批次等待。计算整个窗口的处理速率应使用累计成功输出量和 `total_success_time_ns`，不能对各次吞吐率直接取平均。

输入缓存指标不包含同时存活的旧数组、输出包、工作副本、NumPy 内部临时区及 Python 开销，不能当作进程 RSS 或完整内存峰值；字节吞吐也不等于网卡实测带宽。单次 NumPy CPU 测量不能证明 FPGA/DSP 性能或硬实时保证。当前组件同步且不支持同实例并发处理；异常时保留原流缓存以便重试。

## 运行示例

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m examples.fft_runtime_demo
python -m unittest discover -s tests -v
```

示例从 C08 静态定义创建实例，将合成的复数单频信号分块送入 FFT，检查峰值和 IFFT 恢复误差，并打印实测运行指标。只做完整帧处理时可写：

```python
import numpy as np
from task_repr.runtime import DataPacket
from task_repr.component_models.common.fft_component import FFTComponent

fft = FFTComponent(fft_size=1024, window="hann", shift=True)
iq = np.exp(2j * np.pi * 128 * np.arange(1024) / 1024)
result = fft.process(DataPacket(iq, 1_024_000, 100e6, 0.0, "time"))
print(result.domain, result.samples.shape)
print(fft.metrics.latency_ms, fft.metrics.throughput_items_per_s)
```

后续组件复用 DataPacket、RuntimePort 和 RuntimeComponent；算法写在 `_execute()`，以 `_run_frame()` 暴露算法测量点，对外接口通过 `_measure_call()` 统一统计。若持有跨调用输入缓存，重写 `_buffer_nbytes()` 并在关键变化处调用 `_observe_buffer()`；业务指标由 `_specific_metrics()` 提供。当前框架面向一维/二维数值载荷，尚未定义多端口汇聚、容器通信或自动调度。
