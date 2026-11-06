# 射频前端软件仿真范围

## 调研结论

软件可以在**已采样复数 IQ** 上模拟前端对信号造成的可观测影响：固定增益、数字 AGC、复数频移、有限带宽、加性噪声和幅度限幅。GNU Radio 的 [复数 AGC 核心](https://github.com/gnuradio/gnuradio/blob/main/gr-analog/include/gnuradio/analog/agc.h)采用目标幅度与当前输出幅度的误差更新增益；[频移 FIR 滤波器](https://github.com/gnuradio/gnuradio/blob/main/gr-filter/lib/freq_xlating_fir_filter_impl.h)组合数字旋转器和 FIR。它们证明这些**样本域**操作可软件实现。

UHD 的 [`multi_usrp` 接口](https://uhd.readthedocs.io/en/latest/classuhd_1_1usrp_1_1multi__usrp.html)区分收/发频率、增益与设备流控制。软件计算出的 IQ 不能代替天线、模拟混频器/本振、LNA、PA、ADC、DAC 或真实射频发射。当前组件也不模拟校准、频率相关噪声系数、阻抗、相位噪声、镜像抑制、互调、邻道泄漏和法规功率限制；没有实测硬件标定。

## 组件边界

| CID | 范围 | 输入/输出 |
| --- | --- | --- |
| C04 | 数字 AGC：目标幅度、攻击/释放更新速率和增益上下限 | 复数 IQ → 复数 IQ |
| C33 | 接收链 IQ 效应：固定增益、可选复高斯噪声、限幅、频移、低通 FIR | 已采样 IQ → 接收参考 IQ |
| C34 | 发射链 IQ 效应：低通 FIR、频移、固定增益和限幅 | 基带 IQ → 发射参考 IQ |

C03 仍是数字 DDC/DUC，C05 仍是数字滤波器。C33/C34 是较粗的**前端整体行为模型**，不是 C03/C05 的硬件版本。C33 的输入需要由外部仿真器或文件提供；C34 的输出仅是样本数组，不开设备、不调用 socket、不向空中发射。能力映射只登记 `rf_iq_receive_simulation` / `rf_iq_transmit_simulation`，不会自动把任务中泛指真实硬件的 `rf_receive` / `rf_transmit` 绑定到它们。

## 数值约定

- 所有频率参数单位为 Hz，`lo_offset_hz` 必须严格小于输入采样率的一半。接收模型乘 `exp(-j2πfn/fs)`，发射模型乘 `exp(+j2πfn/fs)`；相位在连续 `push()` 调用间保持。`DataPacket.center_frequency` 按既有 C03 元数据约定随频移调整，不是硬件调谐命令。
- 固定增益为 `10^(gain_db/20)`，作用于复数电压幅度。`clip_amplitude` 是接收链滤波前或发射链最终输出的 IQ 幅度上限，不代表真实 PA 的 AM/AM、AM/PM 曲线。接收噪声的 `noise_std` 是复数 IQ 的总 RMS 标准差；`seed` 固定后结果可复现。
- `bandwidth_hz` 是以 IQ 零频为中心的双边通带宽度，需小于采样率。当前 FIR 使用 Hamming 窗和奇数 `filter_taps`；存在 `(filter_taps-1)/2` 个样本的未补偿群时延，记录在输出元数据的 `filter_group_delay_samples` 中，输出时间戳不作补偿。关闭滤波时将 `bandwidth_hz` 设为 `null`，群时延元数据为 0。
- C04 按每个复数样本更新 `gain = clip(gain + rate × (target_amplitude - abs(output)), min_gain, max_gain)`；输出过大时用 `attack_rate`，否则用 `release_rate`。这是简化 AGC 行为模型，不是特定硬件控制环路。

运行示例：

```python
import numpy as np
from components.implementations.runtime_registry import create_runtime_component
from components.runtime import DataPacket

samples = np.exp(2j * np.pi * 100_000 * np.arange(1024) / 1_000_000)
packet = DataPacket(samples, 1_000_000, 100_000_000, 0, "time")
rx = create_runtime_component("C33", lo_offset_hz=100_000,
                              bandwidth_hz=200_000, gain_db=6)
baseband = rx.process(packet)
controlled = create_runtime_component("C04").process(baseband)
```

运行组件延续 `process()` 独立块与 `push()` 连续流接口。输出元数据和 `metrics.component_specific_metrics` 提供当前增益、频移、噪声和限幅样本数。目录中的资源等级仅为相对仿真标记，不能当作 CPU/FPGA 实测占用。
