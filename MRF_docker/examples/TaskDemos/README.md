# 任务实例目录

当前目录包含 9 个统一模板 v2 实例和 1 个 v1 兼容性示例。v2 实例按 CNI、Radar、EW、ISAC 分类；参数均为用于验证模板和调度流程的演示值，不能直接当作设备实测能力或验收指标。

另有 [`fft_runtime_demo.py`](fft_runtime_demo.py) 演示真实复数信号的 C08 FFT/IFFT 运行与测量，不属于任务 JSON 实例。在项目根目录执行 `python -m examples.fft_runtime_demo`，数据契约和指标分母见[运行接口说明](../task_repr/runtime/README.md)。

[`fft_instance01.py`](fft_instance01.py) 提供命名为 `fft_instance01` 的 C08 正向 FFT 组件实例。在项目根目录运行 `python -m examples.fft_instance01`，可查看合成 IQ 信号的峰值频率、资源估计和运行时延。

| 分类 | 任务实例 | 处理链 | 资源估计（CPU / 内存 / DSP / FPGA / 带宽） | 优先级 / 截止时间 |
| --- | --- | --- | --- | --- |
| CNI | [战术话音接收](cni_tactical_voice_task.json) | 接收 → 同步 → 解调 → 信道译码 → 语音解码 | 300m / 128 MiB / 1 / 1 / 0.2 Mbps | 7 / 20 ms |
| CNI | [视频回传](cni_video_backhaul_task.json) | OFDM 同步 → FFT → 均衡 → QAM 解调 → LDPC 译码 | 800m / 2048 MiB / 2 / 1 / 50 Mbps | 4 / 100 ms |
| CNI | [旧版语音发送](voice_tx_task.json) | v1 格式兼容性样例 | v1 未包含完整异构资源画像 | 仅兼容性测试 |
| Radar | [目标搜索](radar_search_task.json) | DDC → 脉冲压缩 → FFT/MTD → CFAR | 500m / 512 MiB / 2 / 1 / 100 Mbps | 8 / 1 ms |
| Radar | [IFF 身份判决](radar_iff_task.json) | 脉冲检测 → 回复解码 → 身份判决 | 300m / 256 MiB / 1 / 1 / 20 Mbps | 9 / 1 ms |
| Radar | [SAR 成像](radar_sar_task.json) | DDC → 脉冲压缩 → 运动补偿 → 方位压缩 → 成像 | 2000m / 4096 MiB / 2 / 2 / 500 Mbps | 3 / 1000 ms |
| EW | [ESM 侦察](ew_esm_task.json) | 信道化 → 频谱感知 → 参数估计与 DOA → 辐射源识别 | 500m / 1024 MiB / 2 / 2 / 500 Mbps | 9 / 1 ms |
| EW | [ECM/DRFM 干扰](ew_ecm_task.json) | DRFM 采集 → 参数调制 → 波形生成 → 发射 | 500m / 1024 MiB / 2 / 2 / 500 Mbps | 10 / 0.5 ms |
| EW | [威胁响应综合链](ew_threat_response_task.json) | 截获/识别 → 威胁融合 → 干扰激活 → DRFM → 发射 | 1000m / 1024 MiB / 2 / 2 / 500 Mbps | 10 / 0.5 ms |
| ISAC | [通感一体化接收](isac_joint_task.json) | RF 接收 → DDC → 感知与通信双分支 | 1000m / 256 MiB / 1 / 1 / 20 Mbps | 5 / 100 ms |

## 共享组件候选

实例中的 `graph.nodes[].shareable` 表示可以进入共享候选集，不表示不同任务必然能共用同一运行实例。调度器仍需检查射频中心频率、采样率、参数配置、数据来源、时间重叠、容量和节点位置。

| 实例组合 | 共享候选 | 共享前提 |
| --- | --- | --- |
| 战术话音 / 视频回传 | `rf_receive` | 两个接收流来自兼容的宽带采样与射频前端；示例采样率不同，当前不能直接认定可合并 |
| Radar Search / SAR | `rf_receive`、`ddc`、`pulse_compression` | 雷达工作频段、采样、LFM 参数及处理窗口兼容；Search 与 SAR 的任务时限和占用仍分别调度 |
| ESM / 威胁响应综合链 | `digital_channelization` | 输入宽带数据和信道化配置一致，且分发带宽、缓冲容量允许扇出 |
| ISAC 内部双分支 | `rf_receive`、`ddc` | 一个接收/下变频结果供感知和通信分支共同消费；这是任务 DAG 内共享，不等于跨任务可共享 |

SAR 示例使用 CPU、DSP、FPGA 表示处理需求。当前模板尚未定义 GPU 资源字段或 GPU 执行配置；若实验需要 GPU 映射，应先扩展资源模型和校验器，不能把 CPU 数值当作 GPU 需求。

## 抢占关系（示例调度策略）

任务级 `preemptible` 表示任务允许在安全边界暂停，`can_preempt` 表示任务有权请求抢占低优先级工作。当前数据模型没有编码逐任务的抢占目标列表，下面的关系是目录级仿真策略，不是 JSON 中已实现的显式关系：

| 抢占者 | 可请求抢占的低优先级任务 | 不应直接抢占的任务 |
| --- | --- | --- |
| ECM/DRFM（优先级 10） | 战术话音（帧边界）、视频回传（帧/GOP 边界）、SAR（处理块边界）、ISAC（允许降级的分支） | IFF 短时解码、Radar Search 的 PRI/CPI 实时流水、持续 ESM 监听 |
| 威胁响应综合链（优先级 10） | 同上；还需预留威胁处理和射频资源 | 同上 |

抢占是否成功还取决于资源所有权、可中断边界、射频链路授权与重构时间。特别是 `can_preempt=true` 只表达调度请求权限，不表示可无条件夺取正在工作的 RF 通道。示例中 ECM 的 0.5 ms 响应预算依赖热备容器、预加载波形和资源预留，不是冷启动保证。

IFF 在当前四类 `family` 枚举中归入 Radar，并以 `task_type=iff_identification` 保留业务差异。这是本项目模板的组织选择，不代表所有系统都把 IFF 归为雷达处理链。
