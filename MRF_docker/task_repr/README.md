# 任务表征与功能模板

`task_repr` 将任务组织为 `T_rep = {S, Q, R, G, X}`，分别表示语义、质量约束、资源需求、功能图和动态状态。`parsing/` 负责输入解析与校验，`template/` 负责提取公共功能图并保存任务差异，`models/` 定义细化后任务的数据结构。

## 外部任务与公共图

外部请求样例放在 [`examples/BusinessRequests/External/`](../examples/BusinessRequests/External/) 和 [`voice_transmit_request.json`](../examples/BusinessRequests/voice_transmit_request.json)；解析规则和输出结构放在 `parsing/data/`。语音发送与其他外部射频任务统一使用 `external_rf_task.v1`：`represent_external_task(raw)` 处理单条请求，`represent_external_tasks(raw)` 处理数组或 `{ "tasks": [...] }`，也可以调用 `POST /api/v1/task-representations/external`。

批量请求按任务族 `CNI`、`Radar`、`EW` 提取模板；`template_id` 随模板输入内容稳定变化，避免不同任务集合共用同一标识。每个功能节点保留原始 `role`，并通过规则中的 `shared_roles` 得到 `canonical_role`。公共节点是参与比较的每条任务都具有的规范化角色；公共边是这些角色之间在每条任务中都存在的依赖。只对明确标记为同一业务功能的角色进行合并，不根据名称相似度猜测。

当前语音发送与数据链发送共享业务层的“接收待发送数据 → 准备载荷 → 经无线链路发送”三节点链。两种业务各自的具体功能名称、节点 ID、质量约束和其他分支保存在 `variants` 中；`restore_task(template, task_id)` 可逐条恢复完整的原始抽象表征。若同一任务族内没有公共节点，模板的公共图为空，任务图仍完整保存在变体中。这里的公共功能需求不等于组件实例可复用。

```python
import json
from pathlib import Path

from task_repr import build_template, parse_external_task, restore_task

base = Path("examples/BusinessRequests/External")
files = ("01_voice_communication.json", "02_data_link_transfer.json")
tasks = [parse_external_task(json.loads((base / name).read_text(encoding="utf-8")))
         for name in files]
template = build_template(tasks)
print(template.common_nodes, template.common_edges)
assert restore_task(template, tasks[0]["task_id"]) == tasks[0]
```

外部任务的 `G` 仍是业务级抽象 DAG，`generation_status` 为 `abstract_only_requires_refinement`。未给出的截止时间和未估算的资源保持 `null`，信号输入也可能尚未绑定。它不能直接传给 `match_task()`。

## 进入组件匹配

已有明确细化方案的业务请求可直接交给 `matching.match_task(raw)`，也可先 `parse_external_task(raw)` 后交给同一入口。细化规则保存在 `parsing/data/business_function_rules.json` 的 `refinement` 或 `refinements` 中：按任务类型、方向和 `request.processing.profile` 选择方案，将业务角色展开为零个、一个或多个规格节点。参数只从 `S.business_parameters` 的指定路径读取；端口格式要么存在于 OPS，要么明确列入 profile 的 `proposed_contracts`。后者只是待实现接口，不会被当成已有组件接口。零节点角色表示外部输入或输出边界；复合角色的内部边和业务角色之间的边都会保留。随后现有匹配器依据能力映射、OPS/OCS、参数及连接条件筛选 CID。

当前已提供 `spectrum_analysis/rx` 方案，样例为 `examples/BusinessRequests/External/07_spectrum_analysis.json`。请求需明确选择 `processing.profile=ddc_fir_psd`。该方案将 `downconvert_iq` 展开为 DDC，将 `analyze_spectrum` 展开为 FIR 滤波和频谱感知。结果的 `refined_graph` 包含可追溯的规格节点、连接和外部边界；`MATCHED` 加 `match_scope=component_dag` 仅表示组件子图静态契约通过，不表示外部 IQ 流已绑定、端到端质量达标或资源已规划。原 `S/Q/R/X` 在细化阶段保留，未知资源仍为 `null`，不会凭空产生估计值。

另有七个显式处理配置：语音 `codec2_ofdm_qpsk_conv`（发送、接收或双向），数据链 `ofdm_qpsk_conv_tx` 和 `ofdm_qpsk_conv_rx`，目标搜索 `lfm_pulse_doppler_ca_cfar`，目标跟踪 `radar_echo_gnn_ekf`，信号侦收 `wideband_channelized_psd_intercept`，信号识别 `iq_features_classifier`。这些配置给出能力节点、参数路径、接口和待实现环节；未提供配置的原始请求仍返回 `NEEDS_REFINEMENT`。

| 业务配置 | 配置中的主要参数 | 当前静态匹配情况 |
| --- | --- | --- |
| 语音 Codec 2 / OFDM | Codec 2 模式、帧长、帧格式、CRC、卷积码、QPSK 和 OFDM 参数 | 接收解调/译码及 OFDM 子链有候选；编解码、帧处理和射频发射缺组件 |
| 数据链发送 / 接收 | 帧格式、CRC、分片长度、卷积码和 OFDM 参数；接收还需已同步、已均衡的输入 | 发送可匹配 OFDM 节点，接收可匹配 C15→C14→C17；分片/重组及完整交付缺组件 |
| 目标搜索 | LFM 参考采样、带宽、脉宽、载频、PRI、CPI、Doppler FFT、CFAR 参数 | DDC、脉压和 CFAR 有候选；跨脉冲组织、功率图及报告缺组件 |
| 目标跟踪 | 测量时间/协方差、坐标系、GNN 和 EKF 配置 | 原始 IQ 到目标测量以及关联、状态更新和报告均缺组件 |
| 信号侦收 | 宽带采样率、信道数量、选定信道、PSD 和测量方法 | 信道化与 PSD 有候选；二维信道数据到一维 IQ 的选取缺组件 |
| 信号识别 | IQ 采样率、特征集、分类器及模型版本、未知类策略 | FIR 预处理有候选；特征提取、分类和报告缺组件 |

细化结果的 `refined_graph` 记录 profile 版本、`prerequisites`、`proposed_contracts`、`covered_outputs` 和 `uncovered_outputs`。`covered_outputs` 表示方案设计了相应输出，仍须由所有节点与边通过匹配才能交付；未覆盖的输出会产生 `OUTPUT_NOT_COVERED`。语音与数据链 profile 校验每符号的有效载荷、帧头、CRC、卷积码和 QPSK 子载波容量。原语音请求的每帧 60 字节与 Codec 2 3200、20 ms 方案的 8 字节不一致，演示输入显式改为 8 字节；直接给原请求添加该 profile 会返回冲突诊断。数据链 DAG 把 500 字节报文表示为多个 16 字节分片，但分片/重组算法尚无运行组件，也没有验证 50 ms 吞吐与交付质量。目标搜索检查业务要求的距离分辨率、最大距离和无模糊速度是否符合所选 LFM 带宽、采样窗口、载频与 PRI；`tx_rx` 配置只细化回波接收检测，未提供发射波形调度。跟踪配置中的原始 IQ 到测量记录仍是待实现复合能力。当前 `PARTIAL` 和 `UNSUPPORTED` 都不表示任务完成。

可运行的配置样例在 [`examples/business_refinement_demo.py`](../examples/business_refinement_demo.py)。运行 `python -m examples.business_refinement_demo` 汇总六类任务及数据链接收方案；加 `--detail` 查看所有规格节点、端口、候选组件和诊断；使用数字 `1` 到 `6` 或 `data-rx` 仅运行一个样例。样例配置只用于验证细化和匹配边界，不作为实际空口或业务帧标准。

独立的已细化 `schema_version=2` 任务图继续由 `represent_task(v2_document)` 校验，再交给 `match_task()`；该路径供已有完整执行与资源估计的任务使用。

已有细化任务可直接验证匹配流程：

```powershell
python -m examples.static_match_demo examples/TaskDemos/signal_analysis_task.json
```

语音发送输入见 `examples/BusinessRequests/voice_transmit_request.json`，可运行 `python -m examples.voice_transmit_representation_demo`。它经过同一外部任务解析器，生成待细化的业务功能图，不预设语音编解码、调制组件或资源数值。

## 可视化与测试

```powershell
python -m task_repr.visualize examples/BusinessRequests/External/all_tasks.json -o outputs/external_task.png
python -m unittest tests.test_external_task tests.test_task_template tests.test_task_repr -v
```

批量可视化会为每条任务生成一张图，展示抽象 DAG 及 `S/Q/R/X` 摘要，不执行组件匹配。历史设计说明见 [`docs/prototype_notes.md`](../docs/prototype_notes.md)。
