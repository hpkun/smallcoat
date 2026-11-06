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

已有明确细化方案的业务请求可直接交给 `matching.match_task(raw)`，也可先 `parse_external_task(raw)` 后交给同一入口。细化规则保存在 `parsing/data/business_function_rules.json` 的 `refinement` 中：按任务类型和方向选择方案，将业务角色展开为零个、一个或多个规格节点，参数只从 `S.business_parameters` 的指定路径读取，端口格式必须存在于 OPS。零节点角色表示外部输入或输出边界；复合角色的内部边和业务角色之间的边都会保留。随后现有匹配器依据能力映射、OPS/OCS、参数及连接条件筛选 CID。

当前已提供 `spectrum_analysis/rx` 方案，样例为 `examples/BusinessRequests/External/07_spectrum_analysis.json`。请求需明确选择 `processing.profile=ddc_fir_psd`。该方案将 `downconvert_iq` 展开为 DDC，将 `analyze_spectrum` 展开为 FIR 滤波和频谱感知。结果的 `refined_graph` 包含可追溯的规格节点、连接和外部边界；`MATCHED` 加 `match_scope=component_dag` 仅表示组件子图静态契约通过，不表示外部 IQ 流已绑定、端到端质量达标或资源已规划。原 `S/Q/R/X` 在细化阶段保留，未知资源仍为 `null`，不会凭空产生估计值。

其他业务类型或缺少明确的输入格式、算法参数时返回 `NEEDS_REFINEMENT` 和 `unresolved`，不会给出 CID 绑定。要扩展业务类型，需补齐该任务角色的规格展开、参数来源、OPS 格式与可运行组件能力；语音发送目前缺少完整编解码和空口实现，因此仍待细化。

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
