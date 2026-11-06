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

先针对具体任务把抽象功能细化为 `schema_version=2` 的任务图：为每个功能节点确定组件能力 ID、输入输出端口及数据格式、必要参数和可验证的时延与资源估计；同时保留原任务的优先级、截止时间和质量约束。`represent_task(v2_document)` 校验图无环、端口和字段契约，然后 `matching.match_task(task)` 依据能力映射、OPS/OCS、参数及连接条件筛选组件。结果包含绑定、已验证的边和 `unresolved` 问题；只有 `MATCHED` 才能继续资源规划。当前没有把抽象请求自动细化为 v2 的通用适配器，细化过程需要明确波形、算法和信号接口，不能凭抽象角色推断组件 CID。

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
