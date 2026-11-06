# 上层业务请求样例

`External/` 保存外部射频业务请求及其合集，根目录的 `voice_transmit_request.json` 也是同一格式。批量解析后，任务表征模块按任务族提取公共抽象功能图；具体格式与组件匹配前的细化步骤见 [`task_repr/README.md`](../../task_repr/README.md)。

`voice_transmit_request.json` 模拟语音业务模块提交的发送请求。`request.audio_source.reference` 是 PCM 缓冲区或流的标识，样例不包含实际音频数据；请求中也不包含组件 ID 或功能 DAG。

在项目根目录运行：

```powershell
python -m examples.voice_transmit_representation_demo
```

也可以启动后端，再通过 HTTP 提交同一个请求：

```powershell
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

在另一个终端运行：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/task-representations/external -ContentType application/json -InFile examples/BusinessRequests/voice_transmit_request.json
```

结果包含 `task_template` 与 `task_representation`。当前抽象功能链为接收语音业务数据、准备语音载荷、经无线链路发送；编解码和调制算法须在后续功能细化时确定。本阶段不读取 PCM、不做组件匹配，也不运行发送链。

可运行 `python -m task_repr.visualize` 生成任务表征图，默认输出到 `outputs/voice_task_representation.png`；使用 `-o outputs/voice_task_representation.svg` 可生成矢量图。
