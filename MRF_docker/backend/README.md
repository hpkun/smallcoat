# 后端接口

`api/` 提供 HTTP 接口，`services/` 组织任务与资源用例，`repositories/` 读取 JSON 文件。任务表征、组件匹配、资源规划和调度分别由对应模块实现。当前调度与资源容量属于进程内仿真，不部署 Docker 容器。

在项目根目录使用同一个 Python 环境安装依赖并启动服务：

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.api.app:app --reload
```

默认地址为 `http://127.0.0.1:8000`，接口文档位于 `/docs`。`MRF_TASK_DIR` 和 `MRF_RESOURCE_DIR` 可指定其他 JSON 目录；文件名去掉 `.json` 后作为查询名称。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/tasks` | 列出任务文件 |
| GET | `/api/v1/tasks/{name}` | 读取任务 JSON |
| POST | `/api/v1/tasks/validate` | 校验细化任务 |
| POST | `/api/v1/task-representations/external` | 解析单条或批量外部射频请求，返回抽象表征及公共功能模板；语音发送也使用此接口 |
| POST | `/api/v1/tasks/{name}/match` | 匹配已保存的细化任务 |
| POST | `/api/v1/matches` | 匹配提交的细化任务 |
| POST | `/api/v1/plans` | 预览归一化资源规划 |
| POST | `/api/v1/simulations/schedule` | 仿真容量预留与实例激活 |
| POST | `/api/v1/simulations/tasks/{task_id}/complete` | 将实例转为 Warm 状态 |
| GET | `/api/v1/simulations/state` | 查看任务及实例状态 |
| GET | `/api/v1/resources` | 列出资源文件 |
| GET | `/api/v1/resources/{name}` | 读取资源 JSON |

抽象任务表征不能直接提交给匹配或规划接口。`TaskService.execute(document, sources)` 在进程内执行已校验的 DAG；`execute_with_schedule` 还会预留仿真容量，并在成功时转为 Warm、失败时释放。HTTP 执行接口尚未定义数据包上传与结果格式。资源 JSON 当前作为独立清单读取，不会自动转换为仿真调度容量。
