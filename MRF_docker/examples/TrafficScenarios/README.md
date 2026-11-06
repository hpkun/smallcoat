# 独立数据链业务流仿真

[`data_link_2s.mgn`](data_link_2s.mgn) 参考 [MGEN 官方 `example.mgn`](https://github.com/USNavalResearchLaboratory/mgen/blob/master/doc/example.mgn) 的事件语法构造。官方文件是流量生成器脚本示例，不是航空数据链实测数据。本场景的流量数值、端口和业务含义由本项目独立设定，未参考 `task_repr` 的模型或模板。

| 业务流 | 时间 | MGEN 事件 | 计划流量 |
| --- | --- | --- | --- |
| 1，持续数据传输 | 0–2 秒 | `ON` + `PERIODIC [50 256]` + `OFF` | 50 包/秒，每包 256 字节，共 100 包 |
| 2，短报文突发 | 0.5 秒开始 | `ON` + `PERIODIC [20 512] COUNT 8` | 20 包/秒，每包 512 字节，共 8 包 |

`LISTEN UDP 5000` 和 `IGNORE UDP 5000` 是接收端监听窗口事件。回环目的地址仅用于场景描述；仿真器不打开 socket、不发送 UDP、不模拟无线信道，也不判断丢包或到达。它为每个计划发送的包记录业务流、序号、时间戳、端口和大小。

在项目根目录运行：

```powershell
python -m examples.data_link_traffic_sim
```

完整事件及逐包发送计划写入 `outputs/data_link_2s_packets.json`。可用 `-o` 指定输出路径，也可传入同一语法子集的其他 `.mgn` 文件。脚本只支持此样例使用的 `ON/OFF/LISTEN/IGNORE`、UDP、`PERIODIC` 和可选 `COUNT`，不声称完整实现 MGEN 语法。

仿真约定：`ON` 时刻计划第一个包，其后按 `1/包率` 秒间隔生成；`OFF` 时刻不再生成包；`COUNT` 达到后停止该流。事件时间与包时间都是虚拟时间，不会等待真实的 2 秒。这个业务场景可以作为后续任务表征模块的外部输入测试材料，但本示例没有执行任何任务表征、组件匹配或调度。

## 交给任务表征模块

业务流脚本仍独立。单独运行适配示例，提取两条流的事实并调用 `task_repr.represent_task()`：
适配脚本根据外部流事件显式构造功能图；目前的任务表征模块负责校验与结构化，不能直接从原始 `.mgn` 推断 DAG。

```powershell
python -m examples.data_link_representation_demo
python -m task_repr.visualize outputs/data_link_task_representation.json -o outputs/data_link_task_representation.png
```

第一个命令生成 `outputs/data_link_task_representation.json`；第二个命令绘制两条独立功能链。它们不进行组件匹配或数据包发送。转换使用 v1 表征，因为原始场景没有提供 v2 所要求的设备执行画像和链路质量。`Q.deadline_ms=2000` 仅用作场景结束时刻，**不是 MGEN 声明的业务截止约束**；资源量、节点耗时和优先级中的 `0` 也是缺失信息的占位值，不能据此做容量或 QoS 结论。`COUNT` 和每条流的 `ON/OFF` 时间保存在节点参数中。
