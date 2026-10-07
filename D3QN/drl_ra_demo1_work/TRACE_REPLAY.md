# 固定 trace 的生成和回放

在项目目录运行：

```powershell
python generate_trace.py --checkpoint outputs/screen_seed42/drl-ra_seed42/model.pt --seed 42 --steps 2000 --output outputs/traces/tasks2000_seed42.json
python replay_trace.py --trace outputs/traces/tasks2000_seed42.json --output-dir outputs/fixed_trace_seed42
```

`generate_trace.py` 也支持 `--config configs/paper.yaml` 和 `--set KEY=VALUE`。
默认生成 2000 个任务；回放长度由保存的 trace 决定。两个脚本均拒绝覆盖已有输出，明确需要覆盖时使用 `--overwrite`。

默认回放 `outputs/screen_seed42` 下的 D3QN、DRL-RA 和 Learned Replica 完整检查点。
自定义检查点使用 `--d3qn-checkpoint`、`--drl-ra-checkpoint`、`--learned-replica-checkpoint`。
CPU 推理默认使用一个 PyTorch 线程，可用 `--torch-threads` 调整。

Trace 保存实际初始拓扑、设备类型和容量、任务属性、到达时间/间隔、逐任务/卫星/UAV 的信道随机量，以及逐任务/节点的成功随机量。生成时五个独立随机数流分别负责拓扑、任务、到达间隔、信道和成功抽样。回放不随副本数、候选是否可用或观测次数改变输入。

三个算法使用 trace 中相同的环境和奖励配置，网络结构使用各检查点的训练配置。检查点拓扑维度必须匹配；评估参数变化记录在 `evaluation_overrides`。D3QN 保留单副本行为，DRL-RA 使用原解析副本策略，Learned Replica 使用其副本网络。组合检查点加载时跳过外部 primary warm-start 文件。

各算法的队列、资源占用、电池和中继负载按实际动作独立演化。因此相同 trace 并不意味着每个算法有相同的候选可行性或 reward oracle。节点成功事件使用固定均匀随机量与该算法当前节点可靠性比较，保证共享抽样条件。

输出包括：

- `audit_d3qn.json`、`audit_drl_ra.json`、`audit_learned_replica.json`：完整逐任务 Step 2–5 字段、动作、reward、任务输入哈希、汇总和可靠性/任务类型分组。
- `comparison.json`、`comparison.md`：三算法对比、trace/检查点 SHA-256、逐任务输入一致性验证。
- `behavior_diagnostics.json`：同一次 Learned Replica 回放的诊断，停止原因分为主动 STOP、达到上限、无可行候选三类。

Step 2 的 `max_reliability_all` 与内在不可行性沿用现有审计定义：在全局可行候选中最多选择 3 个副本。
`min_replicas_to_requirement=-1` 表示在该预算内不可达。
CVR 分解为互斥的内在不可行、Primary 瓶颈、副本策略失败三项。

Step 4 的 reward oracle 沿用现有快照定义：枚举全局可行集合，时延取所选候选的最小时延，不对成功抽样求期望。
`snapshot_reward_regret` 将所选集合与相同快照下的全局最优 reward 比较。

Step 5 的三类停止原因仅适用于 Learned Replica；其余算法在汇总中标为不适用。
容量、可见性、电池、覆盖阻塞是候选级计数，可以重叠，不是互斥的任务停止原因。

验证：

```powershell
python -m unittest discover -s tests -v
```

新增测试覆盖不同副本数下输入一致、重复观测/跳过中继时信道一致、节点成功抽样与选择顺序无关、trace 校验/重置/耗尽，以及三个真实格式检查点的完整回放。
