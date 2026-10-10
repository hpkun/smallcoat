# 固定 trace 的生成和回放

## Seed 42：500 episodes V2 违约原因拆分

使用已保存的 V2 检查点和原来的 2000-task trace，只回放 V2。脚本会核对原比较报告中的
检查点、trace、逐任务输入哈希和九项指标；匹配后才保存分类结果。

```powershell
Set-Location F:\small_artical\recur\D3QN\drl_ra_demo1_work
$runDir = "outputs/seed42_v2_stopmask_500ep_20261008"
python -u audit_violation_causes.py --trace outputs/traces/tasks2000_seed42.json --checkpoint "$runDir/drl-ra-learned-replica_seed42/model.pt" --comparison "$runDir/replay/metrics_comparison.json" --output-dir "$runDir/violation_breakdown" --device cpu --torch-threads 1
if ($LASTEXITCODE -ne 0) { throw "违约原因拆分失败" }
Get-Content "$runDir/violation_breakdown/violation_breakdown.md" -Encoding UTF8
```

分类使用任务执行前、自回归选节点时的同一候选快照。当前独立副本可靠性模型中，
全局可靠性最优组合为可行节点中可靠性最高的三个；固定 primary 的最优组合为
primary 加上可靠性最高的两个合法 backup。因此无需运行 reward oracle。

三类依次为：未到三副本且没有合法 backup 的 `no_backup_infeasible`；到三副本且
全局最多三节点的最优组合仍未达标的 `max3_intrinsic_infeasible`；到三副本且存在
含当前 primary 的达标组合、实际选择却未达标的 `max3_placement_failure`。
若全局有解而固定 primary 无解，严格说不属于上述三类，单独记录为
`max3_primary_bottleneck`，并标记三类是否覆盖全部违约。

输出 `violation_breakdown.md` 给出任务数、占全部任务比例及占违约任务比例；
`violation_breakdown.json` 额外保存违约任务 ID、候选可靠性、实际集合与最优组合，
便于逐任务核查。已有输出会拒绝覆盖；重复运行时请更换输出目录。

## Seed 42：100 episodes 精简比较

以下命令从头训练三组，每组 100 episodes、每 episode 1000 个任务。V2 使用
`gamma_intra=1.0`、`gamma_inter=0.99`。训练和回放的 `--metrics-only` 均跳过
可行性审计及 reward oracle，只保留正常执行与统计；不修改环境、reward 或策略。

```powershell
Set-Location F:\small_artical\recur\D3QN\drl_ra_demo1_work
$runDir = "outputs/seed42_100ep_fixed2000_20261007"
if (Test-Path $runDir) { throw "输出目录已存在，请更换 runDir 后再运行。" }

python -u train.py --config configs/paper.yaml --method d3qn --seed 42 --device cpu --torch-threads 1 --metrics-only --set training.episodes=100 --output-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "D3QN 训练失败" }
python -u train.py --config configs/paper.yaml --method drl-ra --seed 42 --device cpu --torch-threads 1 --metrics-only --set training.episodes=100 --output-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "DRL-RA 训练失败" }
python -u train.py --config configs/paper.yaml --method drl-ra-learned-replica --seed 42 --device cpu --torch-threads 1 --metrics-only --set training.episodes=100 --output-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "V2 训练失败" }

# 复用已保存的同一份 2000-task trace。
python -u replay_trace.py --metrics-only --torch-threads 1 --trace outputs/traces/tasks2000_seed42.json --d3qn-checkpoint "$runDir/d3qn_seed42/model.pt" --drl-ra-checkpoint "$runDir/drl-ra_seed42/model.pt" --learned-replica-checkpoint "$runDir/drl-ra-learned-replica_seed42/model.pt" --output-dir "$runDir/replay"
if ($LASTEXITCODE -ne 0) { throw "固定 trace 回放失败" }
Get-Content "$runDir/replay/metrics_comparison.md" -Encoding UTF8
```

结果仅比较 TCR、CVR、MeanReplicas、Energy（mJ/task）、Latency（ms）、
ReliabilityShortfall（平均 `max(0, required - achieved)`），以及 1/2/3 副本比例。
`metrics_comparison.json` 保存 trace、检查点哈希及逐任务输入验证结果；
`shared_inputs_verified=true` 表示三个算法逐任务读取了相同输入。
精简 JSON 也保存 Learned Replica 的 `stop_reasons` 与 `infeasible_task_ids`，
不运行额外审计。`infeasible` 表示当前副本集合未达可靠性要求且没有可行 backup；
达到 3 副本仍记录为 `max_replica_stop`。

如果尚无 trace，先生成一次：

```powershell
python generate_trace.py --config configs/paper.yaml --seed 42 --steps 2000 --output outputs/traces/tasks2000_seed42.json
```

## 完整 Step 2–5 审计

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
- `behavior_diagnostics.json`：同一次 Learned Replica 回放的诊断，停止原因分为主动 STOP、达到上限、达标但无候选、未达标且无候选（`infeasible`）四类。

Step 2 的 `max_reliability_all` 与内在不可行性沿用现有审计定义：在全局可行候选中最多选择 3 个副本。
`min_replicas_to_requirement=-1` 表示在该预算内不可达。
CVR 分解为互斥的内在不可行、Primary 瓶颈、副本策略失败三项。

Step 4 的 reward oracle 沿用现有快照定义：枚举全局可行集合，时延取所选候选的最小时延，不对成功抽样求期望。
`snapshot_reward_regret` 将所选集合与相同快照下的全局最优 reward 比较。

Step 5 的四类停止原因仅适用于 Learned Replica；其余算法在汇总中标为不适用。
容量、可见性、电池、覆盖阻塞是候选级计数，可以重叠，不是互斥的任务停止原因。

验证：

```powershell
python -m unittest discover -s tests -v
```

新增测试覆盖不同副本数下输入一致、重复观测/跳过中继时信道一致、节点成功抽样与选择顺序无关、trace 校验/重置/耗尽，以及三个真实格式检查点的完整回放。
