# CMADDPGPRO

面向 SAGIN（UAV / BS / LEO）任务卸载场景的 CMADDPG 复现实验项目。当前版本已经包含任务生成、通信与计算环境、KMDUC 动态聚类、混合动作空间、共享奖励、多智能体经验回放、Actor/Critic 训练、冗余卸载、基线评估和指标绘图脚本。

## 项目目标

本项目围绕论文式的多 UAV 协同任务卸载问题构建可运行仿真环境，并用 CMADDPG 学习簇头 CH 的卸载决策。每个 CH agent 根据局部观测，为一个或多个任务槽位选择计算目标，并输出任务优先级和冗余卸载强度。

核心问题包括：

- UAV 动态移动和 KMDUC 聚类维护
- 地面任务到达、UAV 接入、CH 决策和多层计算节点执行
- UAV / BS / LEO 三层候选计算资源选择
- deadline、capacity、reliability 等约束下的收益最大化
- 混合动作空间中的离散卸载目标与连续控制变量联合学习
- workflow DAG 任务依赖、ready task 释放和工作流 SLA 统计
- 传输、计算、冗余副本与提前取消过程的实际能耗统计

## 系统流程

一次训练时隙按以下流程推进：

1. UAV 根据速度和航向更新位置，KMDUC 维护簇成员与簇头。
2. 独立任务按泊松过程到达；workflow 模式则按 DAG 依赖释放 ready tasks。
3. 普通成员 UAV 的任务汇聚到所属 CH，CH 或孤立 UAV 作为决策 agent。
4. Actor 为每个任务槽位输出卸载目标、计算优先级和冗余强度。
5. 环境应用候选 mask，检查唯一卸载、deadline、capacity 和 reliability 约束。
6. 任务在 UAV、BS 或 LEO 的非抢占式优先级队列中执行；冗余任务可提交主副本和备份副本。
7. 首个成功副本完成后，尚未完成的副本按物理时间取消，并结算实际发生的能耗。
8. 环境生成共享奖励、任务/工作流指标和联合经验，CMADDPG 更新 Actor 与 Critic。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行最小烟雾测试：

```bash
python tests/test_smoke.py
```

运行单元测试：

```bash
python -m pytest
```

训练默认 `training` 环境：

```bash
python train.py
```

更短的调试训练：

```bash
python train.py --episodes 5 --steps 10 --batch-size 8 --progress-interval 1
```

小规模环境训练：

```bash
python train.py --env small --episodes 20 --steps 20
```

启用 workflow DAG 任务模式：

```bash
python train.py --task-mode workflow --episodes 50 --steps 50
```

运行原基线兼容的三种任务分布：

```bash
python train.py --scenario delay-sensitive
python train.py --scenario computation-intensive
python train.py --scenario balanced
```

关闭冗余卸载作为 plain offloading 对照：

```bash
python train.py --redundancy-mode none
```

启用 Actor 自注意力编码器：

```bash
python train.py --actor-attention
```

评估启发式与随机基线：

```bash
python scripts/evaluate.py
```

训练日志默认写入：

```text
outputs/metrics/train_metrics.json
```

绘制系统能耗结果图：

```bash
python scripts/plot_energy_bars.py outputs/metrics/train_metrics.json \
  --output outputs/figures/energy_bars.png \
  --group-size 10
```

UAV battery history is logged for every simulation step and at the end of
every episode. Plot all UAVs, or select specific UAV IDs:

```bash
python scripts/plot_uav_battery.py outputs/metrics/train_metrics.json
python scripts/plot_uav_battery.py outputs/metrics/train_metrics.json \
  --level episode --uavs uav-0 uav-3 \
  --output outputs/figures/uav_battery_episode.png
```

左图分别展示全部 episode 的 Computing 与 Transmission 总能耗，右图展示按
`--group-size` 汇总后的逐组能耗趋势。两幅图中的能耗分项共用同一个 Y 轴，便于直接
比较绝对能耗；当 Transmission 远小于 Computing 时，其柱或曲线会贴近零轴，但实际
数值仍保存在日志和左图柱顶标签中。能耗单位均为焦耳（J）。

## 能耗模型

能耗计算实现在 `src/energy.py`，由环境在任务副本真正提交执行时调用。每个任务副本的
总能耗由传输能耗和计算能耗组成：

```text
E_total = E_tx + E_compute
E_tx = P_tx * T_tx
E_compute = kappa_node * f_energy,node^2 * C_task / eta_task
```

其中，`P_tx` 是链路发射功率，`T_tx` 是 backhaul 传输时延，`kappa_node` 是不同计算
节点的有效开关电容系数，`f_energy,node` 是分层等效能耗频率，`C_task` 是任务总计算周期数，
`eta_task` 是任务并行效率。节点总计算吞吐量只用于计算时延和队列服务，不再直接平方计入能耗。传播时延不计入发射能耗。

当前统计边界如下：

- UAV 本地执行没有 backhaul，因此 `transmission_energy_j` 为 0。
- 卸载到 BS 或 LEO 时，Transmission 统计 UAV 到目标节点的 backhaul 发射能耗。
- Computing 按任务实际提交到的 UAV、BS 或 LEO 节点参数计算。
- 冗余卸载统计所有已执行副本的能耗，而不是只统计最先成功的副本。
- 副本被提前取消时，仅累计取消前实际发生的传输和计算能耗，同时记录
  `cancellation_energy_saved_j`。

主要代码入口：

- `src/energy.py`：能耗公式与 `EnergyBreakdown`。
- `src/environment.py`：任务执行、冗余副本和取消场景的能耗结算。
- `src/entities.py`：执行记录中的能耗字段。
- `src/trainer.py`：step-level 与 episode-level 能耗汇总和日志写入。
- `scripts/plot_energy_bars.py`：系统总能耗柱状图与分组 episode 折线图。

## 环境预设

`train.py` 提供三个环境规模：

| 预设 | UAV | BS | LEO | 说明 |
| --- | ---: | ---: | ---: | --- |
| `small` | 2 | 2 | 1 | 快速调试和单元测试 |
| `medium` | 40 | 25 | 1 | 中等任务到达率实验 |
| `training` | 40 | 25 | 1 | 默认训练配置 |

默认 `training` 环境参数包括：

- 区域边长：5000 m
- 时隙长度：0.1 s
- UAV 数量：40
- BS 数量：25
- LEO 数量：1
- 任务到达率：25 tasks/s
- 聚类通信半径：1200 m
- 每个 CH 最多同时决策任务槽位：2

## 代码结构

```text
src/
  action_space.py        混合动作空间、动作编解码、critic 动作编码
  baselines.py           random / heuristic 基线策略
  clustering.py          KMDUC 聚类、簇维护、簇头重选
  cmaddpg.py             多智能体系统、联合 critic 输入、训练更新逻辑
  communication.py       路损、Shannon 速率、传输/传播时延
  config.py              仿真区域、移动性、聚类和链路配置
  constraints.py         deadline、binary action、capacity 等约束检查
  debug_tools.py         动作、观测与环境状态诊断工具
  energy.py              传输、计算与任务副本总能耗模型
  entities.py            UAV、BS、LEO、任务实例、执行记录
  environment.py         SAGIN 基础环境和任务执行流程
  evaluation.py          基线评估入口逻辑
  maddpg_agent.py        单个 MADDPG agent 的 Actor/Critic 封装
  metrics_logger.py      训练指标记录和 JSON 导出
  networks.py            Actor、Critic、MLP、自注意力 Actor 编码器
  objective.py           论文目标函数拆解
  observation_builder.py 局部观测构造
  plotting.py            训练汇总、分层诊断和轨迹等综合绘图
  queue_manager.py       非抢占式优先级队列
  replay_buffer.py       多智能体经验回放池
  reward.py              共享奖励函数
  rl_env.py              CMADDPG 环境封装、CH 决策上下文
  scenario_generator.py  UAV、BS、LEO 场景与网络拓扑生成
  task_generator.py      独立泊松任务生成器
  task_model.py          任务模型、收益、计算时延
  trainer.py             训练主循环和指标统计
  workflow_encoder.py    workflow DAG 特征编码
  workflow_generator.py  合成 workflow DAG 生成器
  workflow_manager.py    依赖推进、ready task 释放和 SLA 管理
  workflow_model.py      workflow、节点和依赖数据模型
```

辅助脚本：

```text
train.py                         训练入口
scripts/evaluate.py              baseline 评估
scripts/plot_energy_bars.py      总能耗柱状图与分组 episode 能耗曲线
scripts/plot_uav_battery.py      逐步/逐回合 UAV 剩余电量曲线
scripts/plot_fig4_reward.py      Fig.4 风格 reward 曲线
scripts/plot_metrics.py          指标曲线绘制
scripts/plot_redundancy_scheme_bars.py 冗余方案对比柱状图
scripts/plot_scenario_baselines.py 三种任务分布的收益与可靠按时完成率
scripts/plot_service_metrics.py  服务质量指标绘制
scripts/plot_uav_trajectories.py UAV 轨迹绘制
scripts/plot_workflow_metrics.py workflow 指标绘制
scripts/small_experiment.py      小规模训练实验入口
scripts/debug_task_lifecycle.py  任务生命周期调试
scripts/summarize_task_lifecycle.py 任务生命周期日志汇总
examples/basic_simulation.py     基础环境仿真示例
```

## RL 建模

### Agent

每个决策 agent 对应一个 CH UAV 或孤立 UAV。普通 CM UAV 的任务会汇聚到所属 CH，由 CH 进行簇级卸载决策。

### 状态

每个任务槽位的观测由三部分组成：

```text
node_load_vector: 6 维
task_vector: 6 维
link_feature_matrix: 11 个候选目标 * 3 维
```

Actor 观测中暂不使用工作流嵌入，所以每个槽位观测维度为：

```text
6 + 6 + 11 * 3 = 45
```

每个 CH 最多 2 个任务槽位，因此单 agent 状态维度为：

```text
2 * 45 = 90
```

### 动作

每个任务槽位输出：

```text
候选目标 logits + priority_eta + redundancy_eta
```

候选目标并集由所有 UAV、所有 BS 和 LEO 组成。默认 `training` 环境中：

```text
K = 40 + 25 + 1 = 66
```

所以每个槽位输出维度：

```text
66 + 2 = 68
```

每个 agent 两个槽位：

```text
2 * 68 = 136
```

离散卸载目标通过 mask 后的 argmax 解码；`priority_eta` 和 `redundancy_eta` 经过 sigmoid 映射到 `(0, 1)`。

### Critic

Actor 使用局部观测，Critic 使用联合状态和联合动作：

```text
critic_input_dim = sum(local_state_dim_i) + sum(local_action_dim_i)
```

在一次默认 `training` reset 中，当前聚类结构产生了 12 个决策 agent：

```text
joint_state_dim = 12 * 90 = 1080
joint_action_dim = 12 * 136 = 1632
critic_input_dim = 2712
```

这个数会随 CH 数量变化。如果所有 40 个 UAV 都成为决策 agent，联合 critic 输入最坏可到：

```text
40 * (90 + 136) = 9040
```

## 搜索空间与维度爆炸

这个项目的搜索空间并不小。以默认 `training` 环境为例，单个 agent 有 2 个任务槽位，每个槽位有 66 个候选目标，仅离散目标组合就是：

```text
66^2 = 4356
```

如果同时有 `M` 个决策 agent，离散联合目标组合约为：

```text
66^(2M)
```

当 `M = 12` 时：

```text
66^24 ≈ 10^43.7
```

这还没有计算 `priority_eta` 和 `redundancy_eta` 两个连续控制变量。因此如果采用枚举搜索，空间会迅速不可承受。

本项目中的维度爆炸可以用下面的公式描述：

```text
D_critic = M * [D_state_per_agent + T * (K + 2)]
```

其中：

- `M` 是当前决策 agent / CH 数量
- `T` 是每个 CH 的任务槽位数，当前默认为 2
- `K` 是候选计算目标数量，默认 training 环境为 66
- `D_state_per_agent` 当前为 90

也就是说：

```text
D_critic = M * [90 + 2 * (66 + 2)] = M * 226
```

项目通过以下方式控制搜索空间：

- KMDUC 聚类：只让 CH 或孤立 UAV 作为决策 agent
- 任务槽位上限：每个 CH 最多同时决策 2 个任务
- 固定候选目标并集和槽位级合法 mask
- 链路特征截断到 9 条候选链路
- Actor 直接输出动作向量，避免枚举所有组合
- Critic 集中式训练、Actor 分布式执行

## 论文变量说明

论文在 Section III-B 中使用：

- `Task_k = (phi, rho, delta, G)`
- `phi` 表示输入数据大小
- `rho` 表示计算负载
- `delta` 表示容忍时延
- `G = phi * rho * exp(-lambda * delta)`

但在 Section III-C 中，论文又复用了 `phi/rho` 的含义：

- `phi_k` 被用于表示总计算需求
- `rho_k` 被用于表示并行化效率系数

为了避免实现时混淆，代码内部统一采用无歧义字段名：

- `input_size_bits`
- `total_compute_cycles`
- `tolerable_latency_s`
- `parallel_efficiency`

对应关系：

- `input_size_bits` 用于传输时延计算
- `total_compute_cycles` 对应总计算需求
- `parallel_efficiency` 对应并行化效率
- `cycles_per_bit = total_compute_cycles / input_size_bits`，用于收益函数

## 当前建模假设

- 地面任务按泊松过程到达
- 每个任务先上传到最近 UAV
- LEO 作为全局协调中心，辅助聚类和簇级调度
- UAV / BS / LEO 都可作为计算节点
- 计算节点采用非抢占式优先级队列
- 通信链路使用路径损耗和 Shannon 容量公式
- 地面近距离链路默认忽略传播时延，卫星链路可计算传播时延
- UAV 根据位置、航向和速度在时隙间移动
- reward 为共享奖励，综合收益、完成、时延、超时和约束表现
- workflow 模式下，DAG 任务按依赖关系释放 ready tasks
- 执行故障和传输故障按 Poisson 暴露过程采样
- hybrid 冗余模式根据任务优先级和 Actor 输出决定是否创建备份副本
- 首个成功副本触发后续副本取消，队列资源和能耗按实际时间结算
- 系统能耗包含所有实际执行的主/备副本，传播时延本身不产生发射能耗

## 常用输出指标

训练日志中包含 step-level 和 episode-level 指标，主要包括：

- `shared_reward`
- `system_profit` / `episode_system_profit`
- `u_base` / `episode_u_base`
- `u_net` / `episode_u_net`
- `reliable_on_time_completion_rate`
- `episode_reliable_on_time_completion_rate`
- `actor_loss`
- `critic_loss`
- `system_transmission_energy_j`
- `system_computing_energy_j`
- `system_total_energy_j`
- `episode_transmission_energy_j`
- `episode_computing_energy_j`
- `episode_total_energy_j`
- `task_completion_rate`
- `task_timeout_or_drop_rate`
- `task_deadline_failure_rate`
- `task_capacity_drop_rate`
- `redundancy_rate`
- `redundancy_success_rate`
- `backup_selection_rate`
- `avg_end_to_end_reliability`
- `uav_arrival_rate` / `bs_arrival_rate` / `leo_arrival_rate`
- `uav_avg_delay_s` / `bs_avg_delay_s` / `leo_avg_delay_s`
- `completed_workflows`
- `failed_workflows`
- `workflow_sla_violation_rate`
- `avg_completed_workflow_makespan_s`

基线兼容收益指标的定义为：

```text
U_base = episode_system_profit / normalize_profit_scale
U_net = U_base - energy_penalty_weight * episode_total_energy_j / normalize_energy_j
```

`Reliable On-Time Completion Rate` 按任务数加权统计，仅将按时完成、未发生实际失效且达到期望可靠性的任务计为成功。

Tasks may execute across multiple slots. Each node admits work while its remaining
queued service plus the new task is within the layer-specific queue limit: 0.40
seconds for UAV, 0.20 seconds for BS, and 0.10 seconds for LEO. Overflow is
reported as `task_capacity_drop_rate`; admitted work that finishes after its task
limit is reported separately as `task_deadline_failure_rate`.

## 测试

测试目录覆盖以下关键行为：

- `test_smoke.py`：基础环境构建和最小运行链路
- `test_action_space.py`：混合动作编码、mask、随机动作和 critic 动作向量
- `test_attention_actor.py`：Actor 自注意力编码器
- `test_cluster_head_agents.py`：CH/孤立 UAV agent 构建和联合维度
- `test_redundant_offloading.py`：主备副本选择、可靠性和冗余执行
- `test_queue_cancellation.py`：副本取消、队列资源释放和部分能耗
- `test_energy.py`：传输、计算、总能耗公式与日志汇总
- `test_reward.py`：共享奖励各组成项

运行全部测试：

```bash
python -m pytest
```

当前 `requirements.txt` 只包含运行时依赖。若环境中未安装测试工具，需要额外执行：

```bash
pip install pytest
```

## 开发备注

- 默认训练环境较大，调参或排错时建议先用 `--env small`。
- 如果 loss 不稳定，可以先降低任务到达率、缩短候选目标集合或关闭 workflow。
- 如果训练速度慢，优先检查当前 CH 数量，因为联合 critic 输入随 CH 数线性增长。
- `--actor-attention` 会让 Actor 在每个任务槽位内对 node / task / link tokens 做自注意力编码，表达能力更强，但训练开销也更高。
