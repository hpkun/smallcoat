# 联合 D3QN 实验

`joint-d3qn` 使用一个共享编码器和四个 dueling Q head，顺序选择主节点、总副本数、附加副本节点和计算资源比例。总副本数包含主节点，最多为 `environment.max_replicas`。`drl-ra-resource` 使用同样的资源状态与资源 head，但保留原 DRL-RA 规则副本规划，作为资源感知的解耦对照。两者使用相同的资源执行与奖励，便于比较学习副本策略的影响。原 `drl-ra`、`d3qn`、`drl-ra-learned-replica` 和 `d3qn-ppo` 入口继续可用。

默认 20 个候选节点对应 185 维联合状态：4 维任务信息、每节点 7 维资源/队列/可靠性/链路特征、已选集合、主节点 one-hot 和目标副本数。主节点单独编码，避免同一副本集合中不同主节点产生状态混淆。

资源等级在 `joint.resource_levels` 配置，默认为 `[0.25, 0.5, 0.75, 1.0]`。所有副本共享本次选择的比例，各节点实际分配的算力为该比例乘以节点当前可用算力。本地节点使用该设备的本地算力；本地队列继续独立计算。计算时间采用 `cycles / allocated_capacity`。远端沿用原计算功耗模型，本地沿用随算力平方变化的能耗模型，因此提高资源比例对两类节点的能耗影响不同。

`step_joint()` 验证完整动作并重新计算候选节点的时延、能耗和可靠性，再复用 `_execute_replica_set()` 的预留、赢家分配和副本取消流程。远端预留量为实际分配算力。节点与资源掩码排除容量不足或不能在卫星可见窗口内完成的动作。本地仍允许提交截止时间不可满足的任务，以保留兜底执行路径；延迟和可靠性惩罚反映失败风险。

环境奖励为：

```text
reward.reliability * (R_joint - R_primary)
- reward.latency * min(latency / deadline, 3)
- reward.energy * min(total_energy_mj / 2000, 3)
- reward.resource * (resource_ratio * replica_count / max_replicas)
- reward.violation * max(0, R_required - R_joint)
```

`R_primary` 是同一资源等级下仅主节点的可靠性。默认 `reward.resource=0.1`。训练另复用既有滑动窗口 Lagrange 约束：从任务奖励减去自适应乘子乘以现有平滑可靠性缺口成本。只在最后的资源决策记录任务奖励；任务内阶段使用折扣 1，跨任务使用 `training.gamma`。回放按当前 head 和下一 head 配对，在线网络在下一 head 的有效掩码内选动作，目标网络评估该动作，实现 Double Q。

训练与评估示例（在项目目录运行）：

```powershell
python train.py --method joint-d3qn --seed 42 --torch-threads 1
python evaluate.py --method checkpoint --checkpoint outputs/joint-d3qn_seed42/model.pt --seeds 0 1 2 --output outputs/joint_evaluation.json

python train.py --method drl-ra-resource --seed 42 --torch-threads 1
python evaluate.py --method checkpoint --checkpoint outputs/drl-ra-resource_seed42/model.pt --seeds 0 1 2 --output outputs/resource_evaluation.json
```

运行小规模功能验证时，可给训练添加 `--set training.episodes=2 --set environment.episode_steps=16 --set training.batch_size=4`，给评估添加 `--set environment.episode_steps=32`。`--metrics-only` 跳过训练中的可选副本 oracle 审计。

输出包含原 TCR、CVR、能耗、时延、可靠性缺口和平均副本数，以及 `mean_reliability_gain`、`mean_resource_ratio`、`mean_resource_cost`、`compute_utilization_pct`。原 `resource_utilization_pct` 仍表示边缘队列忙碌比例；新增计算利用率是每次任务预留后远端已用与预留算力占远端总算力的比例，不是按时间积分的利用率，也不包含本地设备。

可用 `--set environment.reliability_requirement_override=0.99` 或 `--set environment.arrival_rate=25` 控制可靠性要求和负载。正式对比应统一配置、训练预算和评估 seeds，使用多个训练 seed；短运行只验证功能，不代表收敛或性能优势。

验证命令：`python -m pytest tests -q`。指定正式测试目录，以免收集 `tmp_smoke` 中历史恢复副本的同名测试。
