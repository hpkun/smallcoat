# DRL-RA 论文复现

本仓库复现论文 *Reliable Low-Latency Task Offloading and Resource Allocation Method for Space-Air-Ground Integrated Networks*（Bu 等，2026）的核心方法。实现范围包括 SAGIN 三层仿真环境、Dueling Double DQN、可靠性感知平滑 CMDP 代价、滑动窗口拉格朗日更新、容量门控的跨层冗余，以及论文采用的主要指标与消融入口。

## 已复现内容

- 论文配置的 10 km × 10 km 场景：100 个设备、10 个边缘服务器、6 架 UAV、3 颗 LEO 卫星。
- 20 个离散动作：本地执行、Edge、UAV、Satellite。
- 四类论文任务分布及数据量、CPU 周期、截止期、可靠性要求。
- 异构链路、节点可用性、排队/计算/传输时延、卫星可见窗口与安全余量。
- 式 (28)–(30) 的链路 × 节点 × 平滑时延复合可靠性。
- 式 (33)–(34) 的 Dueling Double DQN，MLP 为 `[256, 128, 64]`。
- 式 (36)–(37) 的拉格朗日约束更新，以及式 (38)–(42) 的跨层副本与容量门控。
- Random、Greedy-Nearest、Greedy-Reliability、RA-Opt-inspired 单任务枚举近似、DQN、D3QN、DRL-RA 和主要消融。

## 安装与快速验证

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce.py --profile smoke
```

正式训练单个 DRL-RA：

```powershell
python train.py --method drl-ra --seed 0
python evaluate.py --method checkpoint --checkpoint outputs/drl-ra_seed0/model.pt --seeds 0 1 2 3 4 5 6 7 8 9
```

一键跑对照实验：

```powershell
# 日常调试：30 episodes × 3 seeds
python reproduce.py --profile quick

# 论文设置：1000 episodes × 10 seeds，耗时较长
python reproduce.py --profile paper
python plot_results.py
```

训练完成后绘制论文 Figure 4 风格的可靠性阈值敏感性图：

```powershell
python reliability_sweep.py --checkpoint-dir output/experiment_paper/checkpoints --output-dir output/experiment_paper/reliability_sweep
```

脚本在 `0.80/0.85/0.90/0.95/0.98` 五个阈值上重新评估每个 D3QN 和 DRL-RA 检查点，绘制 TCR、CVR 的均值曲线和标准差阴影。当前未实现 Lagrangian-PPO，因此不会伪造该曲线。

所有 YAML 参数都可覆盖，例如：

```powershell
python train.py --set training.episodes=10 --set environment.episode_steps=200
```

输出包括检查点、逐种子 JSON、汇总 CSV 和对比图，默认位于 `outputs/`。

## 与论文文字的差异及假设

这篇论文没有公开作者代码或实验 trace，且 `Availability of Data and Materials` 明确写的是需向通讯作者索取。因此这是基于论文公式和参数的独立复现，无法保证精确得到表 7 的数值。

论文式 (35) 写作 `softplus(ρ_min - R)`，其在零缺口时为 `log(2) ≈ 0.693`，与表 6 的成本预算 `d=0.05` 以及表 8 的 `E[c]=0.043` 数量级矛盾。代码采用温度归一化的平滑正部：

`c = τ_c · softplus((ρ_min - R) / τ_c)`，默认 `τ_c=0.05`。

这保留了论文所述“归一化平滑可靠性短缺”的含义，并使预算可解释；测试也显式锁定这一行为。另有以下合理化处理：

- 论文给出参数范围但未给逐节点实例，均由固定随机种子在范围内采样。
- 论文第 6.1.1 节给出正式训练规模为 1000 episodes × 1000 steps；`paper.yaml` 按此设置，`quick/smoke` profile 仅用于开发验证。
- 每个正式种子训练后独立运行 10,000 个评估决策；训练和评估长度由 `training.episodes`、`environment.episode_steps`、`training.evaluation_steps` 分别控制。
- TCR 同时记录 `deadline_satisfaction_pct` 和实际 `tcr`。默认开启按复合可靠性 Bernoulli 采样的链路/节点成功事件，因此实际 TCR 是“按时且可靠性事件成功”的比例。
- LEO 可见性用周期窗口近似短时星历；不声称复现真实轨道传播。
- 队列以剩余服务时间更新，保持 M/M/1 期望排队思想而避免离散事件模拟的额外未报告假设；UAV 位置使用 5–15 m/s 的真实物理时间轨迹。
- 环境把每一步解释为一个设备策略流中的下一次泊松到达，平均间隔为 `1/12.5 s`；100 台设备并行运行同一策略，而不是把所有设备事件串行压缩成 `1/(100×12.5) s`。这避免了 UAV/卫星时间尺度与物理速度冲突。
- RA-Opt 使用可行动作枚举和单任务 CPU 可行性过滤，是论文批量 MIP 的 RA-Opt-inspired 近似，不宣称等价于完整 CVXPY/ECOS_BB 求解。
- 训练和推理目前共用固定 108 维、带邻居可行性掩码的 observation。它包含论文列出的本地任务、邻居负载、链路估计和窗口信息，但没有把训练全局状态与“仅 30%–40% 全局维度”的执行 observation 做成两种不同张量；这是当前 CTDE 复现边界。
- 地面、空中、卫星链路实现了论文给出的模型结构、参数范围和可靠性组合，但雨衰、星历和瞬时信道仍由固定种子仿真，不是基于真实城市测量或完整轨道传播器的校准结果。

## 代码结构

- `drl_ra/environment.py`：SAGIN 环境、可靠性、冗余、指标。
- `drl_ra/models.py`：Dueling Q 网络与动作掩码。
- `drl_ra/agent.py`：Double Q、回放、目标网络与拉格朗日更新。
- `train.py` / `evaluate.py`：单模型训练和评估。
- `reproduce.py`：多方法、多种子实验。
- `configs/paper.yaml`：论文超参数和环境配置。
- `tests/`：环境、网络、动作掩码、Double-Q 数值目标、代价尺度、固定设备类别、独立本地队列、UAV 速度、卫星窗口、CPU 容量门控、预留/使用资源释放和训练更新测试。

## 复现边界

论文表 7 还列出 PPO、CPO、RCPO、Lagrangian-PPO 和 FOCOPS。它们不是论文核心新方法，且完整稳定实现会引入额外框架与大量未报告调参；当前仓库聚焦复现 DRL-RA 核心、D3QN 组件、启发式基线和关键消融。代码没有硬编码论文结果，所有指标均来自实际仿真运行。
