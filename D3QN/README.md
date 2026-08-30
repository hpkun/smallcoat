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
- Random、Greedy-Nearest、Greedy-Reliability、RA-Opt（单任务枚举等价实现）、DQN、D3QN、DRL-RA 和主要消融。

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

# 默认完整设置：300 episodes × 10 seeds，耗时较长
python reproduce.py --profile paper
python plot_results.py
```

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
- 论文没有报告训练 episode 数；默认完整配置采用 300 episodes，可通过 YAML 或 `--set` 调整。
- LEO 可见性用周期窗口近似短时星历；不声称复现真实轨道传播。
- 队列以剩余服务时间更新，保持 M/M/1 期望排队思想而避免离散事件模拟的额外未报告假设。
- RA-Opt 在单任务决策时使用可行动作枚举；这等价于其二元卸载变量的单任务子问题，不依赖 ECOS_BB。

## 代码结构

- `drl_ra/environment.py`：SAGIN 环境、可靠性、冗余、指标。
- `drl_ra/models.py`：Dueling Q 网络与动作掩码。
- `drl_ra/agent.py`：Double Q、回放、目标网络与拉格朗日更新。
- `train.py` / `evaluate.py`：单模型训练和评估。
- `reproduce.py`：多方法、多种子实验。
- `configs/paper.yaml`：论文超参数和环境配置。
- `tests/`：环境、网络、动作掩码、代价尺度和训练更新测试。

## 复现边界

论文表 7 还列出 PPO、CPO、RCPO、Lagrangian-PPO 和 FOCOPS。它们不是论文核心新方法，且完整稳定实现会引入额外框架与大量未报告调参；当前仓库聚焦完整复现 DRL-RA、D3QN 组件、启发式基线和关键消融。代码没有硬编码论文结果，所有指标均来自实际仿真运行。
