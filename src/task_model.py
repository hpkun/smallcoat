from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


MBIT = 1_000_000
MCYCLE = 1_000_000
MILLISECOND = 1e-3


@dataclass(frozen=True)
class UniformRange:
    """论文中用于任务参数采样的均匀分布区间。"""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """从闭区间 [low, high] 中采样一个浮点数。"""
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class TaskModelConfig:
    """
    论文《Cluster-Based Multi-Agent Task Scheduling for
    Space-Air-Ground Integrated Networks》中的任务模型参数。

    这里显式拆开两套语义，避免论文中符号重用带来的实现歧义：

    1. Section III-B 任务生成语义
       - phi_b: 输入数据大小，单位 bits
       - rho_b: 每比特所需 CPU 周期数，单位 cycles/bit
       - delta: 容忍时延

    2. Section III-C 计算模型语义
       - phi_c: 总计算需求，单位 cycles
       - rho_c: 并行化效率系数，取值范围 (0, 1]

    在工程实现中：
    - `input_size_bits` 对应 III.B 中的数据大小，用于传输时延计算
    - `total_compute_cycles` 对应 III.C 中的 phi_k，用于计算时延
    - `parallel_efficiency` 对应 III.C 中的 rho_k，用于计算时延
    """

    arrival_rate_tasks_per_s: float = 25.0
    input_size_bits: UniformRange = UniformRange(10 * MBIT, 90 * MBIT)
    total_compute_cycles: UniformRange = UniformRange(
        1_000 * MCYCLE,
        3_000 * MCYCLE,
    )
    tolerable_latency_s: UniformRange = UniformRange(
        0 * MILLISECOND,
        200 * MILLISECOND,
    )
    parallel_efficiency: UniformRange = UniformRange(0.8, 1.0)
    expected_reliability: UniformRange = UniformRange(0.90, 0.99)
    delay_sensitivity_lambda: Optional[float] = None

    @property
    def mean_arrivals_per_slot_requires_slot_length(self) -> str:
        """提示：泊松到达率需要结合时隙长度换算到每个时隙。"""
        return (
            "Poisson arrivals use mu_slot = arrival_rate_tasks_per_s * slot_length_s. "
            "The paper excerpt confirms mu = 25 tasks/s, but the timeslot length should "
            "be set from the environment definition before sampling per-slot arrivals."
        )


@dataclass(frozen=True)
class Task:
    """
    任务对象。

    字段设计遵循“工程字段名明确、论文语义可回溯”的原则，
    避免直接复用 phi/rho 这类在不同小节中含义变化的符号。
    """

    input_size_bits: float
    total_compute_cycles: float
    tolerable_latency_s: float
    parallel_efficiency: float
    profit: Optional[float] = None
    expected_reliability: float = 0.95
    forward_count: int = 0

    @property
    def cycles_per_bit(self) -> float:
        """III.B 语义下的每比特计算负荷，单位 cycles/bit。"""
        if self.input_size_bits <= 0:
            raise ValueError("input_size_bits must be positive.")
        return self.total_compute_cycles / self.input_size_bits

    @property
    def paper_phi_b(self) -> float:
        """映射论文 III.B 中的 phi：输入数据大小。"""
        return self.input_size_bits

    @property
    def paper_rho_b(self) -> float:
        """映射论文 III.B 中的 rho：每比特计算负荷。"""
        return self.cycles_per_bit

    @property
    def paper_phi_c(self) -> float:
        """映射论文 III.C 中的 phi_k：总计算需求。"""
        return self.total_compute_cycles

    @property
    def paper_rho_c(self) -> float:
        """映射论文 III.C 中的 rho_k：并行化效率系数。"""
        return self.parallel_efficiency


def compute_task_profit(
    input_size_bits: float,
    cycles_per_bit: float,
    tolerable_latency_s: float,
    delay_sensitivity_lambda: float,
) -> float:
    """
    按论文 Section III-B 的收益公式计算任务收益。

    使用 III.B 的原始语义：
    G = phi * rho * exp(-lambda * delta)
    其中：
    - phi 是输入数据大小 bits
    - rho 是每比特计算负荷 cycles/bit
    """

    if input_size_bits < 0:
        raise ValueError("input_size_bits must be non-negative.")
    if cycles_per_bit < 0:
        raise ValueError("cycles_per_bit must be non-negative.")
    if tolerable_latency_s < 0:
        raise ValueError("tolerable_latency_s must be non-negative.")
    if delay_sensitivity_lambda < 0:
        raise ValueError("delay_sensitivity_lambda must be non-negative.")

    return float(
        input_size_bits * cycles_per_bit * math.exp(-delay_sensitivity_lambda * tolerable_latency_s)
    )


def compute_cycles_per_bit(
    input_size_bits: float,
    total_compute_cycles: float,
) -> float:
    """根据总计算需求和输入数据大小反推每比特计算负荷。"""

    if input_size_bits <= 0:
        raise ValueError("input_size_bits must be positive.")
    if total_compute_cycles < 0:
        raise ValueError("total_compute_cycles must be non-negative.")

    return total_compute_cycles / input_size_bits


def compute_computing_delay(
    total_compute_cycles: float,
    device_compute_rate_cycles_per_s: float,
    parallel_efficiency: float,
    offload_indicator: float = 1.0,
) -> float:
    """
    按论文 III.C 的计算模型计算纯计算时延。

    对应公式：
        d_{k,j}^c = alpha_{k,j} * phi_k / (rho_k * f_j)

    其中：
    - phi_k 对应 total_compute_cycles
    - rho_k 对应 parallel_efficiency
    - f_j 为目标设备算力，单位 cycles/s
    """

    if total_compute_cycles < 0:
        raise ValueError("total_compute_cycles must be non-negative.")
    if device_compute_rate_cycles_per_s <= 0:
        raise ValueError("device_compute_rate_cycles_per_s must be positive.")
    if not (0 < parallel_efficiency <= 1):
        raise ValueError("parallel_efficiency must be in (0, 1].")
    if offload_indicator < 0:
        raise ValueError("offload_indicator must be non-negative.")

    return float(
        offload_indicator
        * total_compute_cycles
        / (parallel_efficiency * device_compute_rate_cycles_per_s)
    )


def sample_task(
    config: TaskModelConfig,
    rng: np.random.Generator,
    *,
    delay_sensitivity_lambda: Optional[float] = None,
) -> Task:
    """
    按论文中的实验分布采样一个任务。

    当前采用的分布假设：
    - 到达过程：Poisson(mu = 25 tasks/s)
    - 任务大小：Uniform(10, 90) Mbits
    - 总计算需求：Uniform(1000, 3000) M cycles
    - 容忍时延：Uniform(0, 200) ms
    - 并行化效率：额外工程属性，默认 Uniform(0.8, 1.0)
    """

    input_size_bits = config.input_size_bits.sample(rng)
    total_compute_cycles = config.total_compute_cycles.sample(rng)
    tolerable_latency_s = config.tolerable_latency_s.sample(rng)
    parallel_efficiency = config.parallel_efficiency.sample(rng)
    expected_reliability = config.expected_reliability.sample(rng)

    resolved_lambda = (
        delay_sensitivity_lambda
        if delay_sensitivity_lambda is not None
        else config.delay_sensitivity_lambda
    )

    profit = None
    if resolved_lambda is not None:
        profit = compute_task_profit(
            input_size_bits=input_size_bits,
            cycles_per_bit=compute_cycles_per_bit(
                input_size_bits=input_size_bits,
                total_compute_cycles=total_compute_cycles,
            ),
            tolerable_latency_s=tolerable_latency_s,
            delay_sensitivity_lambda=resolved_lambda,
        )

    return Task(
        input_size_bits=input_size_bits,
        total_compute_cycles=total_compute_cycles,
        tolerable_latency_s=tolerable_latency_s,
        parallel_efficiency=parallel_efficiency,
        profit=profit,
        expected_reliability=expected_reliability,
    )


def sample_num_arrivals(
    config: TaskModelConfig,
    slot_length_s: float,
    rng: np.random.Generator,
) -> int:
    """在一个时隙内采样到达任务数。"""

    if slot_length_s <= 0:
        raise ValueError("slot_length_s must be positive.")

    # 泊松分布参数需要从“每秒到达率”换算成“每时隙到达率”。
    mean_arrivals = config.arrival_rate_tasks_per_s * slot_length_s
    return int(rng.poisson(mean_arrivals))
