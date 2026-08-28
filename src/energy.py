from __future__ import annotations

from dataclasses import dataclass

from .communication import LinkProfile
from .task_model import Task


@dataclass(frozen=True)
class EnergyConfig:
    """各类计算节点的有效开关电容系数，单位由 ``kappa * f^2 * C`` 决定。"""

    # 等效能耗频率与节点的聚合计算吞吐量分离。
    uav_energy_clock_hz: float = 10e9
    bs_energy_clock_hz: float = 10e9
    leo_energy_clock_hz: float = 10e9
    uav_switching_capacitance: float = 1e-27
    bs_switching_capacitance: float = 1e-27
    leo_switching_capacitance: float = 1e-27
    uav_battery_capacity_j: float = 150_000.0
    uav_safe_energy_ratio: float = 0.15
    uav_propulsion_power_w: float = 100.0

    def __post_init__(self) -> None:
        if self.uav_energy_clock_hz <= 0:
            raise ValueError("uav_energy_clock_hz must be positive.")
        if self.bs_energy_clock_hz <= 0:
            raise ValueError("bs_energy_clock_hz must be positive.")
        if self.leo_energy_clock_hz <= 0:
            raise ValueError("leo_energy_clock_hz must be positive.")
        if self.uav_battery_capacity_j <= 0:
            raise ValueError("uav_battery_capacity_j must be positive.")
        if not 0.0 <= self.uav_safe_energy_ratio < 1.0:
            raise ValueError("uav_safe_energy_ratio must be in [0, 1).")
        if self.uav_propulsion_power_w < 0:
            raise ValueError("uav_propulsion_power_w must be non-negative.")

    def coefficient_for(self, node_type: str) -> float:
        """根据目标节点类型返回计算能耗系数。"""

        coefficients = {
            "uav": self.uav_switching_capacitance,
            "bs": self.bs_switching_capacitance,
            "leo": self.leo_switching_capacitance,
        }
        try:
            coefficient = coefficients[node_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported compute node type: {node_type}") from exc
        if coefficient < 0:
            raise ValueError("Switching capacitance coefficient must be non-negative.")
        return coefficient

    def energy_clock_for(self, node_type: str) -> float:
        """返回分层等效能耗频率，不使用节点总计算吞吐量。"""

        clocks = {
            "uav": self.uav_energy_clock_hz,
            "bs": self.bs_energy_clock_hz,
            "leo": self.leo_energy_clock_hz,
        }
        try:
            return float(clocks[node_type])
        except KeyError as exc:
            raise ValueError(f"Unsupported compute node type: {node_type}") from exc


@dataclass(frozen=True)
class EnergyBreakdown:
    """单个任务副本的传输、计算和总能耗，单位均为焦耳。"""

    transmission_energy_j: float
    computing_energy_j: float
    total_energy_j: float


class EnergyModel:
    """离散任务卸载能耗模型。"""

    def __init__(self, config: EnergyConfig | None = None) -> None:
        self.config = config or EnergyConfig()

    @staticmethod
    def transmission_energy_j(
        transmission_delay_s: float,
        profile: LinkProfile | None,
    ) -> float:
        """按 E_tx = P_tx * T_tx 计算链路传输能耗，不计传播时延。"""

        if transmission_delay_s < 0:
            raise ValueError("transmission_delay_s must be non-negative.")
        if profile is None:
            if transmission_delay_s > 0:
                raise ValueError("A link profile is required for non-zero transmission delay.")
            return 0.0
        if profile.transmit_power_w < 0:
            raise ValueError("transmit_power_w must be non-negative.")
        return float(profile.transmit_power_w * transmission_delay_s)

    def computing_energy_j(
        self,
        task: Task,
        node_type: str,
    ) -> float:
        """
        计算动态 CPU 能耗。

        节点服务时延仍由总计算吞吐量决定；这里仅使用分层等效能耗频率：
        E_comp = kappa_layer * energy_clock^2 * C / rho。
        """

        if task.total_compute_cycles < 0:
            raise ValueError("total_compute_cycles must be non-negative.")
        if not 0 < task.parallel_efficiency <= 1:
            raise ValueError("parallel_efficiency must be in (0, 1].")
        coefficient = self.config.coefficient_for(node_type)
        energy_clock_hz = self.config.energy_clock_for(node_type)
        return float(
            coefficient
            * energy_clock_hz**2
            * task.total_compute_cycles
            / task.parallel_efficiency
        )

    def compute(
        self,
        task: Task,
        node_type: str,
        backhaul_transmission_delay_s: float,
        backhaul_profile: LinkProfile | None,
    ) -> EnergyBreakdown:
        """汇总一个任务副本的上行传输能耗与目标节点计算能耗。"""

        transmission = self.transmission_energy_j(
            backhaul_transmission_delay_s,
            backhaul_profile,
        )
        computing = self.computing_energy_j(
            task,
            node_type,
        )
        return EnergyBreakdown(
            transmission_energy_j=transmission,
            computing_energy_j=computing,
            total_energy_j=float(transmission + computing),
        )
