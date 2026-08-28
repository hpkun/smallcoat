from __future__ import annotations

from dataclasses import dataclass, field
import math

from .communication import LinkProfile
from .communication import NetworkProfiles
from .energy import EnergyConfig


@dataclass(frozen=True)
class AreaConfig:
    """二维仿真区域配置。"""

    side_length_m: float = 5_000.0

    @property
    def area_m2(self) -> float:
        """区域面积。"""
        return self.side_length_m * self.side_length_m


@dataclass(frozen=True)
class MobilityConfig:
    """UAV 运动模型配置。"""

    mean_speed_m_per_s: float = 20.0
    std_speed_m_per_s: float = 5.0
    max_turn_angle_rad: float = math.pi / 9.0


@dataclass(frozen=True)
class ClusteringConfig:
    """KMDUC 聚类与维护配置。"""

    communication_radius_m: float = 1_000.0
    coverage_threshold_pmax: float = 0.9
    logistic_zeta: float = 0.02
    clustering_period_slots: int = 10
    ch_reselection_slots: int = 3
    kmeans_max_iterations: int = 50


def build_default_network_profiles() -> NetworkProfiles:
    """构建一组更贴论文原式的默认链路配置。"""

    # 为了贴近论文式 (4)，这里直接使用线性增益和线性噪声功率。
    # 噪声功率采用一个固定工程量级，用于保持当前小规模实验可运行。
    common_noise_power_w = 1e-13

    return NetworkProfiles(
        ground_to_uav=LinkProfile(
            bandwidth_hz=20e6,
            carrier_frequency_hz=2.4e9,
            transmit_power_w=0.5,
            transmit_antenna_gain_linear=2.0,
            receive_antenna_gain_linear=2.0,
            noise_power_w=common_noise_power_w,
            path_loss_exponent=2.2,
            gaussian_shadowing_db=1.0,
            ignore_propagation_delay=True,
            transmission_failure_rate=0.02,
        ),
        uav_to_bs=LinkProfile(
            bandwidth_hz=40e6,
            carrier_frequency_hz=3.5e9,
            transmit_power_w=2.0,
            transmit_antenna_gain_linear=4.0,
            receive_antenna_gain_linear=6.3,
            noise_power_w=common_noise_power_w,
            path_loss_exponent=2.5,
            gaussian_shadowing_db=1.5,
            ignore_propagation_delay=True,
            transmission_failure_rate=0.03,
        ),
        uav_to_leo=LinkProfile(
            bandwidth_hz=100e6,
            carrier_frequency_hz=20e9,
            transmit_power_w=10.0,
            transmit_antenna_gain_linear=63.1,
            receive_antenna_gain_linear=316.2,
            noise_power_w=common_noise_power_w,
            path_loss_exponent=2.0,
            rain_attenuation_db=2.0,
            ignore_propagation_delay=False,
            transmission_failure_rate=0.05,
        ),
    )


@dataclass(frozen=True)
class QueueCapacityConfig:
    """不同计算层允许的最大队列工作量，单位为秒。"""

    uav_max_workload_s: float = 0.40
    bs_max_workload_s: float = 0.20
    leo_max_workload_s: float = 0.10

    def __post_init__(self) -> None:
        if self.uav_max_workload_s <= 0.0:
            raise ValueError("uav_max_workload_s must be positive.")
        if self.bs_max_workload_s <= 0.0:
            raise ValueError("bs_max_workload_s must be positive.")
        if self.leo_max_workload_s <= 0.0:
            raise ValueError("leo_max_workload_s must be positive.")

    def limit_for(self, node_type: str) -> float:
        """根据 UAV、BS 或 LEO 节点类型返回对应队列阈值。"""

        limits = {
            "uav": self.uav_max_workload_s,
            "bs": self.bs_max_workload_s,
            "leo": self.leo_max_workload_s,
        }
        try:
            return float(limits[node_type])
        except KeyError as exc:
            raise ValueError(f"Unsupported compute node type: {node_type}") from exc


@dataclass(frozen=True)
class SimulationConfig:
    """统一的仿真配置中心。"""

    slot_length_s: float = 0.1
    rng_seed: int = 42
    # Preserve the original low-level environment default for backwards-compatible
    # unit scenarios; paper-scale builders opt into QueueCapacityConfig defaults.
    queue_capacity: QueueCapacityConfig = field(
        default_factory=lambda: QueueCapacityConfig(
            uav_max_workload_s=0.35,
            bs_max_workload_s=0.15,
            leo_max_workload_s=0.08,
        )
    )
    area: AreaConfig = field(default_factory=AreaConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    network_profiles: NetworkProfiles = field(default_factory=build_default_network_profiles)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
