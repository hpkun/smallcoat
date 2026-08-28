from __future__ import annotations

from dataclasses import dataclass
import math

from .entities import Position


@dataclass(frozen=True)
class LinkProfile:
    """
    链路参数配置。

    尽量贴近论文通信模型中的符号：
    - B: 带宽
    - P_T: 发射功率
    - G_T / G_R: 发射 / 接收天线增益
    - P_N: 噪声功率
    - n: 路径损耗指数
    - X_sigma: 高斯阴影衰落项
    - d0: 参考距离
    - f: 载波频率
    """

    bandwidth_hz: float
    carrier_frequency_hz: float
    transmit_power_w: float
    transmit_antenna_gain_linear: float
    receive_antenna_gain_linear: float
    noise_power_w: float
    path_loss_exponent: float
    gaussian_shadowing_db: float = 0.0
    reference_distance_m: float = 1.0
    rain_attenuation_db: float = 0.0
    ignore_propagation_delay: bool = False
    transmission_failure_rate: float = 0.0


@dataclass(frozen=True)
class NetworkProfiles:
    """环境中的链路配置集合。"""

    ground_to_uav: LinkProfile
    uav_to_bs: LinkProfile
    uav_to_leo: LinkProfile


class CommunicationModel:
    """
    贴近论文原式的通信模型。

    对应论文：
    - 路径损耗：式 (3)
    - 传输速率：式 (4)
    - 传输时延：式 (5)
    - 传播时延：式 (6)
    """

    SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

    @staticmethod
    def db_to_linear(value_db: float) -> float:
        """dB 转线性值。"""
        return 10.0 ** (value_db / 10.0)

    @staticmethod
    def wavelength_m(carrier_frequency_hz: float) -> float:
        """计算电磁波波长 lambda_wave = c / f。"""

        if carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive.")
        return CommunicationModel.SPEED_OF_LIGHT_M_PER_S / carrier_frequency_hz

    def path_loss_db(
        self,
        distance_m: float,
        profile: LinkProfile,
    ) -> float:
        """
        计算路径损耗 PL_ij。

        按论文式 (3)：
            PL_ij = 20 log10(4*pi*d0 / lambda_wave)
                    + 10*n*log10(d / d0)
                    + X_sigma

        其中 rain_attenuation_db 作为论文中“雨衰影响”的工程补充项。
        """

        safe_distance_m = max(distance_m, profile.reference_distance_m)
        lambda_wave_m = self.wavelength_m(profile.carrier_frequency_hz)
        reference_term_db = 20.0 * math.log10(
            4.0 * math.pi * profile.reference_distance_m / lambda_wave_m
        )
        distance_term_db = 10.0 * profile.path_loss_exponent * math.log10(
            safe_distance_m / profile.reference_distance_m
        )
        return (
            reference_term_db
            + distance_term_db
            + profile.gaussian_shadowing_db
            + profile.rain_attenuation_db
        )

    def path_loss_linear(
        self,
        distance_m: float,
        profile: LinkProfile,
    ) -> float:
        """将路径损耗从 dB 转为线性值。"""
        return self.db_to_linear(self.path_loss_db(distance_m, profile))

    def link_rate_bps(
        self,
        sender: Position,
        receiver: Position,
        profile: LinkProfile,
    ) -> float:
        """
        计算链路传输速率 R_ij。

        按论文式 (4)：
            R_ij = B * log2(1 + P_T * G_T * G_R / (P_N * PL_ij(d)))
        """

        distance_m = sender.distance_to(receiver)
        path_loss_linear = self.path_loss_linear(distance_m, profile)
        signal_to_noise_ratio = (
            profile.transmit_power_w
            * profile.transmit_antenna_gain_linear
            * profile.receive_antenna_gain_linear
            / (profile.noise_power_w * path_loss_linear)
        )
        return profile.bandwidth_hz * math.log2(1.0 + signal_to_noise_ratio)

    def propagation_delay_s(
        self,
        sender: Position,
        receiver: Position,
        profile: LinkProfile,
    ) -> float:
        """
        计算传播时延 tau_prop。

        按论文式 (6)：
        - 若目标为卫星链路，则 tau_prop = d / c
        - 若目标为 BS 链路，则忽略传播时延
        """

        if profile.ignore_propagation_delay:
            return 0.0
        return sender.distance_to(receiver) / self.SPEED_OF_LIGHT_M_PER_S

    def transmission_delay_s(
        self,
        data_size_bits: float,
        sender: Position,
        receiver: Position,
        profile: LinkProfile,
    ) -> float:
        """
        计算纯传输时延。

        按论文式 (5) 的前半部分：
            d_tran = phi_k / R_ij
        """

        if data_size_bits < 0:
            raise ValueError("data_size_bits must be non-negative.")

        rate_bps = self.link_rate_bps(sender, receiver, profile)
        if rate_bps <= 0:
            raise ValueError("link_rate_bps must be positive.")
        return data_size_bits / rate_bps

    def total_link_delay_s(
        self,
        data_size_bits: float,
        sender: Position,
        receiver: Position,
        profile: LinkProfile,
    ) -> tuple[float, float]:
        """
        计算一条链路的总通信时延组成。

        返回：
        - 纯传输时延
        - 传播时延
        """

        transmission_delay_s = self.transmission_delay_s(
            data_size_bits=data_size_bits,
            sender=sender,
            receiver=receiver,
            profile=profile,
        )
        propagation_delay_s = self.propagation_delay_s(
            sender=sender,
            receiver=receiver,
            profile=profile,
        )
        return transmission_delay_s, propagation_delay_s
