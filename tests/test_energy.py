from __future__ import annotations

import math

from src.communication import LinkProfile
from src.energy import EnergyConfig
from src.energy import EnergyModel
from src.task_model import Task


def _task(*, parallel_efficiency: float = 1.0) -> Task:
    return Task(
        input_size_bits=1_000_000.0,
        total_compute_cycles=10_000_000.0,
        tolerable_latency_s=1.0,
        parallel_efficiency=parallel_efficiency,
    )


def _profile(power_w: float) -> LinkProfile:
    return LinkProfile(
        bandwidth_hz=1.0,
        carrier_frequency_hz=1.0,
        transmit_power_w=power_w,
        transmit_antenna_gain_linear=1.0,
        receive_antenna_gain_linear=1.0,
        noise_power_w=1.0,
        path_loss_exponent=2.0,
    )


def test_transmission_energy_uses_only_transmission_time() -> None:
    model = EnergyModel()

    energy = model.transmission_energy_j(2.5, _profile(power_w=4.0))

    assert energy == 10.0


def test_computing_energy_accounts_for_parallel_efficiency() -> None:
    model = EnergyModel(
        EnergyConfig(
            uav_energy_clock_hz=10e9,
            uav_switching_capacitance=2e-27,
        )
    )

    energy = model.computing_energy_j(
        _task(parallel_efficiency=0.5),
        node_type="uav",
    )

    expected = 2e-27 * (10e9) ** 2 * 10_000_000.0 / 0.5
    assert math.isclose(energy, expected)


def test_local_uav_execution_has_no_transmission_energy() -> None:
    model = EnergyModel()

    energy = model.compute(
        task=_task(),
        node_type="uav",
        backhaul_transmission_delay_s=0.0,
        backhaul_profile=None,
    )

    assert energy.transmission_energy_j == 0.0
    assert energy.total_energy_j == energy.computing_energy_j


def test_equal_layer_energy_clocks_give_equal_computing_energy() -> None:
    model = EnergyModel(
        EnergyConfig(
            uav_energy_clock_hz=10e9,
            bs_energy_clock_hz=10e9,
            leo_energy_clock_hz=10e9,
        )
    )

    energies = [
        model.computing_energy_j(_task(), node_type=node_type)
        for node_type in ("uav", "bs", "leo")
    ]

    assert energies[0] == energies[1] == energies[2]
