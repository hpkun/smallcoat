"""C04: sample-wise complex-IQ automatic gain control."""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import real


class AGCComponent(BlockComponent):
    cid = "C04"
    metadata_key = "agc"

    def __init__(self, target_amplitude=1.0, attack_rate=0.05, release_rate=0.01,
                 initial_gain=1.0, min_gain=0.01, max_gain=100.0):
        super().__init__(target_amplitude=target_amplitude, attack_rate=attack_rate,
                         release_rate=release_rate, initial_gain=initial_gain,
                         min_gain=min_gain, max_gain=max_gain)

    def _validate_config(self, target_amplitude=1.0, attack_rate=0.05, release_rate=0.01,
                         initial_gain=1.0, min_gain=0.01, max_gain=100.0):
        values = {
            "target_amplitude": real(target_amplitude, "target_amplitude", 0),
            "attack_rate": real(attack_rate, "attack_rate", 0),
            "release_rate": real(release_rate, "release_rate", 0),
            "initial_gain": real(initial_gain, "initial_gain", 0),
            "min_gain": real(min_gain, "min_gain", 0),
            "max_gain": real(max_gain, "max_gain", 0),
        }
        for name in ("target_amplitude", "min_gain", "max_gain"):
            if values[name] <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("attack_rate", "release_rate"):
            if not 0 < values[name] <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not values["min_gain"] <= values["initial_gain"] <= values["max_gain"]:
            raise ValueError("gain must satisfy min_gain <= initial_gain <= max_gain")
        return values

    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    def _execute(self, samples):
        p = self._parameters
        gain = self._state.get("gain", p["initial_gain"])
        output = np.empty_like(samples, dtype=complex)
        for index, sample in enumerate(samples):
            output[index] = sample * gain
            error = p["target_amplitude"] - abs(output[index])
            rate = p["attack_rate"] if error < 0 else p["release_rate"]
            gain = float(np.clip(gain + rate * error, p["min_gain"], p["max_gain"]))
        self._state["gain"] = gain
        self._business = {"final_gain": gain, "target_amplitude": p["target_amplitude"]}
        return output
