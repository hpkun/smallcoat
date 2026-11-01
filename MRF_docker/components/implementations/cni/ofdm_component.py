"""C15：明确子载波索引的 OFDM 发射/接收，含导频与循环前缀。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, vector


class OFDMComponent(BlockComponent):
    cid = "C15"
    metadata_key = "ofdm"

    def __init__(self, mode="tx", fft_length=64, cp_length=16,
                 subcarrier_allocation=None, pilot_pattern=None, pilot_values=None,
                 guard_carriers=None):
        super().__init__(mode=mode, fft_length=fft_length, cp_length=cp_length,
                         subcarrier_allocation=subcarrier_allocation, pilot_pattern=pilot_pattern,
                         pilot_values=pilot_values, guard_carriers=guard_carriers)

    def _validate_config(self, mode="tx", fft_length=64, cp_length=16,
                         subcarrier_allocation=None, pilot_pattern=None, pilot_values=None,
                         guard_carriers=None):
        if mode not in ("tx", "rx"):
            raise ValueError("mode must be tx or rx")
        integer(fft_length, "fft_length", 2)
        integer(cp_length, "cp_length", 0)
        if cp_length > fft_length:
            raise ValueError("cp_length must not exceed fft_length")
        pilots = [] if pilot_pattern is None else list(pilot_pattern)
        guards = [0] if guard_carriers is None else list(guard_carriers)
        allocation = ([i for i in range(fft_length) if i not in pilots + guards]
                      if subcarrier_allocation is None else list(subcarrier_allocation))
        all_indices = allocation + pilots + guards
        if not allocation or len(set(all_indices)) != len(all_indices):
            raise ValueError("data, pilots and guard indices must be nonempty/disjoint/unique")
        for i in all_indices:
            integer(i, "subcarrier", 0)
            if i >= fft_length:
                raise ValueError("subcarrier index outside FFT")
        values = np.ones(len(pilots), dtype=complex) if pilot_values is None else vector(pilot_values, "pilot_values")
        if len(values) != len(pilots):
            raise ValueError("one pilot_value is required for each pilot index")
        return dict(mode=mode, fft_length=fft_length, cp_length=cp_length,
                    subcarrier_allocation=allocation, pilot_pattern=pilots,
                    pilot_values=values, guard_carriers=guards)

    def _setup(self):
        tx = self._parameters["mode"] == "tx"
        self.input_port = RuntimePort("in", "input", ("symbols" if tx else "time",), ("c",), (1,), "symbols" if tx else "samples")
        self.output_port = RuntimePort("out", "output", ("time" if tx else "symbols",), ("c",), (1,), "samples" if tx else "symbols")

    def _frame_size(self):
        p = self._parameters
        return len(p["subcarrier_allocation"]) if p["mode"] == "tx" else p["fft_length"] + p["cp_length"]

    def _execute(self, samples):
        p = self._parameters
        if p["mode"] == "tx":
            spectrum = np.zeros(p["fft_length"], dtype=complex)
            spectrum[p["subcarrier_allocation"]] = samples
            spectrum[p["pilot_pattern"]] = p["pilot_values"]
            time = self._run_stage("ifft", np.fft.ifft, spectrum)
            output = np.concatenate((time[-p["cp_length"]:], time)) if p["cp_length"] else time
        else:
            spectrum = self._run_stage("fft", np.fft.fft, samples[p["cp_length"]:])
            output = spectrum[p["subcarrier_allocation"]]
        self._business = dict(mode=p["mode"], fft_length=p["fft_length"], cp_length=p["cp_length"],
                              data_indices=p["subcarrier_allocation"], pilot_indices=p["pilot_pattern"],
                              pilot_values_real=spectrum[p["pilot_pattern"]].real.tolist(),
                              pilot_values_imag=spectrum[p["pilot_pattern"]].imag.tolist(),
                              effective_subcarrier_ratio=len(p["subcarrier_allocation"]) / p["fft_length"],
                              normalization="backward")
        return output

    def _output_context(self, packet):
        p = self._parameters
        ratio = (p["fft_length"] + p["cp_length"]) / len(p["subcarrier_allocation"])
        if p["mode"] == "rx":
            ratio = 1 / ratio
        return packet.sample_rate * ratio, packet.center_frequency, packet.timestamp

    # process 每次一个完整 OFDM 符号；push 自动组帧。接收假定已经同步，不含均衡。
