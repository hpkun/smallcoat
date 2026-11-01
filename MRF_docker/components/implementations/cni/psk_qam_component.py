"""C14：Gray 星座、矩形脉冲成形、硬判决与 max-log LLR。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import bits, integer, real


class PSKQAMComponent(BlockComponent):
    cid = "C14"
    metadata_key = "modem"

    def __init__(self, mode="modulate", modulation="qpsk", order=None,
                 decision_mode="hard", mapping="gray", samples_per_symbol=1,
                 pulse_shaping=None, noise_variance=1.0):
        super().__init__(mode=mode, modulation=modulation, order=order, decision_mode=decision_mode,
                         mapping=mapping, samples_per_symbol=samples_per_symbol,
                         pulse_shaping=pulse_shaping, noise_variance=noise_variance)

    def _validate_config(self, mode="modulate", modulation="qpsk", order=None,
                         decision_mode="hard", mapping="gray", samples_per_symbol=1,
                         pulse_shaping=None, noise_variance=1.0):
        if mode not in ("modulate", "demodulate") or modulation not in ("bpsk", "qpsk", "qam"):
            raise ValueError("unsupported modem mode or modulation")
        expected = {"bpsk": 2, "qpsk": 4}
        order = expected.get(modulation, 16) if order is None else integer(order, "order", 2)
        if modulation in expected and order != expected[modulation]:
            raise ValueError("order does not match modulation")
        if modulation == "qam" and order not in (4, 16, 64, 256):
            raise ValueError("QAM supports square orders 4,16,64,256")
        if decision_mode not in ("hard", "soft") or mapping != "gray":
            raise ValueError("use gray mapping and hard/soft decision")
        integer(samples_per_symbol, "samples_per_symbol")
        if pulse_shaping not in (None, "rect") or samples_per_symbol != 1 and pulse_shaping != "rect":
            raise ValueError("oversampling requires rect pulse shaping")
        noise_variance = real(noise_variance, "noise_variance", 0)
        if noise_variance == 0:
            raise ValueError("noise_variance must be > 0")
        return dict(mode=mode, modulation=modulation, order=order, decision_mode=decision_mode,
                    mapping=mapping, samples_per_symbol=samples_per_symbol,
                    pulse_shaping=pulse_shaping, noise_variance=noise_variance)

    def _setup(self):
        p = self._parameters
        self._k = int(np.log2(p["order"]))
        labels = np.arange(p["order"])
        self._labels = ((labels[:, None] >> np.arange(self._k - 1, -1, -1)) & 1).astype(np.uint8)
        self._constellation = np.empty(p["order"], dtype=complex)
        if p["modulation"] in ("bpsk", "qpsk"):
            for k in range(p["order"]):
                self._constellation[k ^ (k >> 1)] = np.exp(1j * (2 * np.pi * k / p["order"] + (np.pi / 4 if self._k == 2 else 0)))
        else:
            side = int(np.sqrt(p["order"]))
            for i in range(side):
                for q in range(side):
                    label = ((i ^ (i >> 1)) << (self._k // 2)) | (q ^ (q >> 1))
                    self._constellation[label] = ((2 * i - side + 1) + 1j * (2 * q - side + 1)) / np.sqrt(2 * (p["order"] - 1) / 3)
        signal_domain = "time" if p["pulse_shaping"] == "rect" else "symbols"
        signal_unit = "samples" if signal_domain == "time" else "symbols"
        if p["mode"] == "modulate":
            self.input_port = RuntimePort("in", "input", ("bits",), ("b", "i", "u"), (1,), "bits")
            self.output_port = RuntimePort("out", "output", (signal_domain,), ("c",), (1,), signal_unit)
        else:
            self.input_port = RuntimePort("in", "input", (signal_domain,), ("c",), (1,), signal_unit)
            self.output_port = RuntimePort("out", "output", ("bits" if p["decision_mode"] == "hard" else "llr",),
                                           ("u",) if p["decision_mode"] == "hard" else ("f",), (1,), "bits")

    def _frame_size(self):
        return self._k if self._parameters["mode"] == "modulate" else self._parameters["samples_per_symbol"]

    def _validate_samples(self, samples):
        if not samples.size or len(samples) % self._frame_size():
            raise ValueError("input must contain complete bit groups or shaped symbols")

    def _execute(self, samples):
        p = self._parameters
        if p["mode"] == "modulate":
            values = bits(samples).reshape(-1, self._k)
            indices = values @ (1 << np.arange(self._k - 1, -1, -1))
            symbols = self._constellation[indices]
            output = np.repeat(symbols, p["samples_per_symbol"])
        else:
            symbols = samples.reshape(-1, p["samples_per_symbol"]).mean(axis=1)
            distance = abs(symbols[:, None] - self._constellation[None, :]) ** 2
            if p["decision_mode"] == "hard":
                output = self._labels[np.argmin(distance, axis=1)].reshape(-1)
            else:
                # LLR = log(P(bit=0)/P(bit=1))；noise_variance 是判决符号处 E|n|²。
                output = np.stack([(distance[:, self._labels[:, b] == 1].min(axis=1)
                                    - distance[:, self._labels[:, b] == 0].min(axis=1)) / p["noise_variance"]
                                   for b in range(self._k)], axis=1).reshape(-1)
        self._business = dict(mode=p["mode"], modulation=p["modulation"], order=p["order"],
                              bit_count=len(symbols) * self._k, symbol_count=len(symbols),
                              llr_convention="positive_means_bit_zero", average_constellation_power=1.0)
        return output

    def _output_context(self, packet):
        ratio = self._parameters["samples_per_symbol"] / self._k
        if self._parameters["mode"] == "demodulate":
            ratio = 1 / ratio
        return packet.sample_rate * ratio, packet.center_frequency, packet.timestamp

    # 公共 process 处理多符号块；push 缓存不完整的比特组/过采样符号。


PSKQAMModemComponent = PSKQAMComponent
