"""C17：二进制卷积码与硬/软输入 Viterbi，首版不实现 LDPC/Turbo。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import bits, integer


class FECComponent(BlockComponent):
    cid = "C17"
    metadata_key = "fec"

    def __init__(self, mode="encode", block_length=128, constraint_length=7,
                 generators=(0o171, 0o133), hard_or_soft="hard", termination="zero",
                 codec_type="convolutional"):
        super().__init__(mode=mode, block_length=block_length, constraint_length=constraint_length,
                         generators=generators, hard_or_soft=hard_or_soft,
                         termination=termination, codec_type=codec_type)

    def _validate_config(self, mode="encode", block_length=128, constraint_length=7,
                         generators=(0o171, 0o133), hard_or_soft="hard", termination="zero",
                         codec_type="convolutional"):
        if mode not in ("encode", "decode") or codec_type != "convolutional":
            raise ValueError("only convolutional encode/decode is implemented")
        integer(block_length, "block_length")
        integer(constraint_length, "constraint_length", 2)
        if constraint_length > 9:
            raise ValueError("reference Viterbi supports constraint_length <= 9")
        generators = tuple(generators)
        if len(generators) < 2:
            raise ValueError("at least two generator polynomials are required")
        for g in generators:
            integer(g, "generator")
            if g >= 1 << constraint_length:
                raise ValueError("generator exceeds constraint length")
        if not any(g & (1 << (constraint_length - 1)) for g in generators):
            raise ValueError("generators do not use the stated constraint length")
        if hard_or_soft not in ("hard", "soft") or termination not in ("zero", "none"):
            raise ValueError("unsupported decision or termination")
        return dict(mode=mode, block_length=block_length, constraint_length=constraint_length,
                    generators=generators, hard_or_soft=hard_or_soft,
                    termination=termination, codec_type=codec_type)

    def _setup(self):
        p = self._parameters
        self._states = 1 << (p["constraint_length"] - 1)
        self._tails = p["constraint_length"] - 1 if p["termination"] == "zero" else 0
        self._outputs = np.empty((self._states, 2, len(p["generators"])), dtype=np.uint8)
        for state in range(self._states):
            for bit in (0, 1):
                register = (state << 1) | bit
                self._outputs[state, bit] = [(register & g).bit_count() % 2 for g in p["generators"]]
        soft = p["mode"] == "decode" and p["hard_or_soft"] == "soft"
        self.input_port = RuntimePort("in", "input", ("llr" if soft else "bits",),
                                      ("f",) if soft else ("b", "i", "u"), (1,), "bits")
        self.output_port = RuntimePort("out", "output", ("bits",), ("u",), (1,), "bits")

    def _frame_size(self):
        p = self._parameters
        return p["block_length"] if p["mode"] == "encode" else (p["block_length"] + self._tails) * len(p["generators"])

    def _execute(self, samples):
        p = self._parameters
        width = len(p["generators"])
        if p["mode"] == "encode":
            data = np.concatenate((bits(samples), np.zeros(self._tails, dtype=np.uint8)))
            output = np.empty((len(data), width), dtype=np.uint8)
            state = 0
            for i, bit in enumerate(data):
                output[i] = self._outputs[state, bit]
                state = ((state << 1) | int(bit)) & (self._states - 1)
            output = output.reshape(-1)
            path_metric, work_bytes = 0.0, output.nbytes
        else:
            received = (bits(samples) if p["hard_or_soft"] == "hard" else samples).reshape(-1, width)
            costs = np.full(self._states, np.inf)
            costs[0] = 0
            destinations = np.arange(self._states)
            input_bits = destinations & 1
            first = destinations >> 1
            second = first + self._states // 2
            trace = np.empty((len(received), self._states), dtype=bool)
            for t, word in enumerate(received):
                if p["hard_or_soft"] == "hard":
                    a = costs[first] + np.count_nonzero(self._outputs[first, input_bits] != word, axis=1)
                    b = costs[second] + np.count_nonzero(self._outputs[second, input_bits] != word, axis=1)
                else:
                    # 与独立位 softplus 负对数似然仅差一个分支无关常数。
                    a = costs[first] + self._outputs[first, input_bits] @ word
                    b = costs[second] + self._outputs[second, input_bits] @ word
                trace[t] = b < a
                costs = np.minimum(a, b)
            state = 0 if self._tails else int(np.argmin(costs))
            path_metric = float(costs[state])
            decoded = np.empty(len(received), dtype=np.uint8)
            for t in range(len(received) - 1, -1, -1):
                decoded[t] = state & 1
                state = (state >> 1) + (self._states // 2 if trace[t, state] else 0)
            output = decoded[:p["block_length"]]
            work_bytes = trace.nbytes + costs.nbytes
        self._business = dict(codec_type="convolutional", mode=p["mode"], code_rate=1 / width,
                              effective_code_rate=p["block_length"] / ((p["block_length"] + self._tails) * width),
                              block_length=p["block_length"], tail_bits=self._tails,
                              decoder="viterbi", actual_iterations=0, path_metric=path_metric,
                              decode_status="decoded_without_crc" if p["mode"] == "decode" else "encoded",
                              trellis_workspace_bytes=work_bytes)
        return output

    def _output_context(self, packet):
        p = self._parameters
        encoded = (p["block_length"] + self._tails) * len(p["generators"])
        ratio = encoded / p["block_length"]
        if p["mode"] == "decode":
            ratio = 1 / ratio
        return packet.sample_rate * ratio, packet.center_frequency, packet.timestamp

    # 每个码块从零状态开始；push 可缓存不完整码块，flush 不补比特。


FECCodecComponent = FECComponent
