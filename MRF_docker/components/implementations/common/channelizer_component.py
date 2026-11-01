"""C09：DFT 调制分析滤波器组，采用与多相结构等价的直接分支参考计算。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import fir, integer, lowpass, vector


class ChannelizerComponent(BlockComponent):
    cid = "C09"
    metadata_key = "channelizer"

    def __init__(self, channel_count=8, prototype_filter=None, decimation=None,
                 channel_map=None, taps_per_channel=8):
        super().__init__(channel_count=channel_count, prototype_filter=prototype_filter,
                         decimation=decimation, channel_map=channel_map, taps_per_channel=taps_per_channel)

    def _validate_config(self, channel_count=8, prototype_filter=None, decimation=None,
                         channel_map=None, taps_per_channel=8):
        integer(channel_count, "channel_count", 2)
        decimation = channel_count if decimation is None else integer(decimation, "decimation")
        if decimation > channel_count:
            raise ValueError("decimation must not exceed channel_count")
        integer(taps_per_channel, "taps_per_channel", 2)
        if prototype_filter is not None:
            prototype_filter = vector(prototype_filter, "prototype_filter")
        if channel_map is None:
            channel_map = list(range(channel_count))
        if not channel_map or len(set(channel_map)) != len(channel_map):
            raise ValueError("channel_map must be nonempty and unique")
        for channel in channel_map:
            integer(channel, "channel", 0)
            if channel >= channel_count:
                raise ValueError("channel_map entry out of range")
        return dict(channel_count=channel_count, prototype_filter=prototype_filter,
                    decimation=decimation, channel_map=list(channel_map), taps_per_channel=taps_per_channel)

    def _setup(self):
        p = self._parameters
        taps = p["taps_per_channel"] * p["channel_count"]
        taps += 1 if taps % 2 == 0 else 0
        self._h = (lowpass(taps, 0.9 / p["channel_count"]) if p["prototype_filter"] is None
                   else p["prototype_filter"])
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1, 2))

    def _execute(self, samples):
        p, packet = self._parameters, self._active_packet
        count = self._state.get("input_count", 0)
        history = self._state.get("branches", {})
        t = np.arange(len(samples)) + count
        first = (-count) % p["decimation"]
        normalized = np.fft.fftfreq(p["channel_count"])
        outputs, channels = [], []
        for channel in p["channel_map"]:
            offset = float(normalized[channel])
            mixed = samples * np.exp(-2j * np.pi * offset * t)
            filtered, history[channel] = fir(mixed, self._h, history.get(channel))
            output = filtered[first::p["decimation"]]
            outputs.append(output)
            channels.append(dict(channel_id=channel, center_frequency_hz=packet.center_frequency + offset * packet.sample_rate,
                                 sample_rate=packet.sample_rate / p["decimation"],
                                 bandwidth_hz=packet.sample_rate / p["channel_count"],
                                 output_samples=len(output), data_bytes=output.nbytes))
        self._state.update(input_count=count + len(samples), branches=history)
        self._business = dict(channels=channels, channel_count=len(channels),
                              first_input_offset=first, prototype_taps=len(self._h),
                              state_bytes=sum(x.nbytes for x in history.values()),
                              layout="channels_by_samples", implementation="direct_dft_filter_bank")
        return np.stack(outputs)

    def _output_context(self, packet):
        return (packet.sample_rate / self._parameters["decimation"], packet.center_frequency,
                packet.timestamp + self._business["first_input_offset"] / packet.sample_rate)

    # 对整个矩阵的 center_frequency 保留宽带参考；每行的真实中心在 channels 元数据。


PFBChannelizerComponent = ChannelizerComponent
