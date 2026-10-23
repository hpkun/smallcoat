"""任务格式到运行端口的显式映射；未知语义不会按名称猜测。"""

from dataclasses import dataclass

from ..runtime import RuntimePort


@dataclass(frozen=True)
class DataContract:
    domain: str
    dtype_kinds: tuple[str, ...]
    dimensions: tuple[int, ...]
    item_unit: str


FORMATS = {
    "iq/complex": DataContract("time", ("c",), (1,), "samples"),
    "iq/baseband": DataContract("time", ("c",), (1,), "samples"),
    "iq/wideband": DataContract("time", ("c",), (1,), "samples"),
    "iq/ofdm_symbols": DataContract("symbols", ("c",), (1,), "symbols"),
    "ew/channelized_iq": DataContract("time", ("c",), (2,), "samples"),
    "cni/symbols": DataContract("symbols", ("c",), (1,), "symbols"),
    "data/soft_bits": DataContract("llr", ("f",), (1,), "bits"),
    "data/decoded_bits": DataContract("bits", ("u",), (1,), "bits"),
    "data/bits": DataContract("bits", ("u",), (1,), "bits"),
    "radar/range_profile": DataContract("range", ("c",), (1,), "cells"),
    "radar/power_map": DataContract("power", ("f", "i", "u"), (1, 2), "cells"),
    "radar/detections": DataContract("detections", ("b",), (1, 2), "cells"),
    "spectrum/psd": DataContract("psd", ("f",), (1,), "cells"),
    "frequency/complex": DataContract("frequency", ("c",), (1,), "samples"),
    "time/complex": DataContract("time", ("c",), (1,), "samples"),
}


def check_port(format_name: str, port: RuntimePort) -> str | None:
    contract = FORMATS.get(format_name)
    if contract is None:
        return f"no verified runtime contract for task format {format_name}"
    if (contract.domain not in port.domains or
            not set(contract.dtype_kinds) <= set(port.dtype_kinds) or
            not set(contract.dimensions) <= set(port.dimensions) or
            contract.item_unit != port.item_unit):
        return (f"task format {format_name} requires {contract}, "
                f"but component port {port.name} provides/accepts {port}")
    return None


def check_node_ports(node, component) -> str | None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return "current runtime components require one input and one output port"
    input_format = next(iter(node.inputs.values()))
    output_format = next(iter(node.outputs.values()))
    return (check_port(input_format, component.input_port) or
            check_port(output_format, component.output_port))
