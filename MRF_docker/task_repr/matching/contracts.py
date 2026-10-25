"""Resolve task format names through OPS and compare them with OCS and runtime ports."""

import math

from ..specs import ComponentSpec, ProtocolSpec, load_protocol_specs
from ..runtime import DataPacket, RuntimePort


def check_port(format_name: str, port: RuntimePort,
               protocols: dict[str, ProtocolSpec] | None = None) -> str | None:
    protocols = protocols if protocols is not None else load_protocol_specs()
    contract = protocols.get(format_name)
    if contract is None:
        return f"no verified OPS for task format {format_name}"
    return contract.check_port(port)


def check_node_ports(node, component, spec: ComponentSpec | None = None,
                     protocols: dict[str, ProtocolSpec] | None = None,
                     configuration: dict | None = None) -> str | None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return "current runtime components require one input and one output port"
    input_format = next(iter(node.inputs.values()))
    output_format = next(iter(node.outputs.values()))
    for name, format_name, runtime_port in (
        ("in", input_format, component.input_port),
        ("out", output_format, component.output_port),
    ):
        if spec is not None:
            error = spec.check_port(name, format_name, runtime_port, configuration)
            if error:
                return error
        error = check_port(format_name, runtime_port, protocols)
        if error:
            return error
    return None


def validate_connection(task, edge, source, target, left, right,
                        left_binding, right_binding) -> str | None:
    """Check a selected pair using the same edge rules at match and launch time."""
    if (edge.source_port not in source.outputs or edge.target_port not in target.inputs or
            source.outputs[edge.source_port] != target.inputs[edge.target_port]):
        return "task edge formats disagree"
    if not left.output_port.can_connect_to(right.input_port):
        return "runtime port domain, dtype, dimensions or item_unit differ"
    left_length = source.parameters.get("output_block_length")
    right_length = target.parameters.get("input_block_length")
    if left_length is not None and right_length is not None and left_length != right_length:
        return f"block length {left_length} does not equal {right_length}"
    left_rate = source.parameters.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    right_rate = target.parameters.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    if left_rate is not None and right_rate is not None:
        if left_binding.cid in ("C03", "C09"):
            left_rate /= left_binding.parameters["decimation"]
        if left_binding.cid == "C03":
            left_rate *= left_binding.parameters.get("interpolation", 1)
        if not math.isclose(left_rate, right_rate, rel_tol=1e-9):
            return f"sample rate {left_rate} MHz does not equal {right_rate} MHz"
    return None


def validate_runtime_packet(format_name: str, packet: DataPacket,
                            port: RuntimePort | None = None,
                            protocols: dict[str, ProtocolSpec] | None = None) -> str | None:
    protocols = protocols if protocols is not None else load_protocol_specs()
    contract = protocols.get(format_name)
    if contract is None:
        return f"no verified OPS for task format {format_name}"
    error = contract.validate_packet(packet)
    if error:
        return error
    if port is not None:
        error = contract.check_port(port)
        if error:
            return error
        try:
            port.validate(packet)
        except (TypeError, ValueError) as exc:
            return str(exc)
    return None


def validate_signal_packet(task, node, packet: DataPacket, direction: str,
                           binding=None) -> str | None:
    """Check declared block size and input/output sampling rate in samples/s."""
    length = node.parameters.get(f"{direction}_block_length")
    if length is not None and packet.item_count != length:
        return f"{direction} block length must be {length}, got {packet.item_count}"
    rate_mhz = node.parameters.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    if rate_mhz is None:
        return None
    expected = rate_mhz * 1_000_000
    if direction == "output" and binding is not None:
        if binding.cid in ("C03", "C09"):
            expected /= binding.parameters["decimation"]
        if binding.cid == "C03":
            expected *= binding.parameters.get("interpolation", 1)
    if not math.isclose(packet.sample_rate, expected, rel_tol=1e-9):
        return f"{direction} sample rate must be {expected} samples/s, got {packet.sample_rate}"
    return None
