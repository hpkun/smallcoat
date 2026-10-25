"""Resolve task format names through OPS and compare them with OCS and runtime ports."""

from ..specs import ComponentSpec, ProtocolSpec, load_protocol_specs
from ..runtime import RuntimePort


def check_port(format_name: str, port: RuntimePort,
               protocols: dict[str, ProtocolSpec] | None = None) -> str | None:
    protocols = protocols if protocols is not None else load_protocol_specs()
    contract = protocols.get(format_name)
    if contract is None:
        return f"no verified OPS for task format {format_name}"
    return contract.check_port(port)


def check_node_ports(node, component, spec: ComponentSpec | None = None,
                     protocols: dict[str, ProtocolSpec] | None = None) -> str | None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return "current runtime components require one input and one output port"
    input_format = next(iter(node.inputs.values()))
    output_format = next(iter(node.outputs.values()))
    for name, format_name, runtime_port in (
        ("in", input_format, component.input_port),
        ("out", output_format, component.output_port),
    ):
        if spec is not None:
            error = spec.check_port(name, format_name, runtime_port)
            if error:
                return error
        error = check_port(format_name, runtime_port, protocols)
        if error:
            return error
    return None
