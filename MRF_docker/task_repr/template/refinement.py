"""按显式业务规则把抽象功能图细化为静态匹配所需的规格图。"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

from components.specs import load_protocol_specs

from ..errors import ValidationError
from ..parsing.external_task import RULES


@dataclass(frozen=True)
class RefinementIssue:
    code: str
    detail: str
    node_id: str | None = None


@dataclass(frozen=True)
class SpecificationNode:
    node_id: str
    role: str
    capability: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    parameters: dict[str, str | int | float | bool]
    business_node_id: str

    def to_dict(self) -> dict:
        return {"id": self.node_id, "business_node_id": self.business_node_id,
                "role": self.role, "capability": self.capability,
                "inputs": self.inputs, "outputs": self.outputs,
                "parameters": self.parameters}


@dataclass(frozen=True)
class SpecificationEdge:
    source: str
    source_port: str
    target: str
    target_port: str

    def to_dict(self) -> dict:
        return {"source": self.source, "source_port": self.source_port,
                "target": self.target, "target_port": self.target_port}


@dataclass(frozen=True)
class SpecificationGraph:
    nodes: tuple[SpecificationNode, ...]
    edges: tuple[SpecificationEdge, ...]
    boundaries: tuple[dict, ...]
    profile_id: str
    profile_version: str
    completion_scope: str
    proposed_contracts: dict[str, str]
    prerequisites: tuple[str, ...]
    covered_outputs: tuple[str, ...]
    uncovered_outputs: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges],
                "boundaries": list(self.boundaries), "profile_id": self.profile_id,
                "profile_version": self.profile_version,
                "completion_scope": self.completion_scope,
                "proposed_contracts": self.proposed_contracts,
                "prerequisites": list(self.prerequisites),
                "covered_outputs": list(self.covered_outputs),
                "uncovered_outputs": list(self.uncovered_outputs)}


@dataclass(frozen=True)
class SpecificationSemantics:
    family: str
    task_type: str
    direction: str
    extension: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class SpecificationTask:
    task_id: str
    S: SpecificationSemantics
    Q: dict
    R: dict
    G: SpecificationGraph
    X: dict


@dataclass(frozen=True)
class RefinementResult:
    task_id: str
    task: SpecificationTask | None
    issues: tuple[RefinementIssue, ...]


_MISSING = object()


def _source_value(parameters: Mapping, path: str) -> Any:
    value: Any = parameters
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _parameter_value(parameters: Mapping, source: str | Mapping) -> Any:
    if isinstance(source, str):
        value = _source_value(parameters, source)
        return value if type(value) in (str, int, float, bool) else _MISSING
    if not isinstance(source, Mapping) or set(source) != {"path", "encoding"} or source["encoding"] != "json":
        raise ValidationError("refinement parameter source must be a path or JSON encoding rule")
    value = _source_value(parameters, source["path"])
    if value is _MISSING or not isinstance(value, (list, dict)):
        return _MISSING
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return _MISSING


def _missing_required(parameters: Mapping, path: str) -> bool:
    value = _source_value(parameters, path)
    return (value is _MISSING or value is None or value == "" or
            (path != "processing.pilot_pattern" and isinstance(value, (list, dict)) and not value))


def refine_business_task(task: Mapping) -> RefinementResult:
    """细化一条已解析业务任务；信息不足时返回问题而不产生组件绑定。"""
    if not isinstance(task, Mapping) or task.get("schema_version") != "rf_task_representation.v1":
        raise ValidationError("refinement requires rf_task_representation.v1")
    task_id = task.get("task_id")
    semantics = task.get("S")
    graph = task.get("G")
    if (not isinstance(task_id, str) or not task_id or
            not isinstance(semantics, Mapping) or not isinstance(graph, Mapping) or
            graph.get("abstraction_level") != "business_function"):
        raise ValidationError("refinement requires a valid business task and graph")
    task_type = semantics.get("task_type")
    rule = RULES["task_types"].get(task_type)
    if rule is None or semantics.get("direction") not in rule["valid_directions"]:
        raise ValidationError("unknown business task type or direction")
    if semantics.get("family") != rule["canonical_family"]:
        raise ValidationError("business task family does not match its type")
    parameters = semantics.get("business_parameters")
    if not isinstance(parameters, Mapping):
        raise ValidationError("S.business_parameters must be an object")
    profiles = rule.get("refinements") or ([rule["refinement"]] if "refinement" in rule else [])
    profile = next((item for item in profiles
                    if semantics["direction"] in item.get("directions", [item.get("direction")]) and
                    _source_value(parameters, item["selection"]["path"]) == item["selection"]["equals"]), None)
    if profile is None:
        questions = graph.get("unresolved_information", [])
        available = [item["selection"]["equals"] for item in profiles
                     if semantics["direction"] in item.get("directions", [item.get("direction")])]
        detail = (f"no selected refinement profile for {task_type}/{semantics['direction']}; "
                  f"S.business_parameters.processing.profile must be one of {available}")
        if questions:
            detail += f"; unresolved: {', '.join(questions)}"
        return RefinementResult(task_id, None, (RefinementIssue("NEEDS_REFINEMENT", detail),))
    if (profile.get("input_kind") is not None and
            semantics.get("data_input", {}).get("kind") != profile["input_kind"]):
        return RefinementResult(task_id, None, (RefinementIssue(
            "NEEDS_REFINEMENT", f"S.data_input.kind must be {profile['input_kind']}"),))
    if (profile.get("input_format") is not None and
            semantics.get("data_input", {}).get("format") != profile["input_format"]):
        return RefinementResult(task_id, None, (RefinementIssue(
            "NEEDS_REFINEMENT", f"S.data_input.format must be {profile['input_format']}"),))
    condition_issues = [RefinementIssue(
        "NEEDS_REFINEMENT", f"S.business_parameters.{path} must be {expected!r}")
        for path, expected in profile.get("conditions", {}).items()
        if _source_value(parameters, path) != expected]
    if condition_issues:
        return RefinementResult(task_id, None, tuple(condition_issues))
    required_issues = [RefinementIssue("NEEDS_PARAMETERS", f"S.business_parameters.{path} is required")
                       for path in profile.get("required_parameters", [])
                       if _missing_required(parameters, path)]
    if required_issues:
        return RefinementResult(task_id, None, tuple(required_issues))
    if profile.get("frame_contract") == "one_ofdm_symbol_qpsk_zero_tail_convolutional":
        allocation = _source_value(parameters, "processing.subcarrier_allocation")
        generators = _source_value(parameters, "processing.generators")
        block_length = _source_value(parameters, "processing.block_length")
        payload_bits = _source_value(parameters, "processing.frame_payload_bits")
        header_bits = _source_value(parameters, "processing.frame_header_bits")
        crc_bits = _source_value(parameters, "processing.crc_bits")
        constraint_length = _source_value(parameters, "processing.constraint_length")
        fft_length = _source_value(parameters, "processing.fft_length")
        cp_length = _source_value(parameters, "processing.cp_length")
        if (not isinstance(allocation, list) or not allocation or
                not isinstance(generators, list) or len(generators) < 2 or
                any(type(value) is not int for value in allocation + generators) or
                type(block_length) is not int or block_length < 1 or
                type(payload_bits) is not int or payload_bits < 1 or
                type(header_bits) is not int or header_bits < 0 or
                type(crc_bits) is not int or crc_bits < 1 or
                type(constraint_length) is not int or constraint_length < 2 or
                type(fft_length) is not int or fft_length < 2 or
                type(cp_length) is not int or cp_length < 0):
            return RefinementResult(task_id, None, (RefinementIssue(
                "INVALID_PARAMETERS", "OFDM/FEC frame dimensions must be positive integer arrays and lengths"),))
        if payload_bits + header_bits + crc_bits != block_length:
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "frame payload, header and CRC bits must total the FEC input block length"),))
        if _source_value(parameters, "processing.crc") == "crc16" and crc_bits != 16:
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "crc16 requires a 16-bit CRC field"),))
        if task_type == "data_link_transfer":
            fragment_bytes = _source_value(parameters, "processing.fragment_payload_bytes")
            if type(fragment_bytes) is not int or fragment_bytes < 1 or fragment_bytes * 8 != payload_bits:
                return RefinementResult(task_id, None, (RefinementIssue(
                    "INTERFACE_MISMATCH", "fragment payload bytes must equal frame payload bits"),))
        if task_type == "voice_communication":
            mode = _source_value(parameters, "processing.codec2_mode")
            frame_ms = _source_value(parameters, "processing.codec2_frame_ms")
            if (not isinstance(mode, str) or not mode.isdecimal() or
                    type(frame_ms) is not int or frame_ms < 1 or
                    int(mode) * frame_ms != payload_bits * 1000):
                return RefinementResult(task_id, None, (RefinementIssue(
                    "INTERFACE_MISMATCH", "Codec 2 mode and frame duration must produce the declared frame payload bits"),))
            traffic_bytes = _source_value(parameters, "traffic.encoded_payload_bytes_per_frame")
            if (traffic_bytes is not _MISSING and
                    (type(traffic_bytes) is not int or traffic_bytes * 8 != payload_bits)):
                return RefinementResult(task_id, None, (RefinementIssue(
                    "INTERFACE_MISMATCH", "traffic encoded payload bytes conflict with the selected Codec 2 frame"),))
        coded_bits = (block_length + constraint_length - 1) * len(generators)
        if 2 * len(allocation) != coded_bits:
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", f"one QPSK OFDM symbol carries {2 * len(allocation)} bits, "
                f"but the terminated convolutional block produces {coded_bits} bits"),))
        input_length = _source_value(parameters, "processing.input_block_length")
        if input_length is not _MISSING and semantics["direction"] == "rx" and (
                type(input_length) is not int or input_length != fft_length + cp_length):
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "OFDM receive input block must equal fft_length + cp_length"),))

    if task_type == "target_search":
        sample_rate = _source_value(parameters, "processing.baseband_sample_rate_mhz")
        input_rate = _source_value(parameters, "processing.input_sample_rate_mhz")
        decimation = _source_value(parameters, "processing.decimation")
        bandwidth = _source_value(parameters, "processing.pulse_bandwidth_mhz")
        pulse_width = _source_value(parameters, "processing.pulse_width_us")
        carrier = _source_value(parameters, "processing.carrier_frequency_mhz")
        pri = _source_value(parameters, "processing.pri_us")
        cpi_pulses = _source_value(parameters, "processing.cpi_pulses")
        doppler_fft_length = _source_value(parameters, "processing.doppler_fft_length")
        guard_cells = _source_value(parameters, "processing.guard_cells")
        reference_cells = _source_value(parameters, "processing.reference_cells")
        pfa = _source_value(parameters, "processing.pfa")
        input_length = _source_value(parameters, "processing.input_block_length")
        baseband_length = _source_value(parameters, "processing.baseband_block_length")
        reference = _source_value(parameters, "processing.reference_waveform")
        resolution = _source_value(parameters, "range_resolution_requirement_m")
        ranges = _source_value(parameters, "search_volume.range_m")
        velocities = _source_value(parameters, "radial_velocity_span_mps")
        rates = (sample_rate, input_rate, bandwidth, pulse_width, carrier, pri, resolution)
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in rates) or
                type(decimation) is not int or decimation < 1 or
                type(cpi_pulses) is not int or cpi_pulses < 1 or
                type(doppler_fft_length) is not int or doppler_fft_length < cpi_pulses or
                type(guard_cells) is not int or guard_cells < 1 or
                type(reference_cells) is not int or reference_cells < 1 or
                type(pfa) not in (int, float) or not math.isfinite(pfa) or not 0 < pfa < 1 or
                type(input_length) is not int or input_length < 1 or
                type(baseband_length) is not int or baseband_length < 1 or
                not isinstance(reference, list) or not reference or
                not isinstance(ranges, list) or len(ranges) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in ranges) or
                not isinstance(velocities, list) or len(velocities) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in velocities) or
                ranges[0] > ranges[1] or velocities[0] > velocities[1]):
            return RefinementResult(task_id, None, (RefinementIssue(
                "INVALID_PARAMETERS", "radar sampling, pulse and search bounds must be finite positive values"),))
        light_speed = 299_792_458
        if (not math.isclose(input_rate / decimation, sample_rate, rel_tol=1e-9) or
                input_length != baseband_length * decimation or
                sample_rate < bandwidth or
                len(reference) != round(sample_rate * pulse_width) or
                pulse_width >= pri):
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "radar sample rates, block lengths or LFM reference duration disagree"),))
        if bandwidth < light_speed / (2 * resolution * 1e6):
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "LFM bandwidth does not meet requested range resolution"),))
        delay_samples = math.ceil(2 * max(ranges) * sample_rate * 1e6 / light_speed)
        if baseband_length < delay_samples + len(reference) or baseband_length > sample_rate * pri:
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "radar echo window does not cover requested range within one PRI"),))
        unambiguous_velocity = light_speed / (4 * carrier * 1e6 * pri * 1e-6)
        if max(abs(value) for value in velocities) > unambiguous_velocity:
            return RefinementResult(task_id, None, (RefinementIssue(
                "INTERFACE_MISMATCH", "carrier and PRI do not cover requested unambiguous velocity span"),))

    business_nodes = graph.get("nodes")
    business_edges = graph.get("edges")
    if not isinstance(business_nodes, list) or not isinstance(business_edges, list):
        raise ValidationError("business graph nodes and edges must be arrays")
    by_id = {node["id"]: node for node in business_nodes}
    if len(by_id) != len(business_nodes):
        raise ValidationError("business graph node ids must be unique")
    branches = rule["branches"]
    selected = (["main"] if "main" in branches else
                ["tx", "rx"] if semantics["direction"] == "tx_rx" else
                [semantics["direction"]])
    expected_nodes = {f"{branch}-{index:02d}": role
                      for branch in selected
                      for index, role in enumerate(branches[branch], 1)}
    expected_edges = {(f"{branch}-{index:02d}", f"{branch}-{index + 1:02d}")
                      for branch in selected
                      for index in range(1, len(branches[branch]))}
    if ({node_id: node.get("role") for node_id, node in by_id.items()} != expected_nodes or
            {(edge["source"], edge["target"]) for edge in business_edges} != expected_edges or
            len(business_edges) != len(expected_edges)):
        raise ValidationError("business graph differs from its declared task rules")
    roles = profile["roles"]
    if any(node.get("role") not in roles for node in business_nodes):
        raise ValidationError("refinement profile does not cover every business role")
    known_formats = load_protocol_specs()
    proposed_contracts = profile.get("proposed_contracts", {})
    if set(proposed_contracts) & set(known_formats):
        raise ValidationError("proposed contracts must not shadow verified OPS")
    def check_format(format_name: str) -> None:
        if format_name not in known_formats and format_name not in proposed_contracts:
            raise ValidationError(f"unknown refinement format: {format_name}")
    nodes: list[SpecificationNode] = []
    edges: list[SpecificationEdge] = []
    boundaries: list[dict] = []
    endpoints = {}
    issues: list[RefinementIssue] = []
    for business_node in business_nodes:
        original_id = business_node["id"]
        role = business_node["role"]
        definition = roles[role]
        if "boundary" in definition:
            format_name = definition["format"]
            check_format(format_name)
            endpoints[original_id] = (definition["boundary"], format_name)
            boundaries.append({"business_node_id": original_id, "role": role,
                               "direction": definition["boundary"], "format": format_name,
                               "binding_status": semantics.get("data_input", {}).get("binding_status")
                               if definition["boundary"] == "source" else "external_output"})
            if definition["boundary"] == "source":
                source_path = definition.get("source_format_path", "data_input.format")
                input_format = _source_value(semantics, source_path)
                if input_format is _MISSING or input_format is None:
                    issues.append(RefinementIssue("NEEDS_PARAMETERS", f"S.{source_path} is required", original_id))
                elif input_format != format_name:
                    issues.append(RefinementIssue("INTERFACE_MISMATCH", "input format differs from refinement profile", original_id))
            elif (definition.get("expected_output") is not None and
                  definition["expected_output"] not in semantics.get("expected_output", [])):
                issues.append(RefinementIssue("NEEDS_PARAMETERS", f"S.expected_output must include {definition['expected_output']}", original_id))
            continue
        steps = definition.get("steps", [])
        if not steps:
            raise ValidationError(f"role {role} has no refinement steps")
        previous = None
        for step in steps:
            input_format = step["input_format"]
            output_format = step["output_format"]
            check_format(input_format)
            check_format(output_format)
            node_id = f"{original_id}:{step['id']}"
            values = {}
            for name, source in step["parameters"].items():
                value = _parameter_value(parameters, source)
                path = source if isinstance(source, str) else source["path"]
                if value is _MISSING or value is None or value == "":
                    issues.append(RefinementIssue("NEEDS_PARAMETERS", f"S.business_parameters.{path} is required", original_id))
                else:
                    values[name] = value
            node = SpecificationNode(node_id, f"{original_id}:{step['id']}",
                                     step["capability"], {"in": input_format},
                                     {"out": output_format}, values, original_id)
            if previous is not None:
                if previous.outputs["out"] != input_format:
                    issues.append(RefinementIssue("INTERFACE_MISMATCH", f"role {role} has incompatible internal formats", original_id))
                edges.append(SpecificationEdge(previous.node_id, "out", node_id, "in"))
            nodes.append(node)
            previous = node
        endpoints[original_id] = (nodes[-len(steps)], nodes[-1])

    for edge in business_edges:
        source_id, target_id = edge["source"], edge["target"]
        if source_id not in endpoints or target_id not in endpoints:
            raise ValidationError("business edge references an unknown node")
        source_end = endpoints[source_id]
        target_end = endpoints[target_id]
        if source_end[0] == "sink" or target_end[0] == "source":
            raise ValidationError("business boundary has an invalid edge direction")
        source_format = source_end[1] if isinstance(source_end[0], str) else source_end[1].outputs["out"]
        target_format = target_end[1] if isinstance(target_end[0], str) else target_end[0].inputs["in"]
        if source_format != target_format:
            issues.append(RefinementIssue("INTERFACE_MISMATCH", f"business edge {source_id} -> {target_id} has incompatible formats", target_id))
        if not isinstance(source_end[0], str) and not isinstance(target_end[0], str):
            edges.append(SpecificationEdge(source_end[1].node_id, "out", target_end[0].node_id, "in"))

    connected_targets = {edge.target for edge in edges}
    connected_sources = {edge.source for edge in edges}
    source_boundary_targets = {edge["target"] for edge in business_edges
                               if endpoints[edge["source"]][0] == "source"}
    sink_boundary_sources = {edge["source"] for edge in business_edges
                             if endpoints[edge["target"]][0] == "sink"}
    for node in nodes:
        if node.node_id not in connected_targets and node.business_node_id not in source_boundary_targets:
            boundaries.append({"business_node_id": node.business_node_id, "role": node.role,
                               "direction": "source", "format": node.inputs["in"],
                               "binding_status": semantics.get("data_input", {}).get("binding_status")})
        if node.node_id not in connected_sources and node.business_node_id not in sink_boundary_sources:
            boundaries.append({"business_node_id": node.business_node_id, "role": node.role,
                               "direction": "sink", "format": node.outputs["out"],
                               "binding_status": "external_output"})

    if issues:
        return RefinementResult(task_id, None, tuple(issues))
    if not nodes:
        return RefinementResult(task_id, None, (RefinementIssue("NEEDS_REFINEMENT", "profile has no component specifications"),))
    expected_outputs = semantics.get("expected_output", [])
    declared_outputs = profile.get("covered_outputs", {})
    covered_outputs = declared_outputs.get(semantics["direction"], []) if isinstance(declared_outputs, dict) else declared_outputs
    if not isinstance(expected_outputs, list) or any(not isinstance(value, str) for value in expected_outputs):
        raise ValidationError("S.expected_output must be an array of names")
    covered = tuple(value for value in expected_outputs if value in covered_outputs)
    uncovered = tuple(value for value in expected_outputs if value not in covered_outputs)
    return RefinementResult(task_id, SpecificationTask(
        task_id, SpecificationSemantics(semantics["family"], task_type,
                                        semantics["direction"], {}),
        task["Q"], task["R"], SpecificationGraph(
            tuple(nodes), tuple(edges), tuple(boundaries),
            profile.get("profile_id", profile["selection"]["equals"]),
            profile.get("profile_version", "1.0"),
            profile.get("completion_scope", "component_dag"),
            dict(proposed_contracts), tuple(profile.get("prerequisites", [])),
            covered, uncovered), task["X"]), ())
