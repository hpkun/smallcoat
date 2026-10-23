"""静态匹配任务 DAG；创建组件仅为验证配置，不处理数据包。"""

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Mapping

from ..catalog import CapabilityMap, default_capability_map
from ..component_models.runtime_registry import create_runtime_component
from ..models.task import TaskRepresentation, represent_task
from .contracts import check_node_ports
from .rules import MatchRuleError, adapt_parameters


_KNOWN_UNSUPPORTED = {
    "mti_mtd_fft": "complete MTI/MTD requires C22; C22 has no runtime implementation",
    "ldpc_decoding": "C17 implements convolutional FEC only; LDPC has no runtime implementation",
    "ofdm_timing_frequency_sync": "C07 supports known-preamble frame sync, not complete OFDM timing/frequency sync",
    "ofdm_channel_equalization": "no runtime component implements OFDM channel equalization",
}


@dataclass(frozen=True)
class MatchIssue:
    code: str
    detail: str
    node_id: str | None = None
    cid: str | None = None
    edge: tuple[str, str] | None = None

    def to_dict(self) -> dict:
        data = {"code": self.code, "detail": self.detail}
        if self.node_id is not None:
            data["node_id"] = self.node_id
        if self.cid is not None:
            data["cid"] = self.cid
        if self.edge is not None:
            data["edge"] = list(self.edge)
        return data


@dataclass(frozen=True)
class ComponentBinding:
    cid: str
    parameters: dict

    def to_dict(self) -> dict:
        return {"cid": self.cid, "parameters": self.parameters}


@dataclass(frozen=True)
class StaticMatchResult:
    task_id: str
    status: str
    bindings: dict[str, ComponentBinding]
    edges: tuple[tuple[str, str], ...]
    unresolved: tuple[MatchIssue, ...]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "bindings": {key: value.to_dict() for key, value in self.bindings.items()},
                "edges": [list(edge) for edge in self.edges],
                "unresolved": [issue.to_dict() for issue in self.unresolved]}


@dataclass(frozen=True)
class _Candidate:
    binding: ComponentBinding
    component: object


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must be a positive integer")
    return value


def _signal_checks(task, node, candidate):
    p = node.parameters
    config = candidate.binding.parameters
    rate_mhz = p.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    if rate_mhz is not None:
        if type(rate_mhz) not in (int, float) or not math.isfinite(rate_mhz) or rate_mhz <= 0:
            raise MatchRuleError("INVALID_PARAMETERS", "sample_rate_mhz must be positive and finite")
        sample_rate_hz = rate_mhz * 1_000_000
        if candidate.binding.cid == "C03" and abs(config["nco_frequency"]) >= sample_rate_hz / 2:
            raise MatchRuleError("INVALID_PARAMETERS", "nco_frequency exceeds input Nyquist rate")
        if candidate.binding.cid == "C27" and config.get("scan_bandwidth", 0) and config["scan_bandwidth"] > sample_rate_hz:
            raise MatchRuleError("INVALID_PARAMETERS", "scan_bandwidth exceeds sample rate")
    input_length = p.get("input_block_length")
    output_length = p.get("output_block_length")
    if input_length is not None:
        _positive_integer(input_length, "input_block_length")
    if output_length is not None:
        _positive_integer(output_length, "output_block_length")
    cid = candidate.binding.cid
    if cid in ("C08", "C27") and input_length is None:
        raise MatchRuleError("NEEDS_PARAMETERS", "input_block_length is needed to verify the FFT window")
    if cid == "C21" and input_length is None and config.get("pulse_length") is None:
        raise MatchRuleError("NEEDS_PARAMETERS", "input_block_length or pulse_length is needed to verify the echo frame")
    if cid == "C07" and input_length is not None and input_length < len(config["preamble"]):
        raise MatchRuleError("INTERFACE_MISMATCH", "input block is shorter than the preamble")
    if cid in ("C08", "C27"):
        length = config["fft_size"] if cid == "C08" else config["fft_length"]
        if cid == "C08" and input_length is not None and input_length != length:
            raise MatchRuleError("INTERFACE_MISMATCH", "FFT input block must equal fft_size")
        if cid == "C27" and input_length is not None and input_length < length:
            raise MatchRuleError("INTERFACE_MISMATCH", "PSD input block is shorter than fft_length")
        if cid == "C27" and output_length is not None and output_length != length:
            raise MatchRuleError("INTERFACE_MISMATCH", "PSD output block must equal fft_length")
    if cid == "C03" and input_length is not None and output_length is not None:
        expected = (input_length + config["decimation"] - 1) // config["decimation"]
        if output_length != expected:
            raise MatchRuleError("INTERFACE_MISMATCH", f"DDC output length must be {expected}")
    if cid == "C05" and input_length is not None and output_length is not None and input_length != output_length:
        raise MatchRuleError("INTERFACE_MISMATCH", "digital filter preserves block length")
    if cid == "C21" and rate_mhz is not None:
        bandwidth_mhz = p.get("pulse_bandwidth_mhz")
        if bandwidth_mhz is not None and bandwidth_mhz > rate_mhz:
            raise MatchRuleError("INVALID_PARAMETERS", "pulse bandwidth exceeds sample rate")


def _edge_issue(task, edge, left, right, nodes):
    pair = (edge.source, edge.target)
    if left is None or right is None:
        return MatchIssue("BLOCKED_BY_NODE", "one or both endpoint nodes are unresolved", edge=pair)
    source = nodes[edge.source]
    target = nodes[edge.target]
    if source.outputs[edge.source_port] != target.inputs[edge.target_port]:
        return MatchIssue("INTERFACE_MISMATCH", "task edge formats disagree", edge=pair)
    if not left.component.output_port.can_connect_to(right.component.input_port):
        return MatchIssue("INTERFACE_MISMATCH", "runtime port domain, dtype, dimensions or item_unit differ", edge=pair)
    left_length = source.parameters.get("output_block_length")
    right_length = target.parameters.get("input_block_length")
    if left_length is not None and right_length is not None and left_length != right_length:
        return MatchIssue("INTERFACE_MISMATCH", f"block length {left_length} does not equal {right_length}", edge=pair)
    left_rate = source.parameters.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    right_rate = target.parameters.get("sample_rate_mhz", task.S.extension.get("sample_rate_mhz"))
    if left_rate is not None and right_rate is not None:
        if left.binding.cid == "C03":
            left_rate /= left.binding.parameters["decimation"]
        elif left.binding.cid == "C09":
            left_rate /= left.binding.parameters["decimation"]
        if not math.isclose(left_rate, right_rate, rel_tol=1e-9):
            return MatchIssue("INTERFACE_MISMATCH", f"sample rate {left_rate} MHz does not equal {right_rate} MHz", edge=pair)
    return None


def _load_task(task):
    if isinstance(task, TaskRepresentation):
        return task
    if isinstance(task, (str, Path)):
        return represent_task(json.loads(Path(task).read_text(encoding="utf-8")))
    if isinstance(task, Mapping):
        return represent_task(task)
    raise TypeError("task must be a TaskRepresentation, JSON path or task mapping")


def match_task(task, capability_map: CapabilityMap | None = None) -> StaticMatchResult:
    """返回完整或部分静态方案，保留每个无法匹配的节点和边的原因。"""
    task = _load_task(task)
    capability_map = capability_map or default_capability_map()
    nodes = {node.node_id: node for node in task.G.nodes}
    options: dict[str, list[_Candidate]] = {}
    node_issues = []
    for node in task.G.nodes:
        candidates = capability_map.candidates_for(node.capability)
        feasible = []
        failures = []
        if not candidates:
            detail = _KNOWN_UNSUPPORTED.get(node.capability,
                                            f"no runnable capability mapping for {node.capability}")
            failures.append(MatchIssue("UNSUPPORTED", detail, node.node_id))
        for mapping in candidates:
            try:
                parameters = adapt_parameters(task, node, mapping.cid)
                component = create_runtime_component(mapping.cid, **parameters)
                candidate = _Candidate(ComponentBinding(mapping.cid, parameters), component)
                port_error = check_node_ports(node, component)
                if port_error:
                    raise MatchRuleError("INTERFACE_MISMATCH", port_error)
                _signal_checks(task, node, candidate)
                feasible.append(candidate)
            except MatchRuleError as exc:
                failures.append(MatchIssue(exc.code, str(exc), node.node_id, mapping.cid))
            except (TypeError, ValueError) as exc:
                failures.append(MatchIssue("INVALID_PARAMETERS", str(exc), node.node_id, mapping.cid))
        options[node.node_id] = feasible
        if not feasible:
            node_issues.extend(failures)

    # Try every feasible CID assignment until all edges connect. This matters when
    # one capability has more than one implementation.
    active = [node.node_id for node in task.G.nodes if options[node.node_id]]
    chosen = {}
    best = None
    best_issues = None
    for combination in itertools.product(*(options[node_id] for node_id in active)):
        trial = dict(zip(active, combination))
        edge_issues = [_edge_issue(task, edge, trial.get(edge.source), trial.get(edge.target), nodes)
                       for edge in task.G.edges]
        edge_issues = [issue for issue in edge_issues if issue is not None]
        if best is None or len(edge_issues) < len(best_issues):
            best, best_issues = trial, edge_issues
        if not edge_issues:
            break
    chosen = best or {}
    if best_issues is None:
        best_issues = [_edge_issue(task, edge, None, None, nodes) for edge in task.G.edges]
    issues = tuple(node_issues + best_issues)
    bindings = {node_id: candidate.binding for node_id, candidate in chosen.items()}
    # Only expose connections whose two chosen components pass the edge contract.
    edges = tuple((edge.source, edge.target) for edge in task.G.edges
                  if _edge_issue(task, edge, chosen.get(edge.source), chosen.get(edge.target), nodes) is None)
    any_candidate = any(capability_map.candidates_for(node.capability) for node in task.G.nodes)
    status = ("MATCHED" if not issues and len(bindings) == len(nodes) else
              "PARTIAL" if any_candidate else "UNSUPPORTED")
    return StaticMatchResult(task.task_id, status, bindings, edges, issues)
