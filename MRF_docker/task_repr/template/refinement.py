"""按显式业务规则把抽象功能图细化为静态匹配所需的规格图。"""

from collections.abc import Mapping
from dataclasses import dataclass
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

    def to_dict(self) -> dict:
        return {"nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges],
                "boundaries": list(self.boundaries)}


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
    profile = rule.get("refinement")
    if profile is None or profile["direction"] != semantics["direction"]:
        questions = graph.get("unresolved_information", [])
        detail = f"no refinement profile for {task_type}/{semantics['direction']}"
        if questions:
            detail += f"; unresolved: {', '.join(questions)}"
        return RefinementResult(task_id, None, (RefinementIssue("NEEDS_REFINEMENT", detail),))
    selection = profile["selection"]
    selected_value = _source_value(parameters, selection["path"])
    if selected_value != selection["equals"]:
        detail = (f"S.business_parameters.{selection['path']} must select "
                  f"{selection['equals']}; current value: "
                  f"{'missing' if selected_value is _MISSING else selected_value}")
        return RefinementResult(task_id, None, (RefinementIssue("NEEDS_REFINEMENT", detail),))

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
            if format_name not in known_formats:
                raise ValidationError(f"unknown boundary format: {format_name}")
            endpoints[original_id] = (definition["boundary"], format_name)
            boundaries.append({"business_node_id": original_id, "role": role,
                               "direction": definition["boundary"], "format": format_name,
                               "binding_status": semantics.get("data_input", {}).get("binding_status")
                               if definition["boundary"] == "source" else "external_output"})
            if definition["boundary"] == "source":
                input_format = semantics.get("data_input", {}).get("format")
                if input_format is None:
                    issues.append(RefinementIssue("NEEDS_PARAMETERS", "S.data_input.format is required", original_id))
                elif input_format != format_name:
                    issues.append(RefinementIssue("INTERFACE_MISMATCH", "input format differs from refinement profile", original_id))
            elif definition.get("expected_output") not in semantics.get("expected_output", []):
                issues.append(RefinementIssue("NEEDS_PARAMETERS", f"S.expected_output must include {definition['expected_output']}", original_id))
            continue
        steps = definition.get("steps", [])
        if not steps:
            raise ValidationError(f"role {role} has no refinement steps")
        previous = None
        for step in steps:
            input_format = step["input_format"]
            output_format = step["output_format"]
            if input_format not in known_formats or output_format not in known_formats:
                raise ValidationError(f"role {role} uses an unknown OPS format")
            node_id = f"{original_id}:{step['id']}"
            values = {}
            for name, source in step["parameters"].items():
                value = _source_value(parameters, source)
                if value is _MISSING or value is None:
                    issues.append(RefinementIssue("NEEDS_PARAMETERS", f"S.business_parameters.{source} is required", original_id))
                elif type(value) not in (str, int, float, bool):
                    issues.append(RefinementIssue("INVALID_PARAMETERS", f"S.business_parameters.{source} must be scalar", original_id))
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

    if issues:
        return RefinementResult(task_id, None, tuple(issues))
    if not nodes:
        return RefinementResult(task_id, None, (RefinementIssue("NEEDS_REFINEMENT", "profile has no component specifications"),))
    return RefinementResult(task_id, SpecificationTask(
        task_id, SpecificationSemantics(semantics["family"], task_type,
                                        semantics["direction"], {}),
        task["Q"], task["R"], SpecificationGraph(tuple(nodes), tuple(edges),
                                                 tuple(boundaries)), task["X"]), ())
