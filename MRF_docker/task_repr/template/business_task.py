"""从多任务的抽象功能图提取公共 DAG，并保存可恢复的任务差异。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from ..errors import ValidationError


@dataclass(frozen=True)
class BusinessTaskVariant:
    task_id: str
    semantics: dict
    qos: dict
    resources: dict
    state: dict
    graph_metadata: dict
    node_order: tuple[str, ...]
    common_node_bindings: dict[str, dict]
    branch_nodes: tuple[dict, ...]
    branch_edges: tuple[dict, ...]
    edge_order: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BusinessTaskTemplate:
    template_id: str
    schema_version: str
    abstraction_level: str
    common_nodes: tuple[dict, ...]
    common_edges: tuple[dict, ...]
    variants: dict[str, BusinessTaskVariant]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _canonical_role(node: Mapping) -> str:
    return node.get("canonical_role", node["role"])


def build_business_template(tasks: Sequence[Mapping]) -> BusinessTaskTemplate:
    """按规范化功能角色提取同一任务族的严格公共图。"""
    if not tasks:
        raise ValidationError("template requires at least one task")
    first = tasks[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("S"), Mapping):
        raise ValidationError("business template requires normalized task objects")
    family = first.get("S", {}).get("family")
    if not isinstance(family, str) or not family:
        raise ValidationError("business template requires a family")

    role_maps = []
    edge_sets = []
    ids = set()
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("S"), Mapping):
            raise ValidationError("business template requires normalized task objects")
        if task.get("schema_version") != "rf_task_representation.v1":
            raise ValidationError("business template requires rf_task_representation.v1")
        if task["S"].get("family") != family:
            raise ValidationError("business template tasks must have one family")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValidationError("business template task ids must be unique nonempty strings")
        ids.add(task_id)
        graph = task.get("G", {})
        if not isinstance(graph, Mapping):
            raise ValidationError(f"task {task_id} requires a graph")
        if (graph.get("abstraction_level") != "business_function" or
                graph.get("generation_status") != "abstract_only_requires_refinement" or
                graph.get("component_bindings") != []):
            raise ValidationError("business template requires an unbound abstract graph")
        nodes = graph.get("nodes", [])
        if (not isinstance(nodes, list) or not nodes or
                any(not isinstance(node, Mapping) or
                    not isinstance(node.get("id"), str) or not node["id"] or
                    not isinstance(node.get("role"), str) or not node["role"] or
                    not isinstance(node.get("canonical_role", node["role"]), str) or
                    not node.get("canonical_role", node["role"])
                    for node in nodes)):
            raise ValidationError(f"task {task_id} requires valid node ids and roles")
        roles = {_canonical_role(node): node for node in nodes}
        by_id = {node["id"]: node for node in nodes}
        if len(roles) != len(nodes) or len(by_id) != len(nodes):
            raise ValidationError(f"task {task_id} requires unique node ids and canonical roles")
        edges = graph.get("edges", [])
        if any(edge["source"] not in by_id or edge["target"] not in by_id for edge in edges):
            raise ValidationError(f"task {task_id} has an edge with an unknown node")
        role_maps.append(roles)
        edge_sets.append({(_canonical_role(by_id[edge["source"]]),
                           _canonical_role(by_id[edge["target"]]))
                          for edge in edges})

    common_roles = set.intersection(*(set(roles) for roles in role_maps))
    for role in common_roles:
        if len({role_map[role]["binding_status"] for role_map in role_maps}) != 1:
            raise ValidationError(f"role {role} has conflicting binding status")
    common_nodes = tuple({"canonical_role": role,
                          "binding_status": role_maps[0][role]["binding_status"]}
                         for role in sorted(common_roles))
    shared_edges = sorted(edge for edge in set.intersection(*edge_sets)
                          if edge[0] in common_roles and edge[1] in common_roles)
    common_edges = tuple({"source_role": source, "target_role": target}
                         for source, target in shared_edges)
    edge_indices = {edge: index for index, edge in enumerate(shared_edges)}

    variants = {}
    for task in tasks:
        graph = task["G"]
        by_id = {node["id"]: node for node in graph["nodes"]}
        branch_edges = []
        edge_order = []
        for edge in graph["edges"]:
            role_edge = (_canonical_role(by_id[edge["source"]]),
                         _canonical_role(by_id[edge["target"]]))
            if role_edge in edge_indices:
                edge_order.append(("common", edge_indices[role_edge]))
            else:
                edge_order.append(("branch", len(branch_edges)))
                branch_edges.append(deepcopy(edge))
        variants[task["task_id"]] = BusinessTaskVariant(
            task["task_id"], deepcopy(task["S"]), deepcopy(task["Q"]),
            deepcopy(task["R"]), deepcopy(task["X"]),
            {key: deepcopy(value) for key, value in graph.items() if key not in {"nodes", "edges"}},
            tuple(_canonical_role(node) for node in graph["nodes"]),
            {_canonical_role(node): deepcopy(node) for node in graph["nodes"]
             if _canonical_role(node) in common_roles},
            tuple(deepcopy(node) for node in graph["nodes"]
                  if _canonical_role(node) not in common_roles),
            tuple(branch_edges), tuple(edge_order),
        )
    ordered_tasks = sorted(tasks, key=lambda task: task["task_id"])
    fingerprint = sha256(json.dumps(ordered_tasks, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return BusinessTaskTemplate(f"external_{family.lower()}_{fingerprint}_v1",
                                "rf_task_representation.v1",
                                "business_function", common_nodes, common_edges, variants)


def restore_business_task(template: BusinessTaskTemplate, task_id: str) -> dict:
    """用公共图和任务差异恢复原始抽象表征。"""
    if task_id not in template.variants:
        raise ValidationError(f"task {task_id} is not in the template")
    variant = template.variants[task_id]
    by_role = {_canonical_role(node): deepcopy(node) for node in variant.branch_nodes}
    for node in template.common_nodes:
        role = node["canonical_role"]
        by_role[role] = deepcopy(variant.common_node_bindings[role])
    common_edges = [{"source": variant.common_node_bindings[edge["source_role"]]["id"],
                     "target": variant.common_node_bindings[edge["target_role"]]["id"]}
                    for edge in template.common_edges]
    edges = [deepcopy((common_edges if kind == "common" else variant.branch_edges)[index])
             for kind, index in variant.edge_order]
    graph = deepcopy(variant.graph_metadata)
    graph.update(nodes=[by_role[role] for role in variant.node_order], edges=edges)
    return {"schema_version": template.schema_version, "task_id": task_id,
            "S": deepcopy(variant.semantics), "Q": deepcopy(variant.qos),
            "R": deepcopy(variant.resources), "G": graph, "X": deepcopy(variant.state)}
