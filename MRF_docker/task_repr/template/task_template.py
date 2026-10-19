"""从已规范化的任务角色提取公共 DAG 骨架及可恢复的任务变体。"""

from dataclasses import asdict, dataclass
import json
from typing import Sequence

from ..models.resource import ExecutionProfile, Resources, SimulationProfile
from ..models.task import (Dependency, DynamicState, FunctionGraph, FunctionNode,
                           QoS, Semantics, TaskRepresentation)
from ..errors import ValidationError


@dataclass(frozen=True)
class CommonNode:
    """跨任务一致的功能语义和端口契约。"""

    role: str
    capability: str
    inputs: dict[str, str]
    outputs: dict[str, str]


@dataclass(frozen=True)
class CommonEdge:
    """使用规范化角色而非任务局部节点 ID 表示公共依赖。"""

    source_role: str
    source_port: str
    target_role: str
    target_port: str


@dataclass(frozen=True)
class NodeVariant:
    """公共功能在单个任务中的参数和异构执行代价。"""

    node_id: str
    estimated_runtime_ms: float
    parameters: dict[str, str | int | float | bool]
    shareable: bool
    preemptible: bool
    execution_profiles: dict[str, ExecutionProfile]
    simulation_profile: SimulationProfile | None


@dataclass(frozen=True)
class TaskVariant:
    """任务专有字段和公共骨架以外的分支图。"""

    task_id: str
    semantics: Semantics
    qos: QoS
    resources: Resources
    state: DynamicState
    node_order: tuple[str, ...]
    common_nodes: dict[str, NodeVariant]
    branch_nodes: tuple[FunctionNode, ...]
    branch_edges: tuple[Dependency, ...]
    edge_order: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TaskTemplate:
    """公共字段模式由 TaskRepresentation 定义；这里提取共享图和实例差异。"""

    schema_version: int
    common_nodes: tuple[CommonNode, ...]
    common_edges: tuple[CommonEdge, ...]
    variants: dict[str, TaskVariant]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _node_signature(node: FunctionNode) -> tuple:
    return (node.capability, tuple(sorted(node.inputs.items())),
            tuple(sorted(node.outputs.items())))


def _edge_roles(edge: Dependency, nodes: dict[str, FunctionNode]) -> CommonEdge:
    return CommonEdge(nodes[edge.source].role, edge.source_port,
                      nodes[edge.target].role, edge.target_port)


def build_template(tasks: Sequence[TaskRepresentation]) -> TaskTemplate:
    """以唯一 role 对齐，求严格公共节点/边交集，并保存其余图结构。

    这不是一般图的最大公共子图算法。不同任务中复用同一 role 时，
    能力和端口契约必须完全一致；节点参数和执行代价则可不同。
    """
    if not tasks:
        raise ValidationError("template requires at least one task")
    if any(task.schema_version != 2 for task in tasks):
        raise ValidationError("template requires schema_version 2 tasks")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValidationError("template task ids must be unique")

    role_maps = []
    signatures: dict[str, tuple] = {}
    for task in tasks:
        roles = {node.role: node for node in task.G.nodes}
        if None in roles or len(roles) != len(task.G.nodes):
            raise ValidationError(f"task {task.task_id} requires unique node roles")
        for role, node in roles.items():
            signature = _node_signature(node)
            if role in signatures and signatures[role] != signature:
                raise ValidationError(f"role {role} has conflicting capability or port contract")
            signatures[role] = signature
        role_maps.append(roles)

    common_roles = set.intersection(*(set(roles) for roles in role_maps))
    common_nodes = tuple(CommonNode(role, role_maps[0][role].capability,
                                    dict(role_maps[0][role].inputs),
                                    dict(role_maps[0][role].outputs))
                         for role in sorted(common_roles))

    edge_sets = []
    for task in tasks:
        by_id = {node.node_id: node for node in task.G.nodes}
        edge_sets.append({_edge_roles(edge, by_id) for edge in task.G.edges})
    common_edges = tuple(sorted(
        (edge for edge in set.intersection(*edge_sets)
         if edge.source_role in common_roles and edge.target_role in common_roles),
        key=lambda edge: (edge.source_role, edge.source_port, edge.target_role, edge.target_port),
    ))
    shared_edges = set(common_edges)
    shared_edge_index = {edge: index for index, edge in enumerate(common_edges)}

    variants = {}
    for task in tasks:
        by_id = {node.node_id: node for node in task.G.nodes}
        bindings = {node.role: NodeVariant(
            node.node_id, node.estimated_runtime_ms, dict(node.parameters),
            node.shareable, node.preemptible, dict(node.execution_profiles),
            node.simulation_profile,
        ) for node in task.G.nodes if node.role in common_roles}
        branch_edges = []
        edge_order = []
        for edge in task.G.edges:
            common_edge = _edge_roles(edge, by_id)
            if common_edge in shared_edges:
                edge_order.append(("common", shared_edge_index[common_edge]))
            else:
                edge_order.append(("branch", len(branch_edges)))
                branch_edges.append(edge)
        variants[task.task_id] = TaskVariant(
            task.task_id, task.S, task.Q, task.R, task.X,
            tuple(node.role for node in task.G.nodes), bindings,
            tuple(node for node in task.G.nodes if node.role not in common_roles),
            tuple(branch_edges), tuple(edge_order),
        )
    return TaskTemplate(2, common_nodes, common_edges, variants)


def restore_task(template: TaskTemplate, task_id: str) -> TaskRepresentation:
    """用公共骨架、任务参数及分支还原一条任务的完整功能 DAG。"""
    if task_id not in template.variants:
        raise ValidationError(f"task {task_id} is not in the template")
    variant = template.variants[task_id]
    nodes_by_role = {node.role: node for node in variant.branch_nodes}
    for shared in template.common_nodes:
        binding = variant.common_nodes[shared.role]
        nodes_by_role[shared.role] = FunctionNode(
            binding.node_id, shared.capability, dict(shared.inputs), dict(shared.outputs),
            binding.estimated_runtime_ms, dict(binding.parameters), shared.role,
            binding.shareable, binding.preemptible, dict(binding.execution_profiles),
            binding.simulation_profile,
        )
    nodes = tuple(nodes_by_role[role] for role in variant.node_order)
    common_edges = tuple(Dependency(
        nodes_by_role[edge.source_role].node_id, edge.source_port,
        nodes_by_role[edge.target_role].node_id, edge.target_port,
    ) for edge in template.common_edges)
    edges = tuple((common_edges if kind == "common" else variant.branch_edges)[index]
                  for kind, index in variant.edge_order)
    return TaskRepresentation(task_id, variant.semantics, variant.qos,
                              variant.resources, FunctionGraph(nodes, edges),
                              variant.state, template.schema_version)
