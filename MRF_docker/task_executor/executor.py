"""Run one packet through a task DAG using the runtime component registry."""

from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from task_repr.component_models.runtime_registry import (RUNTIME_COMPONENT_REGISTRY,
                                                         create_runtime_component)
from task_repr.matching.contracts import (check_node_ports, validate_connection,
                                          validate_runtime_packet, validate_signal_packet)
from task_repr.matching.matcher import ComponentBinding, StaticMatchResult
from task_repr.models.component import ComponentInstance
from task_repr.models.task import TaskRepresentation
from task_repr.runtime import DataPacket, RuntimeMetrics
from task_repr.specs import load_component_specs, load_protocol_specs


class DAGExecutionError(RuntimeError):
    """任务图无法执行，或节点在运行时失败。"""


@dataclass(frozen=True)
class ExecutionResult:
    """保存节点输出、汇点输出、执行顺序和本次运行的测量结果。"""

    task_id: str
    completed: bool
    full_task_completed: bool
    execution_order: tuple[str, ...]
    instances: dict[str, ComponentInstance]
    node_outputs: dict[str, DataPacket]
    sink_outputs: dict[str, DataPacket]
    metrics: dict[str, RuntimeMetrics]


class DAGExecutor:
    """根据显式节点绑定执行已校验的任务图。

    CID 绑定使用节点参数；ComponentInstance 绑定可提供独立的运行参数。
    未绑定的根节点是数据源，由调用方按节点 ID 提供 DataPacket。
    每次 execute() 让选中节点各运行一次；active_nodes 选择原图的诱导子图。
    """

    def __init__(self, task: TaskRepresentation,
                 bindings: Mapping[str, str | ComponentInstance],
                 active_nodes: Iterable[str] | None = None) -> None:
        if not isinstance(task, TaskRepresentation):
            raise TypeError("task must be a TaskRepresentation")
        self.task = task
        self.status = "pending"
        all_nodes = {node.node_id: node for node in task.G.nodes}
        if not all_nodes or len(all_nodes) != len(task.G.nodes):
            raise DAGExecutionError("graph must have unique nodes")
        if isinstance(active_nodes, str):
            raise TypeError("active_nodes must be an iterable of node IDs")
        selected = set(all_nodes) if active_nodes is None else set(active_nodes)
        if not selected or selected - all_nodes.keys():
            raise DAGExecutionError(f"invalid active nodes: {sorted(selected - all_nodes.keys())}")
        self._nodes = {node.node_id: node for node in task.G.nodes if node.node_id in selected}
        self.covers_full_task = len(self._nodes) == len(task.G.nodes)
        self._incoming = {node_id: [] for node_id in self._nodes}
        self._outgoing = {node_id: [] for node_id in self._nodes}
        # 只保留所选子图内部的边，并检查任务声明的端口连接关系。
        used_inputs = set()
        for edge in task.G.edges:
            if edge.source not in all_nodes or edge.target not in all_nodes:
                raise DAGExecutionError("edge references an unknown node")
            if edge.source not in selected or edge.target not in selected:
                continue
            source, target = self._nodes[edge.source], self._nodes[edge.target]
            if (edge.source_port not in source.outputs or
                    edge.target_port not in target.inputs or
                    source.outputs[edge.source_port] != target.inputs[edge.target_port]):
                raise DAGExecutionError(f"incompatible graph ports: {edge}")
            target_port = (edge.target, edge.target_port)
            if target_port in used_inputs:
                raise DAGExecutionError(f"input port has multiple sources: {target_port}")
            used_inputs.add(target_port)
            self._incoming[edge.target].append(edge)
            self._outgoing[edge.source].append(edge)

        # Kahn 拓扑排序同时确定可执行顺序；未遍历完说明图中存在环。
        pending = {node_id: len(edges) for node_id, edges in self._incoming.items()}
        ready = deque(node_id for node_id in self._nodes if pending[node_id] == 0)
        order = []
        while ready:
            node_id = ready.popleft()
            order.append(node_id)
            for edge in self._outgoing[node_id]:
                pending[edge.target] -= 1
                if pending[edge.target] == 0:
                    ready.append(edge.target)
        if len(order) != len(self._nodes):
            raise DAGExecutionError("graph contains a dependency cycle")
        self.execution_order = tuple(order)
        self.roots = tuple(node_id for node_id in order if not self._incoming[node_id])
        self.sinks = tuple(node_id for node_id in order if not self._outgoing[node_id])

        unknown = set(bindings) - self._nodes.keys()
        if unknown:
            raise DAGExecutionError(f"bindings reference unknown nodes: {sorted(unknown)}")
        self.instances = {}
        instance_ids = set()
        # 当前单包适配器仅接受单输入/单输出组件；未绑定节点只能是数据源。
        for node_id, node in self._nodes.items():
            if len(node.outputs) != 1 or len(node.inputs) > 1 or len(self._incoming[node_id]) > 1:
                raise DAGExecutionError(f"node {node_id} needs a single input/output runtime adapter")
            binding = bindings.get(node_id)
            if binding is None:
                if node_id not in self.roots:
                    raise DAGExecutionError(f"node {node_id} needs a component binding")
                continue
            if isinstance(binding, str):
                instance = ComponentInstance(f"{task.task_id}:{node_id}", binding,
                                             task.task_id, dict(node.parameters))
            elif isinstance(binding, ComponentInstance):
                instance = binding
            else:
                raise TypeError(f"node {node_id} binding must be a CID or ComponentInstance")
            if instance.task_id != task.task_id:
                raise DAGExecutionError(f"node {node_id} instance belongs to another task")
            if instance.cid not in RUNTIME_COMPONENT_REGISTRY:
                raise DAGExecutionError(f"node {node_id}: no runtime implementation for {instance.cid}")
            if instance.instance_id in instance_ids:
                raise DAGExecutionError(f"duplicate component instance ID: {instance.instance_id}")
            instance_ids.add(instance.instance_id)
            self.instances[node_id] = instance

        self._strict = False
        self._bindings: dict[str, ComponentBinding] = {}
        self._protocols = None

    @classmethod
    def from_match(cls, task: TaskRepresentation,
                   match_result: StaticMatchResult) -> "DAGExecutor":
        """Build a fully checked executor from normalized static bindings."""
        if not isinstance(task, TaskRepresentation):
            raise TypeError("task must be a TaskRepresentation")
        if not isinstance(match_result, StaticMatchResult):
            raise TypeError("match_result must be a StaticMatchResult")
        if match_result.task_id != task.task_id or match_result.status != "MATCHED" or match_result.unresolved:
            raise DAGExecutionError("full execution requires a MATCHED result for this task")
        nodes = {node.node_id: node for node in task.G.nodes}
        expected_edges = [(edge.source, edge.target) for edge in task.G.edges]
        if (set(match_result.bindings) != set(nodes) or
                sorted(match_result.edges) != sorted(expected_edges)):
            raise DAGExecutionError("match result does not cover all task nodes and edges")
        bindings = {}
        selected = {}
        for node_id, binding in match_result.bindings.items():
            if not isinstance(binding, ComponentBinding) or not isinstance(binding.parameters, dict):
                raise DAGExecutionError(f"node {node_id} has an invalid match binding")
            selected[node_id] = ComponentBinding(binding.cid, deepcopy(binding.parameters))
            bindings[node_id] = ComponentInstance(f"{task.task_id}:{node_id}", binding.cid,
                                                  task.task_id, deepcopy(binding.parameters))
        executor = cls(task, bindings)
        executor._strict = True
        executor._bindings = selected
        executor._protocols = load_protocol_specs()
        executor._check_contracts(executor._create_components())
        return executor

    def _create_components(self):
        components = {}
        for node_id, instance in self.instances.items():
            try:
                components[node_id] = create_runtime_component(instance.cid,
                                                                **instance.parameters)
            except (TypeError, ValueError) as exc:
                raise DAGExecutionError(f"cannot create node {node_id} ({instance.cid}): {exc}") from exc
        return components

    def _check_contracts(self, components):
        interfaces = load_component_specs(protocols=self._protocols)
        for node_id, node in self._nodes.items():
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise DAGExecutionError(f"node {node_id} needs a single input/output runtime adapter")
            instance = self.instances[node_id]
            spec = interfaces.get(instance.cid)
            if spec is None:
                raise DAGExecutionError(f"node {node_id}: no OCS for {instance.cid}")
            error = spec.check_required(instance.parameters)
            for name, value in instance.parameters.items():
                error = error or spec.check_property(name, value)
            error = error or check_node_ports(node, components[node_id], spec,
                                              self._protocols, instance.parameters)
            if error:
                raise DAGExecutionError(f"node {node_id}: {error}")
        for edge in self.task.G.edges:
            error = validate_connection(self.task, edge, self._nodes[edge.source],
                                        self._nodes[edge.target], components[edge.source],
                                        components[edge.target], self._bindings[edge.source],
                                        self._bindings[edge.target])
            if error:
                raise DAGExecutionError(f"edge {edge.source}->{edge.target}: {error}")

    def _check_packet(self, node_id, packet, direction, component):
        node = self._nodes[node_id]
        formats = node.inputs if direction == "input" else node.outputs
        port = component.input_port if direction == "input" else component.output_port
        error = validate_runtime_packet(next(iter(formats.values())), packet,
                                        port, self._protocols)
        error = error or validate_signal_packet(self.task, node, packet, direction,
                                                self._bindings[node_id])
        if error:
            raise DAGExecutionError(f"node {node_id} {direction}: {error}")

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def execute(self, sources: Mapping[str, DataPacket]) -> ExecutionResult:
        """让每个选中节点处理一次数据包，并返回所有汇点的输出。"""
        if self.status == "running":
            raise DAGExecutionError("executor is already running")
        self.status = "running"
        try:
            if not isinstance(sources, Mapping):
                raise TypeError("sources must map root node IDs to DataPacket values")
            missing = set(self.roots) - sources.keys()
            extra = sources.keys() - set(self.roots)
            if missing or extra:
                raise DAGExecutionError(f"source packets: missing={sorted(missing)}, extra={sorted(extra)}")
            for node_id, packet in sources.items():
                if not isinstance(packet, DataPacket):
                    raise TypeError(f"source {node_id} requires a DataPacket")

            # 每次执行重新创建组件，避免上次运行的状态和指标影响本次结果。
            components = self._create_components()
            if self._strict:
                self._check_contracts(components)

            # 按拓扑顺序沿边取上游输出；未绑定根节点直接转发源数据包。
            outputs = {}
            for node_id in self.execution_order:
                packet = (sources[node_id] if node_id in self.roots else
                          outputs[self._incoming[node_id][0].source])
                component = components.get(node_id)
                if component is None:
                    outputs[node_id] = packet
                    continue
                if self._strict:
                    self._check_packet(node_id, packet, "input", component)
                try:
                    outputs[node_id] = component.process(packet)
                except Exception as exc:
                    raise DAGExecutionError(f"node {node_id} ({component.cid}) failed: {exc}") from exc
                if self._strict:
                    self._check_packet(node_id, outputs[node_id], "output", component)

            # 子图执行完成不等于整个任务完成，分别记录两种完成状态。
            result = ExecutionResult(
                task_id=self.task.task_id,
                completed=len(outputs) == len(self._nodes),
                full_task_completed=self.covers_full_task and len(outputs) == len(self._nodes),
                execution_order=self.execution_order,
                instances=dict(self.instances),
                node_outputs=outputs,
                sink_outputs={node_id: outputs[node_id] for node_id in self.sinks},
                metrics={node_id: component.metrics for node_id, component in components.items()},
            )
            self.status = "completed"
            return result
        except Exception:
            self.status = "failed"
            raise
