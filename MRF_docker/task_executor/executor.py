"""Execute one finite, single-packet pass through a task function DAG."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping

from task_repr.component_models.runtime_registry import create_runtime_component
from task_repr.models.component import ComponentInstance
from task_repr.models.task import TaskRepresentation
from task_repr.runtime import DataPacket, RuntimeMetrics


class DAGExecutionError(RuntimeError):
    """A graph cannot be executed or a node failed during execution."""


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    completed: bool
    full_task_completed: bool
    execution_order: tuple[str, ...]
    instances: dict[str, ComponentInstance]
    node_outputs: dict[str, DataPacket]
    sink_outputs: dict[str, DataPacket]
    metrics: dict[str, RuntimeMetrics]


class DAGExecutor:
    """Run a validated task graph using explicit node-to-component bindings.

    A binding is a CID, using the node's parameters, or a ComponentInstance
    with explicit runtime parameters. Unbound root nodes act as data sources
    whose output packets are supplied by the caller.
    Source packets are keyed by root node ID. Each node runs once per execute().
    active_nodes selects an induced subgraph of an existing task DAG.
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
            if instance.cid != "C08":
                raise DAGExecutionError(f"node {node_id}: only C08 FFT/IFFT is supported")
            if instance.instance_id in instance_ids:
                raise DAGExecutionError(f"duplicate component instance ID: {instance.instance_id}")
            instance_ids.add(instance.instance_id)
            self.instances[node_id] = instance

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def execute(self, sources: Mapping[str, DataPacket]) -> ExecutionResult:
        """Process every ready node once and return outputs from all DAG sinks."""
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

            components = {}
            for node_id, instance in self.instances.items():
                try:
                    components[node_id] = create_runtime_component(instance.cid,
                                                                    **instance.parameters)
                except (TypeError, ValueError) as exc:
                    raise DAGExecutionError(f"cannot create node {node_id} ({instance.cid}): {exc}") from exc

            outputs = {}
            for node_id in self.execution_order:
                packet = (sources[node_id] if node_id in self.roots else
                          outputs[self._incoming[node_id][0].source])
                component = components.get(node_id)
                if component is None:
                    outputs[node_id] = packet
                    continue
                try:
                    outputs[node_id] = component.process(packet)
                except Exception as exc:
                    raise DAGExecutionError(f"node {node_id} ({component.cid}) failed: {exc}") from exc

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
