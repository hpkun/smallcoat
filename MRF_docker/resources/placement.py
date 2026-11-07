"""固定组件 DAG 在处理节点上的初始分配与有界优化搜索。"""

from dataclasses import asdict, dataclass, replace
import math

from matching.matcher import StaticMatchResult
from task_repr.models.task import TaskRepresentation
from task_repr.template.refinement import SpecificationTask

from .measurements import (ComponentMeasurement, MeasurementCatalog,
                           PhysicalHardwareProfile, PhysicalUsage)
from .planner import _time_limit


_FIELDS = tuple(PhysicalUsage.__dataclass_fields__)


@dataclass(frozen=True)
class ProcessingNode:
    node_id: str
    hardware: PhysicalHardwareProfile


@dataclass(frozen=True)
class NetworkLink:
    source: str
    target: str
    capacity_bytes_per_s: float
    latency_ms: float

    def __post_init__(self):
        if (not self.source or not self.target or self.source == self.target or
                not math.isfinite(self.capacity_bytes_per_s) or self.capacity_bytes_per_s <= 0 or
                not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("network link needs distinct endpoints, positive rate and nonnegative latency")


@dataclass(frozen=True)
class EdgeTransfer:
    source: str
    target: str
    bytes_per_call: int
    reserved_bytes_per_s: float

    def __post_init__(self):
        if (not self.source or not self.target or type(self.bytes_per_call) is not int or
                self.bytes_per_call < 0 or not math.isfinite(self.reserved_bytes_per_s) or
                self.reserved_bytes_per_s <= 0):
            raise ValueError("edge transfer needs nonnegative bytes and positive reserved rate")


@dataclass(frozen=True)
class PlacementState:
    node_used: dict[str, PhysicalUsage]
    link_used: dict[tuple[str, str], float]


@dataclass(frozen=True)
class PlacementAllocation:
    node_id: str
    cid: str
    implementation_id: str
    container_unit_id: str
    processing_node_id: str
    processor: str
    demand: PhysicalUsage
    runtime_p95_ms: float
    setup_p95_ms: float

    def to_dict(self) -> dict:
        return {**asdict(self), "demand": asdict(self.demand)}


@dataclass(frozen=True)
class PlacementPlan:
    task_id: str
    status: str
    allocations: tuple[PlacementAllocation, ...]
    node_demand: dict[str, PhysicalUsage]
    link_demand: dict[tuple[str, str], float]
    edge_transfer_ms: dict[tuple[str, str], float]
    estimated_completion_ms: float | None
    time_limit_ms: float | None
    issues: tuple[str, ...]
    explored: int = 0
    exhaustive: bool = True

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "unit_system": "physical", "allocations": [item.to_dict() for item in self.allocations],
                "node_demand": {key: asdict(value) for key, value in self.node_demand.items()},
                "link_demand": {f"{a}->{b}": value for (a, b), value in self.link_demand.items()},
                "edge_transfer_ms": {f"{a}->{b}": value for (a, b), value in self.edge_transfer_ms.items()},
                "estimated_completion_ms": self.estimated_completion_ms,
                "time_limit_ms": self.time_limit_ms, "issues": list(self.issues),
                "explored": self.explored, "exhaustive": self.exhaustive}


def _add(left: PhysicalUsage, right: PhysicalUsage) -> PhysicalUsage:
    return PhysicalUsage(**{key: getattr(left, key) + getattr(right, key) for key in _FIELDS})


def _fits(demand: PhysicalUsage, available: PhysicalUsage) -> bool:
    return all(getattr(demand, key) <= getattr(available, key) for key in _FIELDS)


def _order(task) -> tuple[str, ...]:
    ids = {node.node_id for node in task.G.nodes}
    incoming = {key: 0 for key in ids}
    outgoing = {key: [] for key in ids}
    seen = set()
    for edge in task.G.edges:
        pair = (edge.source, edge.target)
        if edge.source not in ids or edge.target not in ids or pair in seen:
            raise ValueError("DAG has an invalid or duplicate dependency")
        seen.add(pair)
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)
    ready = sorted(key for key, count in incoming.items() if count == 0)
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in outgoing[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(ids):
        raise ValueError("task graph must be acyclic")
    return tuple(order)


class PlacementPlanner:
    """比较固定 CID 绑定的实现与处理节点，不修改静态匹配结果。"""

    def __init__(self, nodes: tuple[ProcessingNode, ...], links: tuple[NetworkLink, ...],
                 measurements: MeasurementCatalog,
                 transfers: tuple[EdgeTransfer, ...]):
        self.nodes = {item.node_id: item for item in nodes}
        self.links = {(item.source, item.target): item for item in links}
        self.measurements = measurements
        self.transfers = {(item.source, item.target): item for item in transfers}
        if (len(self.nodes) != len(nodes) or len(self.links) != len(links) or
                len(self.transfers) != len(transfers)):
            raise ValueError("processing nodes, links and transfers must have unique identifiers")
        if any(link.source not in self.nodes or link.target not in self.nodes for link in links):
            raise ValueError("network link endpoint is not a processing node")

    def plan(self, task: TaskRepresentation | SpecificationTask, match: StaticMatchResult,
             state: PlacementState, fixed_nodes: dict[str, str] | None = None,
             *, max_assignments: int = 100000) -> PlacementPlan:
        if type(max_assignments) is not int or max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        if not match.candidate_bindings:
            return self._plan_single(task, match, state, fixed_nodes,
                                     max_assignments=max_assignments)
        plans = []
        remaining = max_assignments
        for bindings in match.candidate_bindings:
            if remaining <= 0:
                break
            candidate = self._plan_single(task, replace(match, bindings=bindings,
                                                        candidate_bindings=()), state,
                                          fixed_nodes, max_assignments=remaining)
            plans.append(candidate)
            remaining -= candidate.explored
            if not candidate.exhaustive:
                break
        priority = {"FEASIBLE_ESTIMATE": 0, "UNVERIFIED_TIME": 1,
                    "SEARCH_LIMIT": 2, "INFEASIBLE": 3,
                    "NEEDS_CALIBRATION": 4, "NEEDS_TRANSFER_PROFILE": 5}
        chosen = min(plans, key=lambda plan: (priority[plan.status],
                                              plan.estimated_completion_ms or math.inf))
        exhaustive = (match.candidates_exhaustive and
                      len(plans) == len(match.candidate_bindings) and
                      all(plan.exhaustive for plan in plans))
        if not exhaustive and chosen.status != "SEARCH_LIMIT":
            return replace(chosen, status="SEARCH_LIMIT", exhaustive=False,
                           explored=sum(plan.explored for plan in plans),
                           issues=chosen.issues + ("component or placement search limit reached",))
        return replace(chosen, explored=sum(plan.explored for plan in plans),
                       exhaustive=exhaustive)

    def _plan_single(self, task: TaskRepresentation | SpecificationTask, match: StaticMatchResult,
             state: PlacementState, fixed_nodes: dict[str, str] | None = None,
             *, max_assignments: int = 100000) -> PlacementPlan:
        if type(max_assignments) is not int or max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        order = _order(task)
        node_ids = set(order)
        graph_edges = {(edge.source, edge.target) for edge in task.G.edges}
        if (match.task_id != task.task_id or match.status != "MATCHED" or
                set(match.bindings) != node_ids or set(match.edges) != graph_edges):
            raise ValueError("placement requires the complete matched DAG")
        if fixed_nodes is not None and set(fixed_nodes) != node_ids:
            raise ValueError("fixed_nodes must assign every DAG node")
        if fixed_nodes is not None and set(fixed_nodes.values()) - set(self.nodes):
            raise ValueError("fixed_nodes references an unknown processing node")
        if (set(state.node_used) != set(self.nodes) or
                set(state.link_used) != set(self.links)):
            raise ValueError("placement state must cover every processing node and link")
        for node_id, used in state.node_used.items():
            if any(not math.isfinite(value) or value < 0 for value in asdict(used).values()):
                raise ValueError(f"invalid resource state for {node_id}")
        for pair, used in state.link_used.items():
            if not math.isfinite(used) or used < 0:
                raise ValueError(f"invalid link state for {pair}")

        limit = _time_limit(task)
        def result(status, issues, *, allocations=(), demand=None, links=None, transfers=None,
                   duration=None, explored=0, exhaustive=True):
            return PlacementPlan(task.task_id, status, tuple(allocations), demand or {}, links or {},
                                 transfers or {}, duration, limit, tuple(issues), explored, exhaustive)

        missing_transfers = sorted(graph_edges - set(self.transfers))
        if missing_transfers:
            return result("NEEDS_TRANSFER_PROFILE", [f"missing edge transfer profile: {a}->{b}"
                                                     for a, b in missing_transfers])
        options = {}
        missing = []
        for node_id in order:
            binding = match.bindings[node_id]
            allowed = (fixed_nodes[node_id],) if fixed_nodes is not None else tuple(self.nodes)
            candidates = [(host_id, item)
                          for host_id in allowed if host_id in self.nodes
                          for item in self.measurements.candidates(
                              binding.cid, binding.parameters, self.nodes[host_id].hardware)]
            if not candidates:
                missing.append(f"{node_id}: no compatible measurement or processing node for {binding.cid}")
            options[node_id] = candidates
        if missing:
            return result("NEEDS_CALIBRATION", missing)

        available = {key: PhysicalUsage(**{
            field: getattr(node.hardware.capacity, field) - getattr(state.node_used[key], field)
            for field in _FIELDS}) for key, node in self.nodes.items()}
        adjacency = {node_id: [] for node_id in order}
        for edge in task.G.edges:
            adjacency[edge.target].append(edge.source)
        selected = {}
        demands = {key: PhysicalUsage() for key in self.nodes}
        link_demands = {key: 0.0 for key in self.links}
        best = None
        explored = 0
        exhaustive = True
        reasons = set()

        def search(index):
            nonlocal best, explored, exhaustive
            if index == len(order):
                finish = {}
                edge_ms = {}
                for node_id in order:
                    host_id, measurement = selected[node_id]
                    start = 0.0
                    for predecessor in adjacency[node_id]:
                        source_host, _ = selected[predecessor]
                        transfer_ms = 0.0
                        if source_host != host_id:
                            transfer = self.transfers[(predecessor, node_id)]
                            link = self.links[(source_host, host_id)]
                            transfer_ms = link.latency_ms + transfer.bytes_per_call * 1000 / transfer.reserved_bytes_per_s
                        edge_ms[(predecessor, node_id)] = transfer_ms
                        start = max(start, finish[predecessor] + transfer_ms)
                    finish[node_id] = start + measurement.setup_p95_ms + measurement.runtime_p95_ms
                duration = max(finish.values(), default=0)
                if best is None or duration < best[0]:
                    best = (duration, dict(selected), dict(demands), dict(link_demands), edge_ms)
                return
            node_id = order[index]
            for host_id, measurement in options[node_id]:
                if explored >= max_assignments:
                    exhaustive = False
                    return
                explored += 1
                next_demand = _add(demands[host_id], measurement.demand)
                if not _fits(next_demand, available[host_id]):
                    reasons.add(f"{host_id}: node capacity exceeded by {node_id}")
                    continue
                added_links = []
                failure = None
                for predecessor in adjacency[node_id]:
                    source_host, source_measurement = selected[predecessor]
                    transfer = self.transfers[(predecessor, node_id)]
                    if (transfer.bytes_per_call > source_measurement.output_bytes or
                            transfer.bytes_per_call > measurement.input_bytes):
                        failure = f"{predecessor}->{node_id}: transfer bytes exceed measured output or input"
                        break
                    if source_host == host_id:
                        continue
                    pair = (source_host, host_id)
                    if pair not in self.links:
                        failure = f"{predecessor}->{node_id}: no link {source_host}->{host_id}"
                        break
                    requested = link_demands[pair] + transfer.reserved_bytes_per_s
                    if requested + state.link_used[pair] > self.links[pair].capacity_bytes_per_s:
                        failure = f"{pair[0]}->{pair[1]}: link capacity exceeded"
                        break
                    link_demands[pair] = requested
                    added_links.append((pair, transfer.reserved_bytes_per_s))
                if failure:
                    reasons.add(failure)
                    for pair, rate in added_links:
                        link_demands[pair] -= rate
                    continue
                previous = demands[host_id]
                demands[host_id] = next_demand
                selected[node_id] = (host_id, measurement)
                search(index + 1)
                del selected[node_id]
                demands[host_id] = previous
                for pair, rate in added_links:
                    link_demands[pair] -= rate
                if not exhaustive:
                    break

        search(0)
        if best is None:
            return result("SEARCH_LIMIT" if not exhaustive else "INFEASIBLE",
                          sorted(reasons) or ["no feasible assignment"],
                          explored=explored, exhaustive=exhaustive)
        duration, placement, node_demand, link_demand, edge_ms = best
        allocations = tuple(PlacementAllocation(
            node_id, match.bindings[node_id].cid, measurement.implementation_id,
            f"{task.task_id}:{node_id}", host_id, measurement.processor,
            measurement.demand, measurement.runtime_p95_ms, measurement.setup_p95_ms)
            for node_id, (host_id, measurement) in ((key, placement[key]) for key in order))
        issues = []
        if limit is None:
            issues.append("task has no explicit completion deadline")
        elif limit <= 0:
            issues.append("task completion deadline has expired")
        elif duration > limit:
            issues.append("DAG completion time exceeds task time limit")
        if not exhaustive:
            issues.append("assignment search limit reached; optimum is unverified")
        status = ("INFEASIBLE" if limit is not None and (limit <= 0 or duration > limit) and exhaustive
                  else "SEARCH_LIMIT" if not exhaustive
                  else "UNVERIFIED_TIME" if limit is None else "FEASIBLE_ESTIMATE")
        return result(status, issues, allocations=allocations, demand=node_demand,
                      links=link_demand, transfers=edge_ms, duration=duration,
                      explored=explored, exhaustive=exhaustive)
