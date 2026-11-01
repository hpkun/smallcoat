"""Conservative normalized-capacity planning for CPU reference components."""

from dataclasses import asdict, dataclass
import math

from components.catalog import ComponentCatalog, default_component_catalog
from matching.matcher import StaticMatchResult
from task_repr.models.task import TaskRepresentation

from .estimator import ResourceEstimator
from .hardware import HardwareProfile
from .models import ResourceProfile


@dataclass(frozen=True)
class ResourceUsage:
    cpu: float = 0.0
    memory: float = 0.0
    bandwidth: float = 0.0

    def __add__(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(self.cpu + other.cpu,
                             self.memory + other.memory,
                             self.bandwidth + other.bandwidth)

    def __sub__(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(self.cpu - other.cpu,
                             self.memory - other.memory,
                             self.bandwidth - other.bandwidth)


@dataclass(frozen=True)
class NodeAllocation:
    node_id: str
    cid: str
    demand: ResourceUsage
    profile: ResourceProfile

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "cid": self.cid,
                "processor": "cpu", "demand": asdict(self.demand),
                "profile": asdict(self.profile)}


@dataclass(frozen=True)
class ResourcePlan:
    task_id: str
    status: str
    allocations: tuple[NodeAllocation, ...]
    total_demand: ResourceUsage
    available: ResourceUsage
    estimated_serial_ms: float
    time_limit_ms: float
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "allocations": [item.to_dict() for item in self.allocations],
                "total_demand": asdict(self.total_demand),
                "available": asdict(self.available),
                "estimated_serial_ms": self.estimated_serial_ms,
                "time_limit_ms": self.time_limit_ms,
                "issues": list(self.issues)}


class ResourcePlanner:
    """Reserve normalized CPU reference capacity for a complete static match.

    Resource levels are simulation pressure, not measured CPU or accelerator use.
    The current DAG executor runs one node at a time. Memory stays reserved for
    all instances; CPU and bandwidth need the largest active node demand.
    """

    def __init__(self, catalog: ComponentCatalog | None = None,
                 estimator: ResourceEstimator | None = None):
        self.catalog = catalog or default_component_catalog()
        self.estimator = estimator or ResourceEstimator()

    def plan(self, task: TaskRepresentation, match: StaticMatchResult,
             capacity: HardwareProfile | None = None,
             used: ResourceUsage | None = None) -> ResourcePlan:
        if match.task_id != task.task_id or match.status != "MATCHED":
            raise ValueError("resource planning requires a complete match for the same task")
        capacity = capacity or HardwareProfile()
        used = used or ResourceUsage()
        available = ResourceUsage(capacity.cpu_capacity - used.cpu,
                                  capacity.memory_capacity - used.memory,
                                  capacity.bandwidth_capacity - used.bandwidth)
        for value in (*asdict(available).values(), *asdict(used).values()):
            if not math.isfinite(value):
                raise ValueError("resource capacity and usage must be finite")
        if min(asdict(used).values()) < 0:
            raise ValueError("resource usage cannot be negative")

        allocations = []
        total = ResourceUsage()
        for node in task.G.nodes:
            binding = match.bindings[node.node_id]
            profile = self.estimator.estimate(
                self.catalog.get(binding.cid), binding.parameters, capacity)
            demand = ResourceUsage(profile.cpu, profile.memory, profile.bandwidth)
            allocations.append(NodeAllocation(node.node_id, binding.cid,
                                              demand, profile))
            total = ResourceUsage(max(total.cpu, demand.cpu),
                                  total.memory + demand.memory,
                                  max(total.bandwidth, demand.bandwidth))

        # Scenario estimates are inputs, not measured runtime guarantees.
        estimated_serial = sum(node.estimated_runtime_ms for node in task.G.nodes)
        time_limit = min(value for value in (
            task.X.remaining_deadline_ms, task.Q.max_latency_ms,
            task.Q.max_response_ms) if value is not None)
        issues = []
        for name in ("cpu", "memory", "bandwidth"):
            if getattr(total, name) > getattr(available, name):
                issues.append(f"{name} demand exceeds available normalized capacity")
        if estimated_serial > time_limit:
            issues.append("estimated serial runtime exceeds task time limit")
        return ResourcePlan(task.task_id,
                            "INFEASIBLE" if issues else "FEASIBLE_ESTIMATE",
                            tuple(allocations), total, available,
                            estimated_serial, time_limit, tuple(issues))
