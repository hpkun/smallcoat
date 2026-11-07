"""基于静态匹配结果估算 CPU 参考组件的归一化资源容量。"""

from dataclasses import asdict, dataclass
import itertools
import math

from components.catalog import ComponentCatalog, default_component_catalog
from matching.matcher import StaticMatchResult
from task_repr.models.task import TaskRepresentation
from task_repr.template.refinement import SpecificationTask

from .estimator import ResourceEstimator
from .hardware import HardwareProfile
from .models import ResourceProfile
from .measurements import (ComponentMeasurement, MeasurementCatalog,
                           PhysicalHardwareProfile, PhysicalResourceState,
                           PhysicalUsage)


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
    estimated_serial_ms: float | None
    time_limit_ms: float | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "unit_system": "normalized_simulation",
                "allocations": [item.to_dict() for item in self.allocations],
                "total_demand": asdict(self.total_demand),
                "available": asdict(self.available),
                "estimated_serial_ms": self.estimated_serial_ms,
                "time_limit_ms": self.time_limit_ms,
                "issues": list(self.issues)}


@dataclass(frozen=True)
class PhysicalAllocation:
    node_id: str
    measurement: ComponentMeasurement

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "cid": self.measurement.cid,
                "implementation_id": self.measurement.implementation_id,
                "processor": self.measurement.processor,
                "demand": asdict(self.measurement.demand),
                "runtime_p95_ms": self.measurement.runtime_p95_ms,
                "setup_p95_ms": self.measurement.setup_p95_ms,
                "input_bytes": self.measurement.input_bytes,
                "output_bytes": self.measurement.output_bytes,
                "sample_count": self.measurement.sample_count,
                "source": self.measurement.source}


@dataclass(frozen=True)
class PhysicalResourcePlan:
    task_id: str
    status: str
    hardware_id: str
    allocations: tuple[PhysicalAllocation, ...]
    total_demand: PhysicalUsage | None
    available: PhysicalUsage
    estimated_serial_ms: float | None
    time_limit_ms: float | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "unit_system": "physical",
                "hardware_id": self.hardware_id,
                "units": {"cpu_millicores": "millicores", "memory_bytes": "bytes",
                          "bandwidth_bytes_per_s": "bytes/s", "dsp_units": "units",
                          "fpga_units": "units", "runtime_p95_ms": "ms",
                          "setup_p95_ms": "ms", "input_bytes": "bytes",
                          "output_bytes": "bytes"},
                "allocations": [item.to_dict() for item in self.allocations],
                "total_demand": asdict(self.total_demand) if self.total_demand else None,
                "available": asdict(self.available),
                "estimated_serial_ms": self.estimated_serial_ms,
                "time_limit_ms": self.time_limit_ms, "issues": list(self.issues)}


def _time_limit(task: TaskRepresentation | SpecificationTask) -> float | None:
    if isinstance(task, SpecificationTask):
        remaining = task.X.get("remaining_deadline_ms")
        deadline = task.Q.get("deadline_ms")
        observed = task.X.get("observed_at_ms")
        limits = [remaining] if remaining is not None else []
        if deadline is not None and observed is not None:
            limits.append(deadline - observed)
    else:
        limits = [task.X.remaining_deadline_ms, task.Q.max_latency_ms,
                  task.Q.max_response_ms]
    for value in limits:
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("task time limit must be finite")
    return min((value for value in limits if value is not None), default=None)


def _physical_demand(items: tuple[ComponentMeasurement, ...]) -> PhysicalUsage:
    return PhysicalUsage(
        cpu_millicores=max((item.demand.cpu_millicores for item in items), default=0),
        memory_bytes=sum(item.demand.memory_bytes for item in items),
        bandwidth_bytes_per_s=max((item.demand.bandwidth_bytes_per_s for item in items), default=0),
        dsp_units=max((item.demand.dsp_units for item in items), default=0),
        fpga_units=max((item.demand.fpga_units for item in items), default=0))


class ResourcePlanner:
    """为完整匹配的任务估算串行 CPU 执行所需的归一化容量。"""

    def __init__(self, catalog: ComponentCatalog | None = None,
                 estimator: ResourceEstimator | None = None):
        self.catalog = catalog or default_component_catalog()
        self.estimator = estimator or ResourceEstimator()

    def plan_physical(self, task: TaskRepresentation | SpecificationTask,
                      match: StaticMatchResult, hardware: PhysicalHardwareProfile,
                      measurements: MeasurementCatalog,
                      state: PhysicalResourceState) -> PhysicalResourcePlan:
        """使用同一硬件上的实测画像规划；缺少记录时不推断物理用量。"""
        if match.task_id != task.task_id or match.status != "MATCHED" or set(match.bindings) != {node.node_id for node in task.G.nodes}:
            raise ValueError("physical planning requires complete bindings for the same task")
        if state.hardware_id != hardware.hardware_id:
            raise ValueError("physical resource state must belong to the selected hardware")
        used = state.used
        for label, usage in (("capacity", hardware.capacity), ("used", used)):
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in asdict(usage).values()):
                raise ValueError(f"physical {label} must be finite and nonnegative")
        available = PhysicalUsage(**{key: value - getattr(used, key)
                                     for key, value in asdict(hardware.capacity).items()})
        limit = _time_limit(task)
        options = []
        missing = []
        for node in task.G.nodes:
            binding = match.bindings[node.node_id]
            candidates = measurements.candidates(binding.cid, binding.parameters, hardware)
            if not candidates:
                missing.append(f"{node.node_id}: no measurement for {binding.cid} on {hardware.hardware_id} with matched parameters")
            options.append(candidates)
        if missing:
            return PhysicalResourcePlan(task.task_id, "NEEDS_CALIBRATION", hardware.hardware_id,
                                        (), None, available, None, limit, tuple(missing))

        # 当前执行路径串行：处理器及流带宽取节点峰值，常驻实例内存求和。
        combinations = itertools.product(*options)
        best = None
        best_feasible = None
        for items in combinations:
            demand = _physical_demand(items)
            duration = sum(item.runtime_p95_ms + item.setup_p95_ms for item in items)
            candidate = (duration, items, demand)
            if best is None or duration < best[0]:
                best = candidate
            if all(getattr(demand, key) <= getattr(available, key)
                   for key in asdict(demand)) and (best_feasible is None or duration < best_feasible[0]):
                best_feasible = candidate
        duration, items, demand = best_feasible or best
        capacity_issues = [f"{key} demand exceeds available physical capacity"
                           for key in asdict(demand)
                           if getattr(demand, key) > getattr(available, key)]
        issues = list(capacity_issues)
        if limit is not None and limit <= 0:
            issues.append("task completion deadline has expired")
        elif limit is not None and duration > limit:
            issues.append("measured component time sum exceeds task time limit")
        if limit is None:
            issues.append("task has no explicit completion deadline")
        status = "INFEASIBLE" if capacity_issues or (limit is not None and (limit <= 0 or duration > limit)) else (
            "UNVERIFIED_TIME" if limit is None else "FEASIBLE_MEASURED_ESTIMATE")
        allocations = tuple(PhysicalAllocation(node.node_id, item)
                            for node, item in zip(task.G.nodes, items))
        return PhysicalResourcePlan(task.task_id, status, hardware.hardware_id,
                                    allocations, demand, available, duration, limit,
                                    tuple(issues))

    def plan(self, task: TaskRepresentation | SpecificationTask, match: StaticMatchResult,
             capacity: HardwareProfile | None = None,
             used: ResourceUsage | None = None) -> ResourcePlan:
        if match.task_id != task.task_id or match.status != "MATCHED":
            raise ValueError("resource planning requires a complete match for the same task")
        node_ids = {node.node_id for node in task.G.nodes}
        if set(match.bindings) != node_ids:
            raise ValueError("resource planning requires one binding for every task node")
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
        runtimes = []
        for node in task.G.nodes:
            binding = match.bindings[node.node_id]
            definition = self.catalog.get(binding.cid)
            profile = self.estimator.estimate(definition, binding.parameters, capacity)
            demand = ResourceUsage(profile.cpu, profile.memory, profile.bandwidth)
            if any(not math.isfinite(value) or value < 0 for value in asdict(demand).values()):
                raise ValueError(f"invalid resource estimate for {node.node_id}")
            allocations.append(NodeAllocation(node.node_id, binding.cid,
                                              demand, profile))
            total = ResourceUsage(max(total.cpu, demand.cpu),
                                  total.memory + demand.memory,
                                  max(total.bandwidth, demand.bandwidth))
            if isinstance(task, SpecificationTask):
                runtimes.append(profile.runtime_ms if profile.runtime_ms is not None
                                else definition.estimated_runtime_ms)
            else:
                runtimes.append(node.estimated_runtime_ms)

        # 未标定的运行时间和未指定的截止时间不能补成虚构数值。
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in runtimes):
            raise ValueError("component runtime estimates must be finite and nonnegative")
        estimated_serial = sum(runtimes) if all(value is not None for value in runtimes) else None
        time_limit = _time_limit(task)
        issues = []
        infeasible = False
        for name in ("cpu", "memory", "bandwidth"):
            if getattr(total, name) > getattr(available, name):
                issues.append(f"{name} demand exceeds available normalized capacity")
                infeasible = True
        if time_limit is not None and time_limit <= 0:
            issues.append("task completion deadline has expired")
            infeasible = True
        elif estimated_serial is not None and time_limit is not None and estimated_serial > time_limit:
            issues.append("estimated serial runtime exceeds task time limit")
            infeasible = True
        if isinstance(task, SpecificationTask):
            if estimated_serial is None:
                issues.append("component runtime estimates are unavailable for task time verification")
            if time_limit is None:
                issues.append("task has no explicit completion deadline")
        return ResourcePlan(task.task_id,
                            "INFEASIBLE" if infeasible else
                            "UNVERIFIED_TIME" if isinstance(task, SpecificationTask) and issues else
                            "FEASIBLE_ESTIMATE",
                            tuple(allocations), total, available,
                            estimated_serial, time_limit, tuple(issues))
