"""Capacity-aware instance lifecycle simulation."""

from dataclasses import asdict, dataclass
from functools import wraps
from threading import RLock

from matching.matcher import StaticMatchResult
from resources.hardware import HardwareProfile
from resources.planner import ResourcePlan, ResourcePlanner, ResourceUsage
from resources.measurements import PhysicalUsage
from resources.placement import PlacementPlan, PlacementPlanner, PlacementState
from task_repr.models.task import TaskRepresentation


class SchedulingError(ValueError):
    pass


def synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


@dataclass
class InstanceRecord:
    instance_id: str
    cid: str
    parameters: dict
    memory: float
    state: str
    task_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScheduleDecision:
    task_id: str
    plan: ResourcePlan
    instance_ids: dict[str, str]
    reused: tuple[str, ...]
    evicted: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "plan": self.plan.to_dict(),
                "instance_ids": dict(self.instance_ids),
                "reused": list(self.reused), "evicted": list(self.evicted)}


@dataclass
class PlacementInstanceRecord:
    container_unit_id: str
    cid: str
    implementation_id: str
    processor: str
    processing_node_id: str
    memory_bytes: float
    state: str
    task_id: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class TaskScheduler:
    """Model active and warm instances without controlling Docker containers.

    Active tasks reserve their plan's CPU, memory and bandwidth. Warm instances
    retain only normalized memory until reused or evicted. Calls are serialized
    with a lock. This remains an in-memory, single-process scheduler.
    """

    def __init__(self, capacity: HardwareProfile | None = None,
                 planner: ResourcePlanner | None = None,
                 placement_planner: PlacementPlanner | None = None,
                 placement_baseline: PlacementState | None = None):
        self.capacity = capacity or HardwareProfile()
        self.planner = planner or ResourcePlanner()
        self.instances: dict[str, InstanceRecord] = {}
        self.active: dict[str, ScheduleDecision] = {}
        self._next_id = 1
        self._lock = RLock()
        self.placement_planner = placement_planner
        self.placement_baseline = placement_baseline
        self.placement_active: dict[str, PlacementPlan] = {}
        self.placement_instances: dict[str, PlacementInstanceRecord] = {}

    def _placement_state(self) -> PlacementState:
        if self.placement_planner is None:
            raise SchedulingError("placement planner is not configured")
        planner = self.placement_planner
        baseline = self.placement_baseline or PlacementState(
            {key: PhysicalUsage() for key in planner.nodes},
            {key: 0.0 for key in planner.links})
        node_used = dict(baseline.node_used)
        link_used = dict(baseline.link_used)
        for plan in self.placement_active.values():
            for key, demand in plan.node_demand.items():
                previous = node_used[key]
                node_used[key] = PhysicalUsage(**{
                    field: getattr(previous, field) + getattr(demand, field)
                    for field in PhysicalUsage.__dataclass_fields__})
            for key, demand in plan.link_demand.items():
                link_used[key] += demand
        for instance in self.placement_instances.values():
            if instance.state == "WARM":
                previous = node_used[instance.processing_node_id]
                node_used[instance.processing_node_id] = PhysicalUsage(
                    **{**asdict(previous), "memory_bytes": previous.memory_bytes + instance.memory_bytes})
        return PlacementState(node_used, link_used)

    @synchronized
    def preview_placement(self, task, match: StaticMatchResult,
                          fixed_nodes: dict[str, str] | None = None,
                          *, max_assignments: int = 100000) -> PlacementPlan:
        if task.task_id in self.active or task.task_id in self.placement_active:
            raise SchedulingError(f"task {task.task_id} is already active")
        if self.placement_planner is None:
            raise SchedulingError("placement planner is not configured")
        return self.placement_planner.plan(task, match, self._placement_state(),
                                           fixed_nodes, max_assignments=max_assignments)

    @synchronized
    def submit_placement(self, task, match: StaticMatchResult,
                         fixed_nodes: dict[str, str] | None = None,
                         *, max_assignments: int = 100000) -> PlacementPlan:
        plan = self.preview_placement(task, match, fixed_nodes,
                                      max_assignments=max_assignments)
        if plan.status != "FEASIBLE_ESTIMATE":
            raise SchedulingError("; ".join(plan.issues) or plan.status)
        for allocation in plan.allocations:
            identifier = allocation.container_unit_id
            if (identifier in self.placement_instances and
                    self.placement_instances[identifier].state != "RELEASED"):
                raise SchedulingError(f"container unit {identifier} already exists")
        for allocation in plan.allocations:
            self.placement_instances[allocation.container_unit_id] = PlacementInstanceRecord(
                allocation.container_unit_id, allocation.cid,
                allocation.implementation_id, allocation.processor,
                allocation.processing_node_id,
                allocation.demand.memory_bytes, "ACTIVE", task.task_id)
        self.placement_active[task.task_id] = plan
        return plan

    @synchronized
    def complete_placement(self, task_id: str, *, keep_warm: bool = False) -> None:
        if task_id not in self.placement_active:
            raise SchedulingError(f"task {task_id} is not active")
        plan = self.placement_active.pop(task_id)
        for allocation in plan.allocations:
            instance = self.placement_instances[allocation.container_unit_id]
            instance.state = "WARM" if keep_warm else "RELEASED"
            instance.task_id = None

    @synchronized
    def evict_placement_warm(self, container_unit_id: str) -> None:
        instance = self.placement_instances.get(container_unit_id)
        if instance is None or instance.state != "WARM":
            raise SchedulingError(f"container unit {container_unit_id} is not warm")
        instance.state = "RELEASED"

    @synchronized
    def placement_snapshot(self) -> dict:
        state = self._placement_state()
        return {"node_used": {key: asdict(value) for key, value in state.node_used.items()},
                "link_used": {f"{a}->{b}": value for (a, b), value in state.link_used.items()},
                "active_tasks": list(self.placement_active),
                "instances": [item.to_dict() for item in self.placement_instances.values()]}

    def _used(self) -> ResourceUsage:
        active = ResourceUsage()
        for decision in self.active.values():
            active += decision.plan.total_demand
        warm_memory = sum(instance.memory for instance in self.instances.values()
                          if instance.state == "WARM")
        return active + ResourceUsage(memory=warm_memory)

    def _warm_matches(self, match: StaticMatchResult) -> dict[str, InstanceRecord]:
        result = {}
        chosen = set()
        for node_id, binding in match.bindings.items():
            for instance in self.instances.values():
                if (instance.state == "WARM" and instance.instance_id not in chosen
                        and instance.cid == binding.cid
                        and instance.parameters == binding.parameters):
                    result[node_id] = instance
                    chosen.add(instance.instance_id)
                    break
        return result

    @synchronized
    def preview(self, task: TaskRepresentation,
                match: StaticMatchResult) -> tuple[ResourcePlan, tuple[str, ...], tuple[str, ...]]:
        if match.task_id != task.task_id or match.status != "MATCHED":
            raise SchedulingError("scheduling requires a complete match for this task")
        if task.task_id in self.active:
            raise SchedulingError(f"task {task.task_id} is already active")
        warm = self._warm_matches(match)
        reused_memory = sum(instance.memory for instance in warm.values())
        used = self._used() - ResourceUsage(memory=reused_memory)
        plan = self.planner.plan(task, match, self.capacity, used)
        evicted = []
        if plan.status != "FEASIBLE_ESTIMATE" and any(
                issue.startswith("memory demand") for issue in plan.issues):
            reusable_ids = {instance.instance_id for instance in warm.values()}
            candidates = sorted((instance for instance in self.instances.values()
                                 if instance.state == "WARM"
                                 and instance.instance_id not in reusable_ids),
                                key=lambda instance: instance.memory, reverse=True)
            for instance in candidates:
                used -= ResourceUsage(memory=instance.memory)
                evicted.append(instance.instance_id)
                plan = self.planner.plan(task, match, self.capacity, used)
                if plan.status == "FEASIBLE_ESTIMATE":
                    break
        return plan, tuple(warm), tuple(evicted)

    @synchronized
    def submit(self, task: TaskRepresentation,
               match: StaticMatchResult) -> ScheduleDecision:
        plan, reused, evicted = self.preview(task, match)
        if plan.status != "FEASIBLE_ESTIMATE":
            raise SchedulingError("; ".join(plan.issues))
        warm = self._warm_matches(match)

        for instance_id in evicted:
            self.instances[instance_id].state = "RELEASED"

        ids = {}
        for allocation in plan.allocations:
            instance = warm.get(allocation.node_id)
            if instance is None:
                identifier = f"sim-{self._next_id:06d}"
                self._next_id += 1
                instance = InstanceRecord(identifier, allocation.cid,
                                          dict(match.bindings[allocation.node_id].parameters),
                                          allocation.demand.memory, "ACTIVE", task.task_id)
                self.instances[identifier] = instance
            else:
                instance.state = "ACTIVE"
                instance.task_id = task.task_id
            ids[allocation.node_id] = instance.instance_id
        decision = ScheduleDecision(task.task_id, plan, ids, reused, evicted)
        self.active[task.task_id] = decision
        return decision

    @synchronized
    def complete(self, task_id: str) -> None:
        if task_id not in self.active:
            raise SchedulingError(f"task {task_id} is not active")
        decision = self.active.pop(task_id)
        for instance_id in decision.instance_ids.values():
            instance = self.instances[instance_id]
            instance.state = "WARM"
            instance.task_id = None

    @synchronized
    def fail(self, task_id: str) -> None:
        if task_id not in self.active:
            raise SchedulingError(f"task {task_id} is not active")
        decision = self.active.pop(task_id)
        for instance_id in decision.instance_ids.values():
            instance = self.instances[instance_id]
            instance.state = "RELEASED"
            instance.task_id = None

    @synchronized
    def evict_warm(self, instance_id: str) -> None:
        instance = self.instances.get(instance_id)
        if instance is None or instance.state != "WARM":
            raise SchedulingError(f"instance {instance_id} is not warm")
        instance.state = "RELEASED"

    @synchronized
    def snapshot(self) -> dict:
        return {"capacity": asdict(self.capacity),
                "used": asdict(self._used()),
                "active_tasks": list(self.active),
                "instances": [instance.to_dict() for instance in self.instances.values()]}
