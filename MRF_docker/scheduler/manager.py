"""Capacity-aware instance lifecycle simulation."""

from dataclasses import asdict, dataclass
from functools import wraps
from threading import RLock

from matching.matcher import StaticMatchResult
from resources.hardware import HardwareProfile
from resources.planner import ResourcePlan, ResourcePlanner, ResourceUsage
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


class TaskScheduler:
    """Model active and warm instances without controlling Docker containers.

    Active tasks reserve their plan's CPU, memory and bandwidth. Warm instances
    retain only normalized memory until reused or evicted. Calls are serialized
    with a lock. This remains an in-memory, single-process scheduler.
    """

    def __init__(self, capacity: HardwareProfile | None = None,
                 planner: ResourcePlanner | None = None):
        self.capacity = capacity or HardwareProfile()
        self.planner = planner or ResourcePlanner()
        self.instances: dict[str, InstanceRecord] = {}
        self.active: dict[str, ScheduleDecision] = {}
        self._next_id = 1
        self._lock = RLock()

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
