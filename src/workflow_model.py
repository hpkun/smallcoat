from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from .entities import TaskInstance
from .task_model import UniformRange


WorkflowPattern = Literal["chain", "fork_join", "diamond", "random_dag"]


@dataclass(frozen=True)
class WorkflowModelConfig:
    """Synthetic workflow generation settings for dependency-aware experiments."""

    arrival_rate_workflows_per_s: float = 5.0
    task_count: UniformRange = UniformRange(3, 6)
    patterns: tuple[WorkflowPattern, ...] = ("chain", "fork_join", "diamond", "random_dag")
    random_edge_probability: float = 0.35
    deadline_scale: float = 1.5


@dataclass(frozen=True)
class WorkflowTaskSpec:
    task_instance: TaskInstance
    predecessor_task_ids: tuple[str, ...] = ()
    successor_task_ids: tuple[str, ...] = ()


@dataclass
class WorkflowInstance:
    workflow_id: str
    owner_ingress_uav_id: str
    owner_ch_id: str | None
    arrival_time_s: float
    deadline_s: float
    task_specs: dict[str, WorkflowTaskSpec]
    owner_agent_id: str | None = None
    completed_task_ids: set[str] = field(default_factory=set)
    failed_task_ids: set[str] = field(default_factory=set)
    released_task_ids: set[str] = field(default_factory=set)
    completion_time_s: float | None = None

    @property
    def task_count(self) -> int:
        return len(self.task_specs)

    @property
    def completed(self) -> bool:
        return len(self.completed_task_ids) == self.task_count

    @property
    def failed(self) -> bool:
        return bool(self.failed_task_ids)

    def ready_task_ids(self) -> list[str]:
        ready: list[str] = []
        for task_id, spec in self.task_specs.items():
            if task_id in self.released_task_ids:
                continue
            if task_id in self.completed_task_ids or task_id in self.failed_task_ids:
                continue
            if all(pred in self.completed_task_ids for pred in spec.predecessor_task_ids):
                ready.append(task_id)
        return ready
