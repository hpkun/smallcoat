"""Task use cases; HTTP details stay in the API layer."""

from collections.abc import Mapping

from task_executor import DAGExecutor
from matching import match_task
from task_repr import (represent_external_task, represent_external_tasks,
                       represent_task)
from components.runtime import DataPacket
from scheduler import TaskScheduler

from ..repositories import JsonDocumentRepository


class TaskService:
    def __init__(self, repository: JsonDocumentRepository,
                 scheduler: TaskScheduler | None = None):
        self.repository = repository
        self.scheduler = scheduler or TaskScheduler()

    def list_names(self) -> list[str]:
        return self.repository.list_names()

    def get(self, name: str) -> dict:
        return self.repository.get(name)

    def validate(self, document: dict) -> dict:
        task = represent_task(document)
        return {"task_id": task.task_id, "node_count": len(task.G.nodes),
                "edge_count": len(task.G.edges)}

    def represent_external(self, request: dict | list) -> dict:
        if isinstance(request, dict) and "tasks" not in request:
            return represent_external_task(request)
        return represent_external_tasks(request)

    def match(self, document: dict) -> dict:
        task = represent_task(document)
        return match_task(task).to_dict()

    def plan(self, document: dict) -> dict:
        task = represent_task(document)
        match = match_task(task)
        if match.status != "MATCHED":
            return {"task_id": task.task_id, "status": "UNMATCHED",
                    "match": match.to_dict()}
        plan, reused, evicted = self.scheduler.preview(task, match)
        return {**plan.to_dict(), "reused": list(reused), "evicted": list(evicted)}

    def simulate_schedule(self, document: dict) -> dict:
        task = represent_task(document)
        return self.scheduler.submit(task, match_task(task)).to_dict()

    def complete_simulation(self, task_id: str) -> dict:
        self.scheduler.complete(task_id)
        return self.scheduler.snapshot()

    def simulation_state(self) -> dict:
        return self.scheduler.snapshot()

    def execute(self, document: dict, sources: Mapping[str, DataPacket]):
        """Reusable application entry point for a future packet transport API."""
        task = represent_task(document)
        result = match_task(task)
        return DAGExecutor.from_match(task, result).execute(sources)

    def execute_with_schedule(self, document: dict,
                              sources: Mapping[str, DataPacket]):
        """Run the CPU DAG under simulated capacity and lifecycle control."""
        task = represent_task(document)
        match = match_task(task)
        decision = self.scheduler.submit(task, match)
        try:
            result = DAGExecutor.from_match(task, match).execute(sources)
        except Exception:
            self.scheduler.fail(task.task_id)
            raise
        self.scheduler.complete(task.task_id)
        return decision, result
