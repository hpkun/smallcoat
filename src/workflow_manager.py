from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

from .entities import ExecutionRecord
from .entities import TaskInstance
from .workflow_model import WorkflowInstance


@dataclass
class WorkflowStepSummary:
    active_workflows: int = 0
    ready_tasks: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    workflow_sla_violations: int = 0
    pending_task_completions: int = 0
    completed_workflow_makespans_s: tuple[float, ...] = ()

    @property
    def avg_completed_workflow_makespan_s(self) -> float:
        if not self.completed_workflow_makespans_s:
            return 0.0
        return float(
            sum(self.completed_workflow_makespans_s)
            / len(self.completed_workflow_makespans_s)
        )

    @property
    def max_completed_workflow_makespan_s(self) -> float:
        if not self.completed_workflow_makespans_s:
            return 0.0
        return float(max(self.completed_workflow_makespans_s))


@dataclass
class WorkflowManager:
    """Track active synthetic workflows and expose only ready tasks."""

    active_workflows: dict[str, WorkflowInstance] = field(default_factory=dict)
    pending_completion_records: list[ExecutionRecord] = field(default_factory=list)

    def reset(self) -> None:
        self.active_workflows.clear()
        self.pending_completion_records.clear()

    def add_workflows(self, workflows: list[WorkflowInstance]) -> None:
        for workflow in workflows:
            self.active_workflows[workflow.workflow_id] = workflow

    def drop_workflows_with_unavailable_owners(
        self,
        available_uav_ids: set[str],
    ) -> int:
        dropped_ids = [
            workflow_id
            for workflow_id, workflow in self.active_workflows.items()
            if workflow.owner_ingress_uav_id not in available_uav_ids
        ]
        for workflow_id in dropped_ids:
            self.active_workflows.pop(workflow_id, None)
        self.pending_completion_records = [
            record
            for record in self.pending_completion_records
            if record.workflow_id not in dropped_ids
        ]
        return len(dropped_ids)

    def release_ready_tasks(self, current_time_s: float) -> tuple[list[TaskInstance], WorkflowStepSummary]:
        summary = self.settle_completions(current_time_s)
        ready_tasks: list[TaskInstance] = []
        for workflow in self.active_workflows.values():
            for task_id in workflow.ready_task_ids():
                workflow.released_task_ids.add(task_id)
                ready_tasks.append(
                    replace(
                        workflow.task_specs[task_id].task_instance,
                        created_at_s=current_time_s,
                    )
                )
        summary.active_workflows = len(self.active_workflows)
        summary.ready_tasks = len(ready_tasks)
        summary.pending_task_completions = len(self.pending_completion_records)
        return ready_tasks, summary

    def apply_records(self, records: list[ExecutionRecord]) -> tuple[list[ExecutionRecord], WorkflowStepSummary]:
        failed_workflows: set[str] = set()
        enriched_records: list[ExecutionRecord] = []

        for record in records:
            workflow_id = record.workflow_id
            if workflow_id is None or workflow_id not in self.active_workflows:
                enriched_records.append(record)
                continue

            workflow = self.active_workflows[workflow_id]
            if record.completed_before_deadline:
                self.pending_completion_records.append(record)
            else:
                workflow.failed_task_ids.add(record.task_id)
                failed_workflows.add(workflow_id)

            enriched_records.append(record)

        for workflow_id in failed_workflows:
            self.active_workflows.pop(workflow_id, None)

        return enriched_records, WorkflowStepSummary(
            active_workflows=len(self.active_workflows),
            failed_workflows=len(failed_workflows),
            pending_task_completions=len(self.pending_completion_records),
        )

    def settle_completions(self, current_time_s: float) -> WorkflowStepSummary:
        completed_workflows: set[str] = set()
        sla_violated_workflows: set[str] = set()
        completed_makespans_s: list[float] = []
        remaining_records: list[ExecutionRecord] = []

        for record in self.pending_completion_records:
            if record.finish_time_s > current_time_s:
                remaining_records.append(record)
                continue
            workflow_id = record.workflow_id
            if workflow_id is None or workflow_id not in self.active_workflows:
                continue

            workflow = self.active_workflows[workflow_id]
            workflow.completed_task_ids.add(record.task_id)

            workflow_completed = workflow.completed
            if workflow_completed:
                workflow.completion_time_s = max(
                    workflow.completion_time_s or workflow.arrival_time_s,
                    record.finish_time_s,
                )
                completed_workflows.add(workflow_id)
                if workflow.completion_time_s > workflow.deadline_s:
                    sla_violated_workflows.add(workflow_id)
                completed_makespans_s.append(
                    workflow.completion_time_s - workflow.arrival_time_s
                )

        self.pending_completion_records = remaining_records
        for workflow_id in completed_workflows:
            self.active_workflows.pop(workflow_id, None)

        return WorkflowStepSummary(
            active_workflows=len(self.active_workflows),
            ready_tasks=sum(len(workflow.ready_task_ids()) for workflow in self.active_workflows.values()),
            completed_workflows=len(completed_workflows),
            workflow_sla_violations=len(sla_violated_workflows),
            pending_task_completions=len(self.pending_completion_records),
            completed_workflow_makespans_s=tuple(completed_makespans_s),
        )

    def enrich_completed_workflow_record(self, record: ExecutionRecord) -> ExecutionRecord:
        workflow_id = record.workflow_id
        if workflow_id is None or workflow_id not in self.active_workflows:
            return record
        workflow = self.active_workflows[workflow_id]
        if not workflow.completed:
            return record
        completion_time_s = workflow.completion_time_s or record.finish_time_s
        return replace(
            record,
            workflow_completed=True,
            workflow_completion_delay_s=completion_time_s - workflow.arrival_time_s,
            workflow_sla_violated=completion_time_s > workflow.deadline_s,
        )
