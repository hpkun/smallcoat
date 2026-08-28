from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from .entities import TaskInstance
from .entities import UAV
from .task_model import TaskModelConfig
from .task_model import sample_num_arrivals
from .task_model import sample_task
from .workflow_model import WorkflowInstance
from .workflow_model import WorkflowModelConfig
from .workflow_model import WorkflowPattern
from .workflow_model import WorkflowTaskSpec


@dataclass
class SyntheticWorkflowGenerator:
    """Generate small DAG workflows while reusing the original task model."""

    task_model_config: TaskModelConfig
    workflow_model_config: WorkflowModelConfig
    next_workflow_index: int = 0
    next_task_index: int = 0

    def generate_workflows(
        self,
        *,
        uavs: list[UAV],
        slot_length_s: float,
        current_time_s: float,
        rng: np.random.Generator,
        delay_sensitivity_lambda: float | None = None,
    ) -> list[WorkflowInstance]:
        workflows: list[WorkflowInstance] = []
        if not uavs:
            return workflows

        count_config = TaskModelConfig(
            arrival_rate_tasks_per_s=self.workflow_model_config.arrival_rate_workflows_per_s
        )
        count = sample_num_arrivals(count_config, slot_length_s, rng)
        serviceable_uavs = [uav for uav in uavs if uav.can_serve]
        association_pool = serviceable_uavs or uavs
        for _ in range(count):
            owner_ingress_uav = association_pool[
                int(rng.integers(0, len(association_pool)))
            ]
            workflows.append(
                self._generate_one_workflow(
                    uavs=[owner_ingress_uav],
                    current_time_s=current_time_s,
                    rng=rng,
                    delay_sensitivity_lambda=delay_sensitivity_lambda,
                )
            )
        return workflows

    def _generate_one_workflow(
        self,
        *,
        uavs: list[UAV],
        current_time_s: float,
        rng: np.random.Generator,
        delay_sensitivity_lambda: float | None,
    ) -> WorkflowInstance:
        workflow_id = f"workflow-{self.next_workflow_index}"
        self.next_workflow_index += 1

        low = int(round(self.workflow_model_config.task_count.low))
        high = int(round(self.workflow_model_config.task_count.high))
        task_count = int(rng.integers(max(1, low), max(1, high) + 1))
        owner_ingress_uav = uavs[int(rng.integers(0, len(uavs)))]
        pattern = self._sample_pattern(task_count, rng)
        edges = self._build_edges(task_count, pattern, rng)
        predecessor_ids_by_index: dict[int, list[str]] = {idx: [] for idx in range(task_count)}
        successor_ids_by_index: dict[int, list[str]] = {idx: [] for idx in range(task_count)}
        task_ids = [f"{workflow_id}-task-{idx}" for idx in range(task_count)]
        for src_idx, dst_idx in edges:
            successor_ids_by_index[src_idx].append(task_ids[dst_idx])
            predecessor_ids_by_index[dst_idx].append(task_ids[src_idx])

        sampled_tasks = [
            sample_task(
                self.task_model_config,
                rng,
                delay_sensitivity_lambda=delay_sensitivity_lambda,
            )
            for _ in range(task_count)
        ]
        workflow_deadline_s = current_time_s + (
            sum(task.tolerable_latency_s for task in sampled_tasks)
            * self.workflow_model_config.deadline_scale
        )

        task_specs: dict[str, WorkflowTaskSpec] = {}
        for idx, task in enumerate(sampled_tasks):
            task_id = task_ids[idx]
            self.next_task_index += 1
            predecessor_task_ids = tuple(predecessor_ids_by_index[idx])
            successor_task_ids = tuple(successor_ids_by_index[idx])
            task_with_dependency_hint = replace(
                task,
                forward_count=len(successor_task_ids),
            )
            task_specs[task_id] = WorkflowTaskSpec(
                task_instance=TaskInstance(
                    task_id=task_id,
                    ingress_uav_id=owner_ingress_uav.node_id,
                    created_at_s=current_time_s,
                    task=task_with_dependency_hint,
                    workflow_id=workflow_id,
                    workflow_arrival_time_s=current_time_s,
                    workflow_deadline_s=workflow_deadline_s,
                    workflow_task_count=task_count,
                    workflow_step_index=idx,
                    predecessor_task_ids=predecessor_task_ids,
                    successor_task_ids=successor_task_ids,
                ),
                predecessor_task_ids=predecessor_task_ids,
                successor_task_ids=successor_task_ids,
            )

        return WorkflowInstance(
            workflow_id=workflow_id,
            owner_ingress_uav_id=owner_ingress_uav.node_id,
            owner_ch_id=None,
            arrival_time_s=current_time_s,
            deadline_s=workflow_deadline_s,
            task_specs=task_specs,
        )

    def _sample_pattern(
        self,
        task_count: int,
        rng: np.random.Generator,
    ) -> WorkflowPattern:
        patterns = [
            pattern
            for pattern in self.workflow_model_config.patterns
            if pattern == "chain" or task_count >= 4
        ]
        if not patterns:
            return "chain"
        return patterns[int(rng.integers(0, len(patterns)))]

    def _build_edges(
        self,
        task_count: int,
        pattern: WorkflowPattern,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        if task_count <= 1:
            return []
        if pattern == "chain":
            return [(idx, idx + 1) for idx in range(task_count - 1)]
        if pattern == "fork_join":
            return self._build_fork_join_edges(task_count)
        if pattern == "diamond":
            return self._build_diamond_edges(task_count)
        return self._build_random_dag_edges(task_count, rng)

    @staticmethod
    def _build_fork_join_edges(task_count: int) -> list[tuple[int, int]]:
        sink = task_count - 1
        edges = [(0, idx) for idx in range(1, sink)]
        edges.extend((idx, sink) for idx in range(1, sink))
        return edges

    @staticmethod
    def _build_diamond_edges(task_count: int) -> list[tuple[int, int]]:
        sink = task_count - 1
        middle = max(2, task_count // 2)
        edges = [(0, idx) for idx in range(1, middle)]
        edges.extend((idx, sink) for idx in range(middle, sink))
        edges.extend((left, right) for left in range(1, middle) for right in range(middle, sink))
        return edges

    def _build_random_dag_edges(
        self,
        task_count: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        edges: set[tuple[int, int]] = {(idx, idx + 1) for idx in range(task_count - 1)}
        edge_probability = min(max(self.workflow_model_config.random_edge_probability, 0.0), 1.0)
        for src_idx in range(task_count):
            for dst_idx in range(src_idx + 2, task_count):
                if rng.random() < edge_probability:
                    edges.add((src_idx, dst_idx))
        return sorted(edges)
