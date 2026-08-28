from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entities import TaskInstance
from .workflow_model import WorkflowInstance


WORKFLOW_GAT_EMBEDDING_DIM = 8


@dataclass(frozen=True)
class WorkflowGraphEncoderConfig:
    """Lightweight CH-side graph attention encoder settings."""

    embedding_dim: int = WORKFLOW_GAT_EMBEDDING_DIM
    attention_temperature: float = 0.5


class WorkflowGraphEncoder:
    """
    Deterministic graph-attention-style encoder for synthetic workflows.

    This module runs at the CH decision side: UAVs first collect ready tasks,
    then the decision UAV encodes the visible workflow DAG before building the
    actor observation. It is intentionally deterministic in this first version,
    so it can be used without changing the MADDPG optimization loop.
    """

    def __init__(self, config: WorkflowGraphEncoderConfig | None = None) -> None:
        self.config = config or WorkflowGraphEncoderConfig()
        if self.config.embedding_dim != WORKFLOW_GAT_EMBEDDING_DIM:
            raise ValueError(
                "Only WORKFLOW_GAT_EMBEDDING_DIM embeddings are supported in this encoder."
            )

    def encode_for_decision_uav(
        self,
        *,
        active_workflows: dict[str, WorkflowInstance],
        ready_tasks: list[TaskInstance],
        member_uav_ids: set[str],
        current_time_s: float,
    ) -> dict[str, np.ndarray]:
        embeddings: dict[str, np.ndarray] = {}
        for task_instance in ready_tasks:
            if task_instance.ingress_uav_id not in member_uav_ids:
                continue
            embeddings[task_instance.task_id] = self.encode_task(
                active_workflows=active_workflows,
                task_instance=task_instance,
                current_time_s=current_time_s,
            )
        return embeddings

    def encode_task(
        self,
        *,
        active_workflows: dict[str, WorkflowInstance],
        task_instance: TaskInstance,
        current_time_s: float,
    ) -> np.ndarray:
        if task_instance.workflow_id is None:
            return np.zeros(WORKFLOW_GAT_EMBEDDING_DIM, dtype=np.float32)

        workflow = active_workflows.get(task_instance.workflow_id)
        if workflow is None:
            return self._fallback_embedding(task_instance)

        current_feature = self._task_feature(task_instance)
        predecessor_features = [
            self._task_feature(workflow.task_specs[pred_id].task_instance)
            for pred_id in task_instance.predecessor_task_ids
            if pred_id in workflow.task_specs
        ]
        successor_features = [
            self._task_feature(workflow.task_specs[succ_id].task_instance)
            for succ_id in task_instance.successor_task_ids
            if succ_id in workflow.task_specs
        ]
        pred_context = self._attention_pool(current_feature, predecessor_features)
        succ_context = self._attention_pool(current_feature, successor_features)

        task_count = max(workflow.task_count, 1)
        progress = len(workflow.completed_task_ids) / task_count
        released_ratio = len(workflow.released_task_ids) / task_count
        remaining_ratio = max(task_count - len(workflow.completed_task_ids), 0) / task_count
        workflow_window_s = max(workflow.deadline_s - workflow.arrival_time_s, 1e-6)
        elapsed_ratio = max(current_time_s - workflow.arrival_time_s, 0.0) / workflow_window_s
        slack_ratio = (workflow.deadline_s - current_time_s) / workflow_window_s

        embedding = np.array(
            [
                current_feature[0],
                current_feature[1],
                current_feature[2],
                pred_context[0],
                succ_context[0],
                progress,
                remaining_ratio,
                min(max(slack_ratio, -1.0), 1.0),
            ],
            dtype=np.float32,
        )
        embedding[5] = float(min(max(embedding[5], 0.0), 1.0))
        embedding[6] = float(min(max(embedding[6], 0.0), 1.0))
        embedding[7] = float(min(max(embedding[7], -1.0), 1.0))
        return embedding

    def _fallback_embedding(self, task_instance: TaskInstance) -> np.ndarray:
        feature = self._task_feature(task_instance)
        return np.array(
            [
                feature[0],
                feature[1],
                feature[2],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _task_feature(task_instance: TaskInstance) -> np.ndarray:
        task = task_instance.task
        return np.array(
            [
                float(task.input_size_bits / 1e8),
                float(task.total_compute_cycles / 1e10),
                float(task.tolerable_latency_s),
                float(len(task_instance.predecessor_task_ids) / 10.0),
                float(len(task_instance.successor_task_ids) / 10.0),
            ],
            dtype=np.float32,
        )

    def _attention_pool(
        self,
        query_feature: np.ndarray,
        neighbor_features: list[np.ndarray],
    ) -> np.ndarray:
        if not neighbor_features:
            return np.zeros_like(query_feature)

        neighbors = np.stack(neighbor_features, axis=0)
        query = query_feature / max(float(np.linalg.norm(query_feature)), 1e-6)
        normalized_neighbors = neighbors / np.maximum(
            np.linalg.norm(neighbors, axis=1, keepdims=True),
            1e-6,
        )
        scores = normalized_neighbors @ query
        temperature = max(self.config.attention_temperature, 1e-6)
        weights = self._softmax(scores / temperature)
        return (weights[:, None] * neighbors).sum(axis=0)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / max(float(np.sum(exp_scores)), 1e-6)

