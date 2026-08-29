from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from .action_space import ActionSpec
from .action_space import MixedActionCodec
from .action_space import MultiTaskOffloadingAction
from .action_space import build_action_spec
from .entities import ExecutionRecord
from .entities import TaskInstance
from .entities import UAV
from .environment import OffloadingAction
from .objective import compute_equation_8_objective
from .observation_builder import ObservationBuilder
from .reward import RewardConfig
from .reward import SharedRewardCalculator
from .workflow_generator import SyntheticWorkflowGenerator
from .workflow_encoder import WorkflowGraphEncoder
from .workflow_manager import WorkflowManager
from .workflow_manager import WorkflowStepSummary


MAX_PEER_UAV_CANDIDATE_SLOTS = 3
MAX_BS_CANDIDATE_SLOTS = 6
MAX_TARGET_SLOTS = 1 + MAX_PEER_UAV_CANDIDATE_SLOTS + MAX_BS_CANDIDATE_SLOTS + 1


@dataclass(frozen=True)
class AgentDecisionContext:
    agent_id: str
    decision_uav_id: str
    task_slots: list[TaskInstance | None]
    ingress_uav_slots: list[UAV]
    slot_target_node_ids: list[list[str]]
    action_spec: ActionSpec


@dataclass(frozen=True)
class DecisionAgentBinding:
    agent_id: str
    cluster_id: int | None
    decision_uav_id: str


class CMADDPGEnv:
    def __init__(
        self,
        base_env,
        observation_builder: ObservationBuilder,
        reward_calculator: SharedRewardCalculator | None = None,
        task_mode: str = "independent",
        workflow_generator: SyntheticWorkflowGenerator | None = None,
        workflow_encoder: WorkflowGraphEncoder | None = None,
    ) -> None:
        if task_mode not in {"independent", "workflow"}:
            raise ValueError("task_mode must be either 'independent' or 'workflow'.")
        if task_mode == "workflow" and workflow_generator is None:
            raise ValueError("workflow_generator is required when task_mode='workflow'.")

        self.base_env = base_env
        self.observation_builder = observation_builder
        self.reward_calculator = reward_calculator or SharedRewardCalculator(RewardConfig())
        self.task_mode = task_mode
        self.workflow_generator = workflow_generator
        self.workflow_encoder = workflow_encoder or WorkflowGraphEncoder()
        self.workflow_manager = WorkflowManager()
        self.last_workflow_summary = WorkflowStepSummary()
        self.pending_tasks: list[TaskInstance] = []
        self.pending_contexts: dict[str, AgentDecisionContext] = {}
        self.current_time_s = 0.0
        self.episode_generated_task_count = 0

    def _generate_next_tasks(self) -> list[TaskInstance]:
        if self.task_mode == "independent":
            self.last_workflow_summary = WorkflowStepSummary()
            tasks = self.base_env.task_generator.generate_tasks(
                uavs=self.base_env.uavs,
                slot_length_s=self.base_env.simulation_config.slot_length_s,
                current_time_s=self.current_time_s,
                rng=self.base_env.rng,
                delay_sensitivity_lambda=self._default_lambda(),
            )
            self.episode_generated_task_count += len(tasks)
            return tasks

        if self.workflow_generator is None:
            raise ValueError("workflow_generator is required in workflow mode.")
        available_uav_ids = {
            uav.node_id for uav in self.base_env.uavs if uav.can_serve
        }
        battery_failed_workflows = self.workflow_manager.drop_workflows_with_unavailable_owners(
            available_uav_ids
        )
        new_workflows = self.workflow_generator.generate_workflows(
            uavs=self.base_env.uavs,
            slot_length_s=self.base_env.simulation_config.slot_length_s,
            current_time_s=self.current_time_s,
            rng=self.base_env.rng,
            delay_sensitivity_lambda=self._default_lambda(),
        )
        self._assign_owner_ch_to_workflows(new_workflows)
        self.workflow_manager.add_workflows(new_workflows)
        self._refresh_active_workflow_owner_ch()
        ready_tasks, summary = self.workflow_manager.release_ready_tasks(self.current_time_s)
        self.episode_generated_task_count += len(ready_tasks)
        summary.failed_workflows += battery_failed_workflows
        self.last_workflow_summary = summary
        return ready_tasks

    def _assign_owner_ch_to_workflows(self, workflows) -> None:
        for workflow in workflows:
            ingress_uav = self.base_env.get_uav_by_id(workflow.owner_ingress_uav_id)
            resolved = self._resolve_agent_binding(ingress_uav)
            if resolved is None:
                continue
            binding, owner_ch = resolved
            workflow.owner_ch_id = owner_ch.node_id
            workflow.owner_agent_id = binding.agent_id
            for task_id, spec in workflow.task_specs.items():
                workflow.task_specs[task_id] = replace(
                    spec,
                    task_instance=replace(
                        spec.task_instance,
                        owner_ch_id=owner_ch.node_id,
                        owner_agent_id=binding.agent_id,
                    ),
                )

    def _refresh_active_workflow_owner_ch(self) -> None:
        for workflow in self.workflow_manager.active_workflows.values():
            ingress_uav = self.base_env.get_uav_by_id(workflow.owner_ingress_uav_id)
            resolved = self._resolve_agent_binding(ingress_uav)
            if resolved is None:
                continue
            binding, owner_ch = resolved
            if (
                workflow.owner_ch_id == owner_ch.node_id
                and workflow.owner_agent_id == binding.agent_id
            ):
                continue
            workflow.owner_ch_id = owner_ch.node_id
            workflow.owner_agent_id = binding.agent_id
            for task_id, spec in workflow.task_specs.items():
                workflow.task_specs[task_id] = replace(
                    spec,
                    task_instance=replace(
                        spec.task_instance,
                        owner_ch_id=owner_ch.node_id,
                        owner_agent_id=binding.agent_id,
                    ),
                )

    def _collect_member_uavs(
        self, binding: DecisionAgentBinding
    ) -> list[UAV]:
        manager = self.base_env.clustering_manager
        if manager is not None and binding.cluster_id is not None:
            cluster = manager.cluster_infos.get(binding.cluster_id)
            if cluster is not None:
                member_ids = set(cluster.member_uav_ids)
                return [uav for uav in self.base_env.uavs if uav.node_id in member_ids]
        return [self.base_env.get_uav_by_id(binding.decision_uav_id)]

    def _decision_uavs(self) -> list[UAV]:
        return [
            self.base_env.get_uav_by_id(binding.decision_uav_id)
            for binding in self._decision_agents()
        ]

    def _decision_agents(self) -> list[DecisionAgentBinding]:
        manager = self.base_env.clustering_manager
        if manager is None:
            return [
                DecisionAgentBinding(
                    agent_id=f"isolated-agent-{uav.node_id}",
                    cluster_id=None,
                    decision_uav_id=uav.node_id,
                )
                for uav in sorted(self.base_env.uavs, key=lambda item: item.node_id)
                if uav.can_serve
            ]

        uavs_by_id = {uav.node_id: uav for uav in self.base_env.uavs}
        bindings: list[DecisionAgentBinding] = []
        clustered_member_ids: set[str] = set()
        for cluster in sorted(
            manager.cluster_infos.values(), key=lambda item: item.logical_agent_id
        ):
            clustered_member_ids.update(cluster.member_uav_ids)
            serviceable_members = [
                uavs_by_id[uav_id]
                for uav_id in cluster.member_uav_ids
                if uav_id in uavs_by_id and uavs_by_id[uav_id].can_serve
            ]
            if not serviceable_members:
                continue
            current_head = uavs_by_id.get(cluster.head_uav_id)
            decision_uav = (
                current_head
                if current_head is not None and current_head.can_serve
                else min(
                    serviceable_members,
                    key=lambda uav: (
                        uav.position.distance_to(cluster.centroid),
                        uav.node_id,
                    ),
                )
            )
            bindings.append(
                DecisionAgentBinding(
                    agent_id=cluster.logical_agent_id,
                    cluster_id=cluster.cluster_id,
                    decision_uav_id=decision_uav.node_id,
                )
            )
        bindings.extend(
            DecisionAgentBinding(
                agent_id=f"isolated-agent-{uav.node_id}",
                cluster_id=None,
                decision_uav_id=uav.node_id,
            )
            for uav in sorted(self.base_env.uavs, key=lambda item: item.node_id)
            if uav.can_serve and uav.node_id not in clustered_member_ids
        )
        return bindings

    def _logical_agent_id(self, decision_uav: UAV) -> str:
        manager = self.base_env.clustering_manager
        if manager is not None:
            logical_agent_id = manager.get_logical_agent_id(decision_uav.node_id)
            if logical_agent_id is not None:
                return logical_agent_id
        return f"isolated-agent-{decision_uav.node_id}"

    def _resolve_agent_binding(
        self, ingress_uav: UAV
    ) -> tuple[DecisionAgentBinding, UAV] | None:
        manager = self.base_env.clustering_manager
        agent_id = (
            manager.get_logical_agent_id(ingress_uav.node_id)
            if manager is not None
            else None
        )
        if agent_id is None:
            agent_id = f"isolated-agent-{ingress_uav.node_id}"
        binding = next(
            (
                candidate
                for candidate in self._decision_agents()
                if candidate.agent_id == agent_id
            ),
            None,
        )
        if binding is None:
            return None
        return binding, self.base_env.get_uav_by_id(binding.decision_uav_id)

    def _tasks_grouped_by_agent(self) -> dict[str, list[tuple[TaskInstance, UAV]]]:
        grouped: dict[str, list[tuple[TaskInstance, UAV]]] = {}
        for task_instance in self.pending_tasks:
            ingress_uav = self.base_env.get_uav_by_id(task_instance.ingress_uav_id)
            resolved = self._resolve_agent_binding(ingress_uav)
            if resolved is None:
                continue
            binding, decision_uav = resolved
            task_instance = replace(
                task_instance,
                owner_agent_id=binding.agent_id,
                owner_ch_id=decision_uav.node_id,
            )
            grouped.setdefault(binding.agent_id, []).append(
                (task_instance, ingress_uav)
            )
        return grouped

    def _build_task_block(
        self,
        decision_uav: UAV,
        member_uavs: list[UAV],
        task_slots: list[TaskInstance | None],
        ingress_uav_slots: list[UAV],
        slot_target_nodes: list[list],
    ) -> np.ndarray:
        task_blocks: list[np.ndarray] = []
        for task_instance, ingress_uav, target_node_order in zip(
            task_slots,
            ingress_uav_slots,
            slot_target_nodes,
        ):
            candidate_nodes = [node for node in target_node_order if node is not None]
            task_blocks.append(
                self.observation_builder.build_agent_state(
                    decision_uav=decision_uav,
                    ingress_uav=ingress_uav,
                    member_uavs=member_uavs,
                    base_stations=self.base_env.base_stations,
                    leo_satellite=self.base_env.leo_satellite,
                    task_instance=task_instance,
                    candidate_nodes=candidate_nodes,
                    target_node_order=target_node_order,
                    current_time_s=self.current_time_s,
                    cluster_radius_m=(
                        self.base_env.clustering_manager.config.communication_radius_m
                        if self.base_env.clustering_manager is not None
                        else None
                    ),
                )
            )
        return np.concatenate(task_blocks, axis=0)

    def _rank_reachable_base_stations(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
    ) -> list:
        """按预计传输、排队和计算总时延选择 Top-K 可达 BS。"""

        reachable = [
            node
            for node in self.base_env.iter_compute_targets(decision_uav, ingress_uav)
            if node.node_type == "bs"
        ]

        def estimated_finish_delay(bs) -> float:
            tx_s, prop_s = self.base_env.communication_model.total_link_delay_s(
                data_size_bits=task_instance.task.input_size_bits,
                sender=ingress_uav.position,
                receiver=bs.position,
                profile=self.base_env.network_profiles.uav_to_bs,
            )
            queue_wait_s = bs.queue_snapshot(self.current_time_s).expected_total_wait_s
            compute_s = bs.estimate_compute_delay(task_instance.task)
            return float(tx_s + prop_s + queue_wait_s + compute_s)

        return sorted(reachable, key=estimated_finish_delay)[:MAX_BS_CANDIDATE_SLOTS]

    def _rank_reachable_uavs(
        self,
        task_instance: TaskInstance,
        ingress_uav: UAV,
        decision_uav: UAV,
    ) -> list[UAV]:
        """Select Top-3 reachable peers by communication, queue, and compute delay."""

        reachable = [
            node
            for node in self.base_env.iter_compute_targets(decision_uav, ingress_uav)
            if isinstance(node, UAV) and node.node_id != ingress_uav.node_id and node.can_serve
        ]

        def estimated_finish_delay(peer_uav: UAV) -> float:
            tx_s, prop_s = self.base_env.communication_model.total_link_delay_s(
                data_size_bits=task_instance.task.input_size_bits,
                sender=ingress_uav.position,
                receiver=peer_uav.position,
                profile=self.base_env.network_profiles.peer_uav_profile(),
            )
            queue_wait_s = peer_uav.queue_snapshot(self.current_time_s).expected_total_wait_s
            compute_s = peer_uav.estimate_compute_delay(task_instance.task)
            return float(tx_s + prop_s + queue_wait_s + compute_s)

        return sorted(reachable, key=estimated_finish_delay)[:MAX_PEER_UAV_CANDIDATE_SLOTS]

    def _build_slot_target_nodes(
        self,
        decision_uav: UAV,
        task_slots: list[TaskInstance | None],
        ingress_uav_slots: list[UAV],
    ) -> list[list]:
        """Build fixed semantic slots: ingress + 3 peer UAVs + 6 BSs + LEO."""

        slot_nodes: list[list] = []
        for task_instance, ingress_uav in zip(task_slots, ingress_uav_slots):
            if task_instance is None or not ingress_uav.can_serve:
                slot_nodes.append([None] * MAX_TARGET_SLOTS)
                continue
            peer_uavs = self._rank_reachable_uavs(
                task_instance,
                ingress_uav,
                decision_uav,
            )
            base_stations = self._rank_reachable_base_stations(
                task_instance,
                ingress_uav,
                decision_uav,
            )
            padded_bs = base_stations + [None] * (
                MAX_BS_CANDIDATE_SLOTS - len(base_stations)
            )
            padded_peers = peer_uavs + [None] * (
                MAX_PEER_UAV_CANDIDATE_SLOTS - len(peer_uavs)
            )
            slot_nodes.append(
                [ingress_uav, *padded_peers, *padded_bs, self.base_env.leo_satellite]
            )
        return slot_nodes

    def _build_action_spec(
        self,
        slot_target_nodes: list[list],
    ) -> tuple[ActionSpec, list[list[str]]]:
        slot_target_node_ids = [
            [node.node_id if node is not None else "" for node in nodes]
            for nodes in slot_target_nodes
        ]
        slot_target_masks = [
            [node is not None for node in nodes]
            for nodes in slot_target_nodes
        ]
        target_slot_ids = [f"target-slot-{index}" for index in range(MAX_TARGET_SLOTS)]
        return (
            build_action_spec(
                target_slot_ids,
                slot_target_masks,
                slot_target_node_ids=slot_target_node_ids,
            ),
            slot_target_node_ids,
        )

    def _attach_workflow_embeddings(
        self,
        *,
        task_pairs: list[tuple[TaskInstance, UAV]],
        member_uavs: list[UAV],
    ) -> list[tuple[TaskInstance, UAV]]:
        if self.task_mode != "workflow" or not task_pairs:
            return task_pairs

        ready_tasks = [task_instance for task_instance, _ in task_pairs]
        member_uav_ids = {uav.node_id for uav in member_uavs}
        embeddings = self.workflow_encoder.encode_for_decision_uav(
            active_workflows=self.workflow_manager.active_workflows,
            ready_tasks=ready_tasks,
            member_uav_ids=member_uav_ids,
            current_time_s=self.current_time_s,
        )
        enriched_pairs: list[tuple[TaskInstance, UAV]] = []
        for task_instance, ingress_uav in task_pairs:
            embedding = embeddings.get(task_instance.task_id)
            if embedding is None:
                enriched_pairs.append((task_instance, ingress_uav))
                continue
            enriched_pairs.append(
                (
                    replace(
                        task_instance,
                        workflow_embedding=tuple(float(value) for value in embedding),
                    ),
                    ingress_uav,
                )
            )
        return enriched_pairs

    def _build_contexts_and_states(self) -> tuple[dict[str, np.ndarray], dict[str, ActionSpec]]:
        states: dict[str, np.ndarray] = {}
        action_specs: dict[str, ActionSpec] = {}
        self.pending_contexts = {}
        grouped_tasks = self._tasks_grouped_by_agent()

        for binding in self._decision_agents():
            agent_id = binding.agent_id
            decision_uav = self.base_env.get_uav_by_id(binding.decision_uav_id)
            member_uavs = self._collect_member_uavs(binding)
            selected_pairs = grouped_tasks.get(agent_id, [])
            if not selected_pairs:
                continue
            selected_pairs = self._attach_workflow_embeddings(
                task_pairs=selected_pairs,
                member_uavs=member_uavs,
            )
            task_slots = [pair[0] for pair in selected_pairs]
            ingress_uav_slots = [pair[1] for pair in selected_pairs]

            slot_target_nodes = self._build_slot_target_nodes(
                decision_uav,
                task_slots,
                ingress_uav_slots,
            )
            action_spec, slot_target_node_ids = self._build_action_spec(slot_target_nodes)
            context = AgentDecisionContext(
                agent_id=agent_id,
                decision_uav_id=decision_uav.node_id,
                task_slots=task_slots,
                ingress_uav_slots=ingress_uav_slots,
                slot_target_node_ids=slot_target_node_ids,
                action_spec=action_spec,
            )
            self.pending_contexts[agent_id] = context
            states[agent_id] = self._build_task_block(
                decision_uav=decision_uav,
                member_uavs=member_uavs,
                task_slots=task_slots,
                ingress_uav_slots=ingress_uav_slots,
                slot_target_nodes=slot_target_nodes,
            )
            action_specs[agent_id] = action_spec
        return states, action_specs

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, ActionSpec]]:
        self.current_time_s = 0.0
        self.episode_generated_task_count = 0
        self.base_env.current_slot_index = 0
        self.base_env.reset_batteries()
        for node in (
            list(self.base_env.uavs)
            + list(self.base_env.base_stations)
            + [self.base_env.leo_satellite]
        ):
            node.queue_manager.reset()
        self.workflow_manager.reset()
        self.pending_tasks = self._generate_next_tasks()
        return self._build_contexts_and_states()

    def _default_lambda(self) -> float | None:
        return self.base_env.task_generator.task_model_config.delay_sensitivity_lambda

    def step(
        self,
        actions_by_agent_id: dict[str, MultiTaskOffloadingAction],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        actions_by_task_id: dict[str, OffloadingAction] = {}
        for agent_id, context in self.pending_contexts.items():
            action = actions_by_agent_id.get(agent_id)
            if action is None or len(action.slot_actions) != len(context.task_slots):
                continue

            for task_instance, slot_action, legal_target_ids in zip(
                context.task_slots,
                action.slot_actions,
                context.slot_target_node_ids,
            ):
                if task_instance is None or not legal_target_ids:
                    continue
                legal_target_set = {node_id for node_id in legal_target_ids if node_id}
                replica_target_node_ids = tuple(
                    node_id
                    for node_id in slot_action.replica_target_node_ids
                    if node_id in legal_target_set
                )
                if not replica_target_node_ids:
                    continue
                if not self.base_env.enable_redundancy:
                    replica_target_node_ids = replica_target_node_ids[:1]
                actions_by_task_id[task_instance.task_id] = OffloadingAction(
                    priority_eta=slot_action.priority_eta,
                    replica_count=(
                        slot_action.replica_count
                        if self.base_env.enable_redundancy
                        else 1
                    ),
                    replica_target_node_ids=replica_target_node_ids,
                )

        executable_tasks = [
            task
            for task in self.pending_tasks
            if task.task_id in actions_by_task_id
            and self.base_env.get_uav_by_id(task.ingress_uav_id).can_serve
        ]
        deferred_tasks = [
            task
            for task in self.pending_tasks
            if task.task_id not in actions_by_task_id
            or not self.base_env.get_uav_by_id(task.ingress_uav_id).can_serve
        ]
        records = self.base_env.step(
            slot_length_s=self.base_env.simulation_config.slot_length_s,
            current_time_s=self.current_time_s,
            delay_sensitivity_lambda=self._default_lambda(),
            move_uavs=False,
            actions_by_task_id=actions_by_task_id,
            external_tasks=executable_tasks,
            apply_pre_step_dynamics=False,
        )
        if self.task_mode == "workflow":
            records, workflow_record_summary = self.workflow_manager.apply_records(records)
        else:
            workflow_record_summary = WorkflowStepSummary()

        shared_reward = self.reward_calculator.aggregate(records)
        rewards = {agent_id: float(shared_reward) for agent_id in self.pending_contexts}
        self.current_time_s += self.base_env.simulation_config.slot_length_s
        self.base_env.advance_system_dynamics(self.base_env.simulation_config.slot_length_s)
        self.pending_tasks = deferred_tasks + self._generate_next_tasks()
        next_states, next_action_specs = self._build_contexts_and_states()
        workflow_summary = {
            "active_workflows": self.last_workflow_summary.active_workflows,
            "ready_tasks": self.last_workflow_summary.ready_tasks,
            "completed_workflows": self.last_workflow_summary.completed_workflows,
            "failed_workflows": (
                self.last_workflow_summary.failed_workflows
                + workflow_record_summary.failed_workflows
            ),
            "workflow_sla_violations": self.last_workflow_summary.workflow_sla_violations,
            "pending_task_completions": self.last_workflow_summary.pending_task_completions,
            "avg_completed_workflow_makespan_s": self.last_workflow_summary.avg_completed_workflow_makespan_s,
            "max_completed_workflow_makespan_s": self.last_workflow_summary.max_completed_workflow_makespan_s,
            "sum_completed_workflow_makespan_s": float(
                sum(self.last_workflow_summary.completed_workflow_makespans_s)
            ),
        }
        info = {
            "records": records,
            "action_specs": next_action_specs,
            "shared_reward": shared_reward,
            "equation8_objective": compute_equation_8_objective(records),
            "workflow_summary": workflow_summary,
            "battery_status": self.base_env.battery_status(),
            "generated_task_count": self.episode_generated_task_count,
            "pending_ground_task_count": len(self.pending_tasks),
            "uncollected_task_count": len(deferred_tasks),
            "energy_constraint_multiplier": self.reward_calculator.energy_constraint_multiplier,
            "energy_budget_violation_j": self.reward_calculator.energy_budget_violation_j,
        }
        return next_states, rewards, False, info

    @staticmethod
    def encode_action_for_env(
        action_spec: ActionSpec,
        actor_output: np.ndarray,
    ) -> MultiTaskOffloadingAction:
        codec = MixedActionCodec(action_spec)
        return codec.decode_numpy(actor_output).to_multi_task_action()

    @staticmethod
    def extract_record_metrics(records: list[ExecutionRecord]) -> dict[str, float]:
        if not records:
            return {
                "avg_delay_s": 0.0,
                "completion_rate": 0.0,
                "system_profit": 0.0,
                "avg_requested_replica_count": 0.0,
                "avg_admitted_replica_count": 0.0,
            }
        requested_counts = np.asarray(
            [record.requested_replica_count for record in records],
            dtype=np.int64,
        )
        requested_targets = [
            node_id
            for record in records
            for node_id in record.replica_target_node_ids
        ]
        layer_counts = {
            layer: sum(node_id.startswith(f"{layer}-") for node_id in requested_targets)
            for layer in ("uav", "bs", "leo")
        }
        redundant_target_sets = [
            record.replica_target_node_ids
            for record in records
            if record.requested_replica_count > 1
            and len(record.replica_target_node_ids) > 1
        ]
        same_layer_count = sum(
            len({node_id.split("-", 1)[0] for node_id in target_ids}) == 1
            for target_ids in redundant_target_sets
        )
        completed_count = sum(record.completed_before_deadline for record in records)
        total_requested_replicas = int(requested_counts.sum())
        total_placements = len(requested_targets)
        return {
            "avg_delay_s": float(np.mean([record.total_delay_s for record in records])),
            "completion_rate": float(np.mean([1.0 if record.completed_before_deadline else 0.0 for record in records])),
            "system_profit": float(np.sum([record.realized_profit for record in records])),
            "avg_requested_replica_count": float(np.mean(requested_counts)),
            "replica_count_1_rate": float(np.mean(requested_counts == 1)),
            "replica_count_2_rate": float(np.mean(requested_counts == 2)),
            "replica_count_3_rate": float(np.mean(requested_counts == 3)),
            "avg_admitted_replica_count": float(
                np.mean([record.admitted_replica_count for record in records])
            ),
            "capacity_rejected_replica_rate": float(
                sum(record.capacity_rejected_replica_count for record in records)
                / max(total_requested_replicas, 1)
            ),
            "uav_replica_share": float(layer_counts["uav"] / max(total_placements, 1)),
            "bs_replica_share": float(layer_counts["bs"] / max(total_placements, 1)),
            "leo_replica_share": float(layer_counts["leo"] / max(total_placements, 1)),
            "same_layer_replica_rate": float(
                same_layer_count / max(len(redundant_target_sets), 1)
            ),
            "cross_layer_replica_rate": float(
                (len(redundant_target_sets) - same_layer_count)
                / max(len(redundant_target_sets), 1)
            ),
            "reliable_on_time_completion_rate": float(
                np.mean(
                    [
                        record.completed_before_deadline
                        and record.satisfies_reliability
                        and not record.failed_due_to_reliability
                        for record in records
                    ]
                )
            ),
            "total_energy_j": float(sum(record.total_energy_j for record in records)),
            "energy_per_completed_task": float(
                sum(record.total_energy_j for record in records) / max(completed_count, 1)
            ),
            "cancellation_energy_saved_j": float(
                sum(record.cancellation_energy_saved_j for record in records)
            ),
            "avg_combined_reliability": float(
                np.mean([record.end_to_end_reliability for record in records])
            ),
        }
