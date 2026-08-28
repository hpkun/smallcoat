from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam

from .action_space import ActionSpec
from .action_space import MultiTaskOffloadingAction
from .action_space import SlotAction
from .networks import VariableTaskActorNetwork
from .networks import VariableTaskValueNetwork
from .observation_builder import OBSERVATION_INPUT_DIM


VARIABLE_TASK_ARCHITECTURE = "variable_task_v1"


@dataclass(frozen=True)
class CMPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_actor_self_attention: bool = False


@dataclass
class CMPPOAgentSample:
    observation: np.ndarray
    primary_indices: np.ndarray
    backup_indices: np.ndarray
    priority_etas: np.ndarray
    redundancy_etas: np.ndarray
    primary_masks: np.ndarray
    backup_masks: np.ndarray
    active_slots: np.ndarray
    old_log_prob: float
    old_value: float


@dataclass
class CMPPOStep:
    local_states: dict[str, np.ndarray]
    shared_reward: float
    agent_samples: dict[str, CMPPOAgentSample]


@dataclass(frozen=True)
class CMPPOUpdateResult:
    actor_loss: float
    critic_loss: float
    entropy: float
    updated_agents: int


def _state_rows(value: np.ndarray, name: str) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size % OBSERVATION_INPUT_DIM:
        raise ValueError(f"{name} has an incomplete task block.")
    return flat.reshape(-1, OBSERVATION_INPUT_DIM)


def _pad_state_sets(
    state_sets: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(1, max((states.shape[0] for states in state_sets), default=0))
    values = np.zeros(
        (len(state_sets), maximum, OBSERVATION_INPUT_DIM), dtype=np.float32
    )
    masks = np.zeros((len(state_sets), maximum), dtype=bool)
    for index, states in enumerate(state_sets):
        count = states.shape[0]
        if count:
            values[index, :count] = states
            masks[index, :count] = True
    return (
        torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
    )


class CMPPOAgent:
    """Independent Actor/Critic pair owned by one CH agent."""

    def __init__(
        self,
        *,
        action_spec: ActionSpec,
        device: torch.device,
        config: CMPPOConfig,
    ) -> None:
        self.num_targets = action_spec.num_discrete_targets
        self.device = device
        self.config = config
        self.per_task_policy_dim = 2 * self.num_targets + 4
        self.actor = VariableTaskActorNetwork(
            per_task_state_dim=OBSERVATION_INPUT_DIM,
            per_task_action_dim=self.per_task_policy_dim,
            use_self_attention=config.use_actor_self_attention,
        ).to(device)
        self.critic = VariableTaskValueNetwork(OBSERVATION_INPUT_DIM).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)

    def policy_parameters(
        self,
        observations: torch.Tensor,
        task_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_output = self.actor(observations, task_mask)
        primary_logits = raw_output[..., : self.num_targets]
        backup_logits = raw_output[..., self.num_targets : 2 * self.num_targets]
        continuous = raw_output[..., 2 * self.num_targets :]
        priority_params = F.softplus(continuous[..., :2]) + 1.0
        redundancy_params = F.softplus(continuous[..., 2:]) + 1.0
        return primary_logits, backup_logits, priority_params, redundancy_params

class CMPPOSystem:
    """Multi-agent PPO with one Actor and centralized Critic per CH."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_spec: ActionSpec,
        device: str | torch.device,
        redundancy_mode: str = "hybrid",
        config: CMPPOConfig | None = None,
    ) -> None:
        if redundancy_mode not in {"none", "hybrid"}:
            raise ValueError("redundancy_mode must be 'none' or 'hybrid'")
        self.device = torch.device(device)
        self.config = config or CMPPOConfig()
        if state_dim <= 0 or state_dim % OBSERVATION_INPUT_DIM:
            raise ValueError("Initial observation does not contain complete task blocks.")
        if action_spec.num_task_slots != state_dim // OBSERVATION_INPUT_DIM:
            raise ValueError("Initial state/action task counts differ.")
        self.default_state_dim = OBSERVATION_INPUT_DIM
        self.num_targets = action_spec.num_discrete_targets
        self.redundancy_mode = redundancy_mode
        self.agents: dict[str, CMPPOAgent] = {}
        self.state_dims: dict[str, int] = {}
        self.action_specs: dict[str, ActionSpec] = {}

    def _sorted_agent_ids(self) -> list[str]:
        return sorted(self.agents)

    def joint_state_dim(self) -> int:
        return OBSERVATION_INPUT_DIM

    def ensure_agent(
        self,
        agent_id: str,
        state_dim: int,
        action_spec: ActionSpec,
    ) -> None:
        if state_dim <= 0 or state_dim % OBSERVATION_INPUT_DIM:
            raise ValueError(f"Agent {agent_id} state has incomplete task blocks.")
        if action_spec.num_task_slots != state_dim // OBSERVATION_INPUT_DIM:
            raise ValueError(f"Agent {agent_id} state/action task counts differ.")
        if action_spec.num_discrete_targets != self.num_targets:
            raise ValueError("CMPPO received a changed per-task target count.")
        self.state_dims[agent_id] = OBSERVATION_INPUT_DIM
        self.action_specs[agent_id] = action_spec
        if agent_id in self.agents:
            return
        self.agents[agent_id] = CMPPOAgent(
            action_spec=action_spec,
            device=self.device,
            config=self.config,
        )

    def _rebuild_joint_critics(self) -> None:
        """Compatibility no-op for variable-set value critics."""

    def build_joint_state(
        self,
        local_states: dict[str, np.ndarray],
        agent_ids: list[str] | None = None,
    ) -> np.ndarray:
        ordered_ids = agent_ids or sorted(local_states)
        parts = []
        for agent_id in ordered_ids:
            if agent_id not in local_states:
                continue
            flat = np.asarray(local_states[agent_id], dtype=np.float32).reshape(-1)
            if flat.size % OBSERVATION_INPUT_DIM:
                raise ValueError(f"Agent {agent_id} state has incomplete task blocks.")
            parts.append(flat.reshape(-1, OBSERVATION_INPUT_DIM))
        return (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros((0, OBSERVATION_INPUT_DIM), dtype=np.float32)
        )

    @staticmethod
    def _node_ids(action_spec: ActionSpec, slot_index: int) -> list[str]:
        if action_spec.slot_target_node_ids is not None:
            return action_spec.slot_target_node_ids[slot_index]
        return action_spec.target_node_ids

    @staticmethod
    def _backup_mask(
        node_ids: list[str],
        primary_index: int,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        primary_id = node_ids[primary_index]
        if primary_id.startswith("uav-"):
            valid_prefixes = ("bs-", "leo-")
        elif primary_id.startswith("bs-"):
            valid_prefixes = ("leo-",)
        else:
            valid_prefixes = ()
        return np.asarray(
            [
                bool(allowed)
                and index != primary_index
                and bool(node_id)
                and node_id.startswith(valid_prefixes)
                for index, (node_id, allowed) in enumerate(
                    zip(node_ids, candidate_mask)
                )
            ],
            dtype=bool,
        )

    def _sample_agent_action(
        self,
        agent_id: str,
        observation: np.ndarray,
        action_spec: ActionSpec,
        joint_state_tensor: torch.Tensor,
        joint_state_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[MultiTaskOffloadingAction, CMPPOAgentSample]:
        agent = self.agents[agent_id]
        observation_tensor = torch.as_tensor(
            _state_rows(observation, f"{agent_id} observation"),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        num_slots = action_spec.num_task_slots
        with torch.no_grad():
            primary_logits, backup_logits, priority_params, redundancy_params = (
                agent.policy_parameters(observation_tensor)
            )
            old_value = float(
                agent.critic(joint_state_tensor, joint_state_mask).item()
            )

        primary_indices = np.zeros(num_slots, dtype=np.int64)
        backup_indices = np.full(num_slots, -1, dtype=np.int64)
        priority_etas = np.full(num_slots, 0.5, dtype=np.float32)
        redundancy_etas = np.zeros(num_slots, dtype=np.float32)
        primary_masks = np.asarray(action_spec.slot_target_masks, dtype=bool).copy()
        backup_masks = np.zeros_like(primary_masks)
        active_slots = primary_masks.any(axis=1)
        slot_actions: list[SlotAction] = []
        old_log_prob = 0.0

        for slot_index in range(num_slots):
            node_ids = self._node_ids(action_spec, slot_index)
            if not active_slots[slot_index]:
                slot_actions.append(
                    SlotAction("", priority_eta=0.5, redundancy_eta=0.0)
                )
                continue
            mask_tensor = torch.as_tensor(
                primary_masks[slot_index], dtype=torch.bool, device=self.device
            )
            primary_dist = Categorical(
                logits=primary_logits[0, slot_index].masked_fill(~mask_tensor, -1e9)
            )
            primary_tensor = (
                primary_dist.probs.argmax()
                if deterministic
                else primary_dist.sample()
            )
            primary_index = int(primary_tensor.item())
            primary_indices[slot_index] = primary_index
            priority_dist = Beta(
                priority_params[0, slot_index, 0],
                priority_params[0, slot_index, 1],
            )
            priority_tensor = (
                priority_params[0, slot_index, 0]
                / priority_params[0, slot_index].sum()
                if deterministic
                else priority_dist.sample()
            )
            priority_etas[slot_index] = float(priority_tensor.item())
            old_log_prob += float(
                (
                    primary_dist.log_prob(primary_tensor)
                    + priority_dist.log_prob(priority_tensor)
                ).item()
            )

            backup_node_id = None
            if self.redundancy_mode == "hybrid":
                backup_mask = self._backup_mask(
                    node_ids, primary_index, primary_masks[slot_index]
                )
                backup_masks[slot_index] = backup_mask
                if backup_mask.any():
                    backup_mask_tensor = torch.as_tensor(
                        backup_mask, dtype=torch.bool, device=self.device
                    )
                    backup_dist = Categorical(
                        logits=backup_logits[0, slot_index].masked_fill(
                            ~backup_mask_tensor, -1e9
                        )
                    )
                    backup_tensor = (
                        backup_dist.probs.argmax()
                        if deterministic
                        else backup_dist.sample()
                    )
                    backup_index = int(backup_tensor.item())
                    backup_indices[slot_index] = backup_index
                    backup_node_id = node_ids[backup_index]
                    redundancy_dist = Beta(
                        redundancy_params[0, slot_index, 0],
                        redundancy_params[0, slot_index, 1],
                    )
                    redundancy_tensor = (
                        redundancy_params[0, slot_index, 0]
                        / redundancy_params[0, slot_index].sum()
                        if deterministic
                        else redundancy_dist.sample()
                    )
                    redundancy_etas[slot_index] = float(redundancy_tensor.item())
                    old_log_prob += float(
                        (
                            backup_dist.log_prob(backup_tensor)
                            + redundancy_dist.log_prob(redundancy_tensor)
                        ).item()
                    )

            slot_actions.append(
                SlotAction(
                    target_node_id=node_ids[primary_index],
                    priority_eta=float(priority_etas[slot_index]),
                    redundancy_eta=float(redundancy_etas[slot_index]),
                    backup_target_node_id=backup_node_id,
                )
            )

        return (
            MultiTaskOffloadingAction(slot_actions=slot_actions),
            CMPPOAgentSample(
                observation=np.asarray(observation, dtype=np.float32),
                primary_indices=primary_indices,
                backup_indices=backup_indices,
                priority_etas=priority_etas,
                redundancy_etas=redundancy_etas,
                primary_masks=primary_masks,
                backup_masks=backup_masks,
                active_slots=active_slots,
                old_log_prob=old_log_prob,
                old_value=old_value,
            ),
        )

    def sample_actions(
        self,
        observations: dict[str, np.ndarray],
        action_specs: dict[str, ActionSpec],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, MultiTaskOffloadingAction], CMPPOStep]:
        for agent_id, observation in observations.items():
            self.ensure_agent(
                agent_id,
                int(observation.shape[0]),
                action_specs[agent_id],
            )
        joint_state = self.build_joint_state(observations)
        joint_state_tensor, joint_state_mask = _pad_state_sets(
            [joint_state], self.device
        )

        actions: dict[str, MultiTaskOffloadingAction] = {}
        samples: dict[str, CMPPOAgentSample] = {}
        for agent_id, observation in observations.items():
            action, sample = self._sample_agent_action(
                agent_id,
                observation,
                action_specs[agent_id],
                joint_state_tensor,
                joint_state_mask,
                deterministic=deterministic,
            )
            actions[agent_id] = action
            samples[agent_id] = sample
        return actions, CMPPOStep(
            local_states={
                agent_id: np.asarray(state, dtype=np.float32)
                for agent_id, state in observations.items()
            },
            shared_reward=0.0,
            agent_samples=samples,
        )

    def load(self, checkpoint_path: str | Path, *, strict: bool = False) -> Path:
        """Load weights into agents already materialized from a compatible environment."""

        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "cmppo":
            raise ValueError(f"Unsupported checkpoint algorithm in {path}")
        if checkpoint.get("architecture") != VARIABLE_TASK_ARCHITECTURE:
            raise ValueError(
                "This checkpoint uses the retired fixed-task-slot architecture; "
                "train a new variable-task checkpoint."
            )
        checkpoint_agents = checkpoint.get("agents", {})
        missing = sorted(set(checkpoint_agents) - set(self.agents))
        unexpected = sorted(set(self.agents) - set(checkpoint_agents))
        if strict and (missing or unexpected):
            raise ValueError(
                f"Checkpoint agents do not match environment: missing={missing}, unexpected={unexpected}"
            )
        for agent_id, state in checkpoint_agents.items():
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            agent.actor.load_state_dict(state["actor_state_dict"])
            agent.critic.load_state_dict(state["critic_state_dict"])
            if "actor_optimizer_state_dict" in state:
                agent.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in state:
                agent.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        return path

    def _evaluate_actor_batch(
        self,
        agent: CMPPOAgent,
        samples: list[CMPPOAgentSample],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(samples)
        maximum = max(sample.primary_indices.size for sample in samples)
        observations_np = np.zeros(
            (batch_size, maximum, OBSERVATION_INPUT_DIM), dtype=np.float32
        )
        task_mask_np = np.zeros((batch_size, maximum), dtype=bool)
        primary_indices_np = np.zeros((batch_size, maximum), dtype=np.int64)
        backup_indices_np = np.full((batch_size, maximum), -1, dtype=np.int64)
        priority_np = np.full((batch_size, maximum), 0.5, dtype=np.float32)
        redundancy_np = np.full((batch_size, maximum), 0.5, dtype=np.float32)
        primary_masks_np = np.zeros(
            (batch_size, maximum, self.num_targets), dtype=bool
        )
        backup_masks_np = np.zeros_like(primary_masks_np)
        active_np = np.zeros((batch_size, maximum), dtype=bool)
        for index, sample in enumerate(samples):
            count = sample.primary_indices.size
            observations_np[index, :count] = _state_rows(
                sample.observation, "PPO observation"
            )
            task_mask_np[index, :count] = True
            primary_indices_np[index, :count] = sample.primary_indices
            backup_indices_np[index, :count] = sample.backup_indices
            priority_np[index, :count] = sample.priority_etas
            redundancy_np[index, :count] = sample.redundancy_etas
            primary_masks_np[index, :count] = sample.primary_masks
            backup_masks_np[index, :count] = sample.backup_masks
            active_np[index, :count] = sample.active_slots
        observations = torch.as_tensor(
            observations_np, dtype=torch.float32, device=self.device
        )
        task_mask = torch.as_tensor(task_mask_np, dtype=torch.bool, device=self.device)
        primary_indices = torch.as_tensor(
            primary_indices_np, dtype=torch.long, device=self.device
        )
        backup_indices = torch.as_tensor(
            backup_indices_np, dtype=torch.long, device=self.device
        )
        priority_etas = torch.as_tensor(
            priority_np, dtype=torch.float32, device=self.device
        ).clamp(1e-6, 1.0 - 1e-6)
        redundancy_etas = torch.as_tensor(
            redundancy_np, dtype=torch.float32, device=self.device
        ).clamp(1e-6, 1.0 - 1e-6)
        primary_masks = torch.as_tensor(
            primary_masks_np, dtype=torch.bool, device=self.device
        )
        backup_masks = torch.as_tensor(
            backup_masks_np, dtype=torch.bool, device=self.device
        )
        active_slots = torch.as_tensor(
            active_np, dtype=torch.float32, device=self.device
        )
        primary_logits, backup_logits, priority_params, redundancy_params = (
            agent.policy_parameters(observations, task_mask)
        )
        safe_primary_masks = primary_masks.clone()
        inactive_slots = ~safe_primary_masks.any(dim=-1)
        safe_primary_masks[..., 0] |= inactive_slots
        primary_dist = Categorical(
            logits=primary_logits.masked_fill(~safe_primary_masks, -1e9)
        )
        priority_dist = Beta(priority_params[..., 0], priority_params[..., 1])
        log_prob = (
            primary_dist.log_prob(primary_indices)
            + priority_dist.log_prob(priority_etas)
        ) * active_slots
        entropy = (
            primary_dist.entropy() + priority_dist.entropy()
        ) * active_slots

        if self.redundancy_mode == "hybrid":
            backup_active = backup_masks.any(dim=-1) & task_mask
            safe_backup_masks = backup_masks.clone()
            safe_backup_masks[..., 0] |= ~backup_active
            backup_dist = Categorical(
                logits=backup_logits.masked_fill(~safe_backup_masks, -1e9)
            )
            redundancy_dist = Beta(
                redundancy_params[..., 0], redundancy_params[..., 1]
            )
            backup_actions = backup_indices.clamp(min=0)
            backup_active_float = backup_active.float()
            log_prob += (
                backup_dist.log_prob(backup_actions)
                + redundancy_dist.log_prob(redundancy_etas)
            ) * backup_active_float
            entropy += (
                backup_dist.entropy() + redundancy_dist.entropy()
            ) * backup_active_float
        return log_prob.sum(dim=-1), entropy.sum(dim=-1)

    def _agent_advantages_and_returns(
        self,
        agent_id: str,
        trajectory: list[CMPPOStep],
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        step_indices: list[int] = []
        advantage_by_step: dict[int, float] = {}
        return_by_step: dict[int, float] = {}
        gae = 0.0
        next_value = 0.0
        for step_index in range(len(trajectory) - 1, -1, -1):
            step = trajectory[step_index]
            sample = step.agent_samples.get(agent_id)
            if sample is None:
                gae = 0.0
                next_value = 0.0
                continue
            delta = (
                step.shared_reward
                + self.config.gamma * next_value
                - sample.old_value
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantage_by_step[step_index] = gae
            return_by_step[step_index] = gae + sample.old_value
            next_value = sample.old_value
        step_indices = sorted(advantage_by_step)
        advantages = np.asarray(
            [advantage_by_step[index] for index in step_indices], dtype=np.float32
        )
        returns = np.asarray(
            [return_by_step[index] for index in step_indices], dtype=np.float32
        )
        return step_indices, advantages, returns

    def update(self, trajectory: list[CMPPOStep]) -> CMPPOUpdateResult | None:
        if not trajectory or not self.agents:
            return None
        self._rebuild_joint_critics()
        ordered_agent_ids = self._sorted_agent_ids()
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        updated_agents = 0

        for agent_id in ordered_agent_ids:
            step_indices, advantages, returns = self._agent_advantages_and_returns(
                agent_id, trajectory
            )
            if not step_indices:
                continue
            agent = self.agents[agent_id]
            samples = [
                trajectory[index].agent_samples[agent_id] for index in step_indices
            ]
            joint_states, joint_masks = _pad_state_sets(
                [
                    self.build_joint_state(trajectory[index].local_states)
                    for index in step_indices
                ],
                self.device,
            )
            return_tensor = torch.as_tensor(
                returns, dtype=torch.float32, device=self.device
            )
            active_indices = [
                index for index, sample in enumerate(samples) if sample.active_slots.any()
            ]
            active_samples = [samples[index] for index in active_indices]
            if active_samples:
                actor_advantages = torch.as_tensor(
                    advantages[active_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                if actor_advantages.numel() > 1:
                    actor_advantages = (
                        actor_advantages - actor_advantages.mean()
                    ) / (actor_advantages.std(unbiased=False) + 1e-8)
                old_log_probs = torch.as_tensor(
                    [sample.old_log_prob for sample in active_samples],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                actor_advantages = None
                old_log_probs = None

            for _ in range(self.config.update_epochs):
                if active_samples and actor_advantages is not None and old_log_probs is not None:
                    actor_order = torch.randperm(
                        len(active_samples), device=self.device
                    )
                    for start in range(
                        0, len(active_samples), self.config.minibatch_size
                    ):
                        indices = actor_order[
                            start : start + self.config.minibatch_size
                        ]
                        minibatch_samples = [
                            active_samples[int(index)] for index in indices.cpu()
                        ]
                        new_log_probs, entropy = self._evaluate_actor_batch(
                            agent, minibatch_samples
                        )
                        ratio = (
                            new_log_probs - old_log_probs[indices]
                        ).clamp(-20.0, 20.0).exp()
                        unclipped = ratio * actor_advantages[indices]
                        clipped = ratio.clamp(
                            1.0 - self.config.clip_ratio,
                            1.0 + self.config.clip_ratio,
                        ) * actor_advantages[indices]
                        actor_loss = -torch.min(unclipped, clipped).mean()
                        entropy_mean = entropy.mean()
                        total_actor_loss = (
                            actor_loss - self.config.entropy_coef * entropy_mean
                        )
                        agent.actor_optimizer.zero_grad()
                        total_actor_loss.backward()
                        nn.utils.clip_grad_norm_(
                            agent.actor.parameters(), self.config.max_grad_norm
                        )
                        agent.actor_optimizer.step()
                        actor_losses.append(float(actor_loss.detach().cpu()))
                        entropies.append(float(entropy_mean.detach().cpu()))

                critic_order = torch.randperm(len(samples), device=self.device)
                for start in range(0, len(samples), self.config.minibatch_size):
                    indices = critic_order[start : start + self.config.minibatch_size]
                    predicted_values = agent.critic(
                        joint_states[indices], joint_masks[indices]
                    )
                    critic_loss = F.mse_loss(
                        predicted_values, return_tensor[indices]
                    )
                    agent.critic_optimizer.zero_grad()
                    (self.config.value_coef * critic_loss).backward()
                    nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), self.config.max_grad_norm
                    )
                    agent.critic_optimizer.step()
                    critic_losses.append(float(critic_loss.detach().cpu()))
            updated_agents += 1

        if updated_agents == 0:
            return None
        return CMPPOUpdateResult(
            actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            updated_agents=updated_agents,
        )

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "cmppo",
                "architecture": VARIABLE_TASK_ARCHITECTURE,
                "arrival_scope": "system",
                "redundancy_mode": self.redundancy_mode,
                "config": asdict(self.config),
                "state_dims": self.state_dims,
                "num_targets": self.num_targets,
                "agents": {
                    agent_id: {
                        "actor_state_dict": agent.actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict(),
                        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
                    }
                    for agent_id, agent in self.agents.items()
                },
            },
            path,
        )
        return path
