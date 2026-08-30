from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from .action_space import ActionSpec, MAX_REPLICA_COUNT, MixedActionCodec
from .maddpg_agent import AgentHyperParameters, CHActorAgent
from .networks import VariableTaskCriticNetwork
from .observation_builder import OBSERVATION_INPUT_DIM
from .replay_buffer import MultiAgentReplayBuffer, MultiAgentTransition


VARIABLE_TASK_ARCHITECTURE = "multi_actor_global_critic_v2"


@dataclass(frozen=True)
class TrainingBatch:
    actor_loss: float
    critic_loss: float
    batch_size: int


def _rows(value: np.ndarray, width: int, name: str) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size % width:
        raise ValueError(f"{name} length {flat.size} is not divisible by {width}.")
    return flat.reshape(-1, width)


def _pad_numpy(
    sequences: list[np.ndarray], width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(1, max((sequence.shape[0] for sequence in sequences), default=0))
    values = np.zeros((len(sequences), maximum, width), dtype=np.float32)
    masks = np.zeros((len(sequences), maximum), dtype=bool)
    for index, sequence in enumerate(sequences):
        count = sequence.shape[0]
        if count:
            values[index, :count] = sequence
            masks[index, :count] = True
    return (
        torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
    )


def _pad_tensors(
    sequences: list[torch.Tensor], width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(1, max((sequence.shape[0] for sequence in sequences), default=0))
    padded: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for sequence in sequences:
        count = sequence.shape[0]
        if count < maximum:
            sequence = torch.cat(
                [sequence, torch.zeros(maximum - count, width, device=device)], dim=0
            )
        padded.append(sequence)
        masks.append(torch.arange(maximum, device=device) < count)
    return torch.stack(padded), torch.stack(masks)


class CMADDPGSystem:
    """CMADDPG with one Actor per logical CH and one set-based global Critic."""

    def __init__(
        self,
        device: str = "cpu",
        agent_hyper_params: AgentHyperParameters | None = None,
        max_actor_count: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.agent_hyper_params = agent_hyper_params or AgentHyperParameters()
        self.actors: dict[str, CHActorAgent] = {}
        self.max_actor_count = max_actor_count
        self.allowed_agent_ids: frozenset[str] | None = None
        self.active_agent_ids: frozenset[str] | None = None
        self.global_critic: VariableTaskCriticNetwork | None = None
        self.target_global_critic: VariableTaskCriticNetwork | None = None
        self.global_critic_optimizer: Adam | None = None
        if self.agent_hyper_params.critic_loss_name == "mse":
            self.critic_loss_fn: nn.Module = nn.MSELoss()
        elif self.agent_hyper_params.critic_loss_name == "smooth_l1":
            self.critic_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(
                f"Unsupported critic_loss_name: {self.agent_hyper_params.critic_loss_name}"
            )
        self.action_specs: dict[str, ActionSpec] = {}
        self.replay_buffer = MultiAgentReplayBuffer()
        self.state_dims: dict[str, int] = {}
        self.action_dims: dict[str, int] = {}

    @property
    def agents(self) -> dict[str, CHActorAgent]:
        """Compatibility view; logical agents now own Actor state only."""
        return self.actors

    def _sorted_agent_ids(self) -> list[str]:
        return sorted(self.actors)

    @property
    def total_actor_count(self) -> int:
        return len(self.actors)

    @property
    def active_actor_count(self) -> int:
        return len(self.active_agent_ids or ())

    def configure_agent_pool(self, agent_ids: list[str] | tuple[str, ...]) -> None:
        """Freeze the logical CH identities allowed to own Actor networks."""

        logical_ids = frozenset(agent_ids)
        if not logical_ids:
            raise ValueError("The logical Actor pool cannot be empty.")
        unexpected_existing = set(self.actors) - logical_ids
        if unexpected_existing:
            raise RuntimeError(
                "Existing Actors are outside the configured logical pool: "
                f"{sorted(unexpected_existing)}"
            )
        self.allowed_agent_ids = logical_ids
        self.max_actor_count = len(logical_ids)

    def set_active_agent_ids(self, agent_ids) -> None:
        active_ids = frozenset(agent_ids)
        if self.allowed_agent_ids is not None:
            unexpected = active_ids - self.allowed_agent_ids
            if unexpected:
                raise RuntimeError(
                    "Active logical agents are outside the configured Actor pool: "
                    f"{sorted(unexpected)}"
                )
        self.active_agent_ids = active_ids

    def _check_actor_creation(self, agent_id: str) -> None:
        if self.allowed_agent_ids is not None and agent_id not in self.allowed_agent_ids:
            raise RuntimeError(
                f"Refusing to create unexpected Actor {agent_id!r}; allowed logical "
                f"roles are {sorted(self.allowed_agent_ids)}."
            )
        if self.max_actor_count is not None and len(self.actors) >= self.max_actor_count:
            raise RuntimeError(
                f"Actor count would exceed configured limit {self.max_actor_count}: "
                f"attempted {agent_id!r}, existing={sorted(self.actors)}"
            )

    def joint_state_dim(self, agent_ids=None) -> int:
        del agent_ids
        return OBSERVATION_INPUT_DIM

    def joint_action_dim(self, agent_ids=None) -> int:
        del agent_ids
        return next(iter(self.action_dims.values()), 0)

    def ensure_agent(
        self, agent_id: str, state_dim: int, action_spec: ActionSpec
    ) -> None:
        if state_dim <= 0 or state_dim % OBSERVATION_INPUT_DIM:
            raise ValueError(
                f"Agent {agent_id} observation must contain complete "
                f"{OBSERVATION_INPUT_DIM}-value task blocks."
            )
        task_count = state_dim // OBSERVATION_INPUT_DIM
        if action_spec.num_task_slots != task_count:
            raise ValueError(f"Agent {agent_id} state/action task counts differ.")
        action_width = action_spec.per_slot_output_dim
        if self.action_dims and action_width not in set(self.action_dims.values()):
            raise ValueError("All CHs must use the same per-task action width.")
        if agent_id in self.action_dims and self.action_dims[agent_id] != action_width:
            raise ValueError(f"Agent {agent_id} per-task action width changed.")
        self.state_dims[agent_id] = OBSERVATION_INPUT_DIM
        self.action_dims[agent_id] = action_width
        self.action_specs[agent_id] = action_spec
        if self.global_critic is None:
            self.global_critic = VariableTaskCriticNetwork(
                per_task_state_dim=OBSERVATION_INPUT_DIM,
                per_task_action_dim=action_width,
            ).to(self.device)
            self.target_global_critic = copy.deepcopy(self.global_critic).to(self.device)
            self.global_critic_optimizer = Adam(
                self.global_critic.parameters(),
                lr=self.agent_hyper_params.critic_lr,
            )
        if agent_id not in self.actors:
            self._check_actor_creation(agent_id)
            self.actors[agent_id] = CHActorAgent(
                per_task_state_dim=OBSERVATION_INPUT_DIM,
                per_task_action_dim=action_width,
                device=self.device,
                hyper_params=self.agent_hyper_params,
            )

    def rebuild_joint_critics(self) -> None:
        """Compatibility no-op; set critics are independent of task/CH counts."""

    def act(
        self, observations: dict[str, np.ndarray], *, add_noise: bool = True
    ) -> dict[str, np.ndarray]:
        return {
            agent_id: self.actors[agent_id].act(observation, add_noise=add_noise)
            for agent_id, observation in observations.items()
        }

    def reset_noise(self) -> None:
        for actor in self.actors.values():
            actor.reset_noise()

    def decode_actions(self, raw_actions: dict[str, np.ndarray]):
        env_actions = {}
        critic_actions = {}
        for agent_id, raw_action in raw_actions.items():
            codec = MixedActionCodec(self.action_specs[agent_id])
            decoded = codec.decode_numpy(raw_action)
            env_actions[agent_id] = decoded.to_multi_task_action()
            critic_actions[agent_id] = codec.encode_for_critic(
                decoded.slot_replica_counts,
                decoded.slot_replica_target_indices,
                decoded.slot_priority_etas,
            )
        return env_actions, critic_actions

    def save(self, output_path: str | Path) -> Path:
        if (
            self.global_critic is None
            or self.target_global_critic is None
            or self.global_critic_optimizer is None
        ):
            raise RuntimeError("Cannot save CMADDPG before the global Critic is initialized.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "cmaddpg",
                "architecture": VARIABLE_TASK_ARCHITECTURE,
                "arrival_scope": "system",
                "agent_hyper_params": asdict(self.agent_hyper_params),
                "state_dims": dict(self.state_dims),
                "action_dims": dict(self.action_dims),
                "actor_count": self.total_actor_count,
                "active_actor_count": self.active_actor_count,
                "max_actor_count": self.max_actor_count,
                "allowed_agent_ids": (
                    sorted(self.allowed_agent_ids)
                    if self.allowed_agent_ids is not None
                    else None
                ),
                "actors": {
                    agent_id: {
                        "actor_state_dict": actor.actor.state_dict(),
                        "target_actor_state_dict": actor.target_actor.state_dict(),
                        "actor_optimizer_state_dict": actor.actor_optimizer.state_dict(),
                    }
                    for agent_id, actor in self.actors.items()
                },
                "global_critic": {
                    "critic_state_dict": self.global_critic.state_dict(),
                    "target_critic_state_dict": self.target_global_critic.state_dict(),
                    "optimizer_state_dict": self.global_critic_optimizer.state_dict(),
                },
            },
            path,
        )
        return path

    def load(self, checkpoint_path: str | Path, *, strict: bool = False) -> Path:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "cmaddpg":
            raise ValueError(f"Unsupported checkpoint algorithm in {path}")
        if checkpoint.get("architecture") != VARIABLE_TASK_ARCHITECTURE:
            raise ValueError(
                "This checkpoint does not use the multi-Actor/global-Critic architecture; "
                "train a new checkpoint."
            )
        saved_actors = checkpoint.get("actors", {})
        missing = sorted(set(saved_actors) - set(self.actors))
        unexpected = sorted(set(self.actors) - set(saved_actors))
        if strict and (missing or unexpected):
            raise ValueError(
                f"Checkpoint actors differ: missing={missing}, unexpected={unexpected}"
            )
        for agent_id, state in saved_actors.items():
            actor = self.actors.get(agent_id)
            if actor is None:
                continue
            actor.actor.load_state_dict(state["actor_state_dict"])
            actor.target_actor.load_state_dict(state["target_actor_state_dict"])
            if "actor_optimizer_state_dict" in state:
                actor.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        if (
            self.global_critic is None
            or self.target_global_critic is None
            or self.global_critic_optimizer is None
        ):
            raise RuntimeError("Initialize at least one Actor before loading a checkpoint.")
        critic_state = checkpoint.get("global_critic")
        if not isinstance(critic_state, dict):
            raise ValueError("Checkpoint is missing global Critic state.")
        self.global_critic.load_state_dict(critic_state["critic_state_dict"])
        self.target_global_critic.load_state_dict(
            critic_state["target_critic_state_dict"]
        )
        if "optimizer_state_dict" in critic_state:
            self.global_critic_optimizer.load_state_dict(
                critic_state["optimizer_state_dict"]
            )
        return path

    def store_transitions(
        self,
        observations: dict[str, np.ndarray],
        critic_actions: dict[str, np.ndarray],
        shared_reward: float,
        next_observations: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        active_ids = sorted(set(observations) & set(critic_actions))
        if not active_ids:
            return
        local_states = {
            agent_id: np.asarray(observations[agent_id], dtype=np.float32).reshape(-1)
            for agent_id in active_ids
        }
        local_actions = {
            agent_id: np.asarray(critic_actions[agent_id], dtype=np.float32).reshape(-1)
            for agent_id in active_ids
        }
        for agent_id in active_ids:
            state_count = _rows(
                local_states[agent_id], OBSERVATION_INPUT_DIM, f"{agent_id} state"
            ).shape[0]
            action_count = _rows(
                local_actions[agent_id], self.action_dims[agent_id], f"{agent_id} action"
            ).shape[0]
            if state_count != action_count:
                raise ValueError(f"Agent {agent_id} state/action task counts differ.")
        next_states = {
            agent_id: np.asarray(state, dtype=np.float32).reshape(-1)
            for agent_id, state in next_observations.items()
        }
        self.replay_buffer.push(
            MultiAgentTransition(
                agent_ids=tuple(sorted(set(active_ids) | set(next_states))),
                reward=float(shared_reward),
                done=float(done),
                local_states=local_states,
                next_local_states=next_states,
                local_actions=local_actions,
            )
        )

    @staticmethod
    def _actor_to_critic(raw: torch.Tensor, target_count: int) -> torch.Tensor:
        parts = [torch.softmax(raw[..., :MAX_REPLICA_COUNT], dim=-1)]
        for head_index in range(MAX_REPLICA_COUNT):
            start = MAX_REPLICA_COUNT + head_index * target_count
            parts.append(torch.softmax(raw[..., start : start + target_count], dim=-1))
        parts.append(torch.sigmoid(raw[..., -1:]))
        return torch.cat(parts, dim=-1)

    def _current_sets(
        self, transition: MultiAgentTransition
    ) -> tuple[np.ndarray, np.ndarray]:
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for agent_id in sorted(set(transition.local_states) & set(transition.local_actions)):
            if agent_id not in self.actors:
                continue
            state_rows = _rows(
                transition.local_states[agent_id], OBSERVATION_INPUT_DIM, "state"
            )
            action_rows = _rows(
                transition.local_actions[agent_id], self.action_dims[agent_id], "action"
            )
            if state_rows.shape[0] != action_rows.shape[0]:
                raise ValueError("Replay state/action task counts differ.")
            states.append(state_rows)
            actions.append(action_rows)
        action_width = next(iter(self.action_dims.values()))
        return (
            np.concatenate(states) if states else np.zeros((0, OBSERVATION_INPUT_DIM), np.float32),
            np.concatenate(actions) if actions else np.zeros((0, action_width), np.float32),
        )

    def _target_sets(
        self, batch: list[MultiAgentTransition]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_sets: list[torch.Tensor] = []
        action_sets: list[torch.Tensor] = []
        action_width = next(iter(self.action_dims.values()))
        with torch.no_grad():
            for transition in batch:
                states: list[torch.Tensor] = []
                actions: list[torch.Tensor] = []
                for agent_id in sorted(transition.next_local_states):
                    actor = self.actors.get(agent_id)
                    if actor is None:
                        continue
                    state_array = _rows(
                        transition.next_local_states[agent_id],
                        OBSERVATION_INPUT_DIM,
                        "next state",
                    )
                    if not state_array.shape[0]:
                        continue
                    state_tensor = torch.as_tensor(
                        state_array, dtype=torch.float32, device=self.device
                    )
                    raw = actor.target_actor(state_tensor)
                    states.append(state_tensor)
                    actions.append(
                        self._actor_to_critic(
                            raw, self.action_specs[agent_id].num_discrete_targets
                        )
                    )
                state_sets.append(
                    torch.cat(states) if states else torch.zeros(
                        0, OBSERVATION_INPUT_DIM, device=self.device
                    )
                )
                action_sets.append(
                    torch.cat(actions) if actions else torch.zeros(
                        0, action_width, device=self.device
                    )
                )
        padded_states, mask = _pad_tensors(
            state_sets, OBSERVATION_INPUT_DIM, self.device
        )
        padded_actions, action_mask = _pad_tensors(
            action_sets, action_width, self.device
        )
        if not torch.equal(mask, action_mask):
            raise ValueError("Target state/action masks differ.")
        return padded_states, padded_actions, mask

    def soft_update_global_critic(self) -> None:
        if self.global_critic is None or self.target_global_critic is None:
            raise RuntimeError("Global Critic is not initialized.")
        tau = self.agent_hyper_params.tau
        for target_param, param in zip(
            self.target_global_critic.parameters(), self.global_critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def update(self, batch_size: int = 64) -> TrainingBatch | None:
        if len(self.replay_buffer) < batch_size or not self.actors:
            return None
        if (
            self.global_critic is None
            or self.target_global_critic is None
            or self.global_critic_optimizer is None
        ):
            raise RuntimeError("Global Critic is not initialized.")
        batch = self.replay_buffer.sample(batch_size)
        update_ids = [
            agent_id
            for agent_id in self._sorted_agent_ids()
            if self.active_agent_ids is None or agent_id in self.active_agent_ids
            if any(agent_id in transition.local_states for transition in batch)
        ]
        if not update_ids:
            return None
        current = [self._current_sets(transition) for transition in batch]
        joint_states, state_mask = _pad_numpy(
            [item[0] for item in current], OBSERVATION_INPUT_DIM, self.device
        )
        action_width = next(iter(self.action_dims.values()))
        joint_actions, action_mask = _pad_numpy(
            [item[1] for item in current], action_width, self.device
        )
        if not torch.equal(state_mask, action_mask):
            raise ValueError("Replay state/action masks differ.")
        rewards = torch.as_tensor(
            [[transition.reward] for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [[transition.done] for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states, next_actions, next_mask = self._target_sets(batch)
        with torch.no_grad():
            target_q = self.target_global_critic(
                next_states, next_actions, next_mask
            )
            expected_q = (
                rewards
                + self.agent_hyper_params.gamma * (1.0 - dones) * target_q
            )
        current_q = self.global_critic(joint_states, joint_actions, state_mask)
        critic_loss = self.critic_loss_fn(current_q, expected_q)
        self.global_critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.agent_hyper_params.critic_grad_clip_norm > 0:
            clip_grad_norm_(
                self.global_critic.parameters(),
                self.agent_hyper_params.critic_grad_clip_norm,
            )
        self.global_critic_optimizer.step()

        actor_losses: list[float] = []
        for parameter in self.global_critic.parameters():
            parameter.requires_grad_(False)
        try:
            for agent_id in update_ids:
                actor = self.actors[agent_id]
                predicted_sets: list[torch.Tensor] = []
                presence: list[bool] = []
                for transition in batch:
                    parts: list[torch.Tensor] = []
                    present = False
                    for other_id in sorted(
                        set(transition.local_states) & set(transition.local_actions)
                    ):
                        if other_id not in self.actors:
                            continue
                        if other_id == agent_id:
                            state_tensor = torch.as_tensor(
                                _rows(
                                    transition.local_states[other_id],
                                    OBSERVATION_INPUT_DIM,
                                    "actor state",
                                ),
                                dtype=torch.float32,
                                device=self.device,
                            )
                            raw = actor.actor(state_tensor)
                            parts.append(
                                self._actor_to_critic(
                                    raw,
                                    self.action_specs[agent_id].num_discrete_targets,
                                )
                            )
                            present = True
                        else:
                            parts.append(
                                torch.as_tensor(
                                    _rows(
                                        transition.local_actions[other_id],
                                        self.action_dims[other_id],
                                        "replay action",
                                    ),
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                    predicted_sets.append(
                        torch.cat(parts)
                        if parts
                        else torch.zeros(0, action_width, device=self.device)
                    )
                    presence.append(present)
                predicted_actions, predicted_mask = _pad_tensors(
                    predicted_sets, action_width, self.device
                )
                if not torch.equal(state_mask, predicted_mask):
                    raise ValueError("Predicted action mask differs from state mask.")
                actor_q = self.global_critic(
                    joint_states, predicted_actions, state_mask
                )
                presence_tensor = torch.as_tensor(
                    presence, dtype=torch.bool, device=self.device
                )
                actor_loss = -actor_q[presence_tensor].mean()
                actor.actor_optimizer.zero_grad()
                actor_loss.backward()
                if actor.hyper_params.actor_grad_clip_norm > 0:
                    clip_grad_norm_(
                        actor.actor.parameters(),
                        actor.hyper_params.actor_grad_clip_norm,
                    )
                actor.actor_optimizer.step()
                actor_losses.append(float(actor_loss.item()))
        finally:
            for parameter in self.global_critic.parameters():
                parameter.requires_grad_(True)

        for agent_id in update_ids:
            self.actors[agent_id].soft_update_actor()
        self.soft_update_global_critic()

        return TrainingBatch(
            actor_loss=float(np.mean(actor_losses)),
            critic_loss=float(critic_loss.item()),
            batch_size=len(batch),
        )
