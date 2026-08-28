from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .action_space import ActionSpec, MixedActionCodec
from .maddpg_agent import AgentHyperParameters, MADDPGAgent
from .observation_builder import OBSERVATION_INPUT_DIM
from .replay_buffer import MultiAgentReplayBuffer, MultiAgentTransition


VARIABLE_TASK_ARCHITECTURE = "variable_task_v1"


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
    """CMADDPG with one shared per-task Actor per CH and set-based Critics."""

    def __init__(
        self,
        device: str = "cpu",
        agent_hyper_params: AgentHyperParameters | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.agent_hyper_params = agent_hyper_params or AgentHyperParameters()
        self.agents: dict[str, MADDPGAgent] = {}
        self.action_specs: dict[str, ActionSpec] = {}
        self.replay_buffer = MultiAgentReplayBuffer()
        self.state_dims: dict[str, int] = {}
        self.action_dims: dict[str, int] = {}

    def _sorted_agent_ids(self) -> list[str]:
        return sorted(self.agents)

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
        if agent_id not in self.agents:
            self.agents[agent_id] = MADDPGAgent(
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
            agent_id: self.agents[agent_id].act(observation, add_noise=add_noise)
            for agent_id, observation in observations.items()
        }

    def reset_noise(self) -> None:
        for agent in self.agents.values():
            agent.reset_noise()

    def decode_actions(self, raw_actions: dict[str, np.ndarray]):
        env_actions = {}
        critic_actions = {}
        for agent_id, raw_action in raw_actions.items():
            codec = MixedActionCodec(self.action_specs[agent_id])
            decoded = codec.decode_numpy(raw_action)
            env_actions[agent_id] = decoded.to_multi_task_action()
            critic_actions[agent_id] = codec.encode_for_critic(
                decoded.slot_target_indices,
                decoded.slot_backup_target_indices,
                decoded.slot_priority_etas,
                decoded.slot_redundancy_etas,
            )
        return env_actions, critic_actions

    def save(self, output_path: str | Path) -> Path:
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
                "agents": {
                    agent_id: {
                        "actor_state_dict": agent.actor.state_dict(),
                        "target_actor_state_dict": agent.target_actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict(),
                        "target_critic_state_dict": agent.target_critic.state_dict(),
                        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
                    }
                    for agent_id, agent in self.agents.items()
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
                "This checkpoint uses the retired fixed-task-slot architecture; "
                "train a new variable-task checkpoint."
            )
        saved_agents = checkpoint.get("agents", {})
        missing = sorted(set(saved_agents) - set(self.agents))
        unexpected = sorted(set(self.agents) - set(saved_agents))
        if strict and (missing or unexpected):
            raise ValueError(
                f"Checkpoint agents differ: missing={missing}, unexpected={unexpected}"
            )
        for agent_id, state in saved_agents.items():
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            agent.actor.load_state_dict(state["actor_state_dict"])
            agent.target_actor.load_state_dict(state["target_actor_state_dict"])
            agent.critic.load_state_dict(state["critic_state_dict"])
            agent.target_critic.load_state_dict(state["target_critic_state_dict"])
            if "actor_optimizer_state_dict" in state:
                agent.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in state:
                agent.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
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
        return torch.cat(
            [
                torch.softmax(raw[..., :target_count], dim=-1),
                torch.softmax(raw[..., target_count : 2 * target_count], dim=-1),
                torch.sigmoid(raw[..., -2:]),
            ],
            dim=-1,
        )

    def _current_sets(
        self, transition: MultiAgentTransition
    ) -> tuple[np.ndarray, np.ndarray]:
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for agent_id in sorted(set(transition.local_states) & set(transition.local_actions)):
            if agent_id not in self.agents:
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
                    agent = self.agents.get(agent_id)
                    if agent is None:
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
                    raw = agent.target_actor(state_tensor)
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

    def update(self, batch_size: int = 64) -> TrainingBatch | None:
        if len(self.replay_buffer) < batch_size or not self.agents:
            return None
        batch = self.replay_buffer.sample(batch_size)
        update_ids = [
            agent_id
            for agent_id in self._sorted_agent_ids()
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
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for index, agent_id in enumerate(update_ids):
            agent = self.agents[agent_id]
            with torch.no_grad():
                target_q = agent.target_critic(next_states, next_actions, next_mask)
                expected_q = rewards + agent.hyper_params.gamma * (1.0 - dones) * target_q
            current_q = agent.critic(joint_states, joint_actions, state_mask)
            critic_loss = agent.loss_fn(current_q, expected_q)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            if agent.hyper_params.critic_grad_clip_norm > 0:
                clip_grad_norm_(
                    agent.critic.parameters(), agent.hyper_params.critic_grad_clip_norm
                )
            agent.critic_optimizer.step()

            predicted_sets: list[torch.Tensor] = []
            presence: list[bool] = []
            for transition in batch:
                parts: list[torch.Tensor] = []
                present = False
                for other_id in sorted(
                    set(transition.local_states) & set(transition.local_actions)
                ):
                    if other_id not in self.agents:
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
                        raw = agent.actor(state_tensor)
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
                    torch.cat(parts) if parts else torch.zeros(
                        0, action_width, device=self.device
                    )
                )
                presence.append(present)
            predicted_actions, predicted_mask = _pad_tensors(
                predicted_sets, action_width, self.device
            )
            if not torch.equal(state_mask, predicted_mask):
                raise ValueError("Predicted action mask differs from state mask.")
            for parameter in agent.critic.parameters():
                parameter.requires_grad_(False)
            actor_q = agent.critic(joint_states, predicted_actions, state_mask)
            presence_tensor = torch.as_tensor(
                presence, dtype=torch.bool, device=self.device
            )
            actor_loss = -actor_q[presence_tensor].mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            for parameter in agent.critic.parameters():
                parameter.requires_grad_(True)
            if agent.hyper_params.actor_grad_clip_norm > 0:
                clip_grad_norm_(
                    agent.actor.parameters(), agent.hyper_params.actor_grad_clip_norm
                )
            agent.actor_optimizer.step()
            agent.soft_update()
            if index + 1 < len(update_ids):
                next_states, next_actions, next_mask = self._target_sets(batch)
            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))

        return TrainingBatch(
            actor_loss=float(np.mean(actor_losses)),
            critic_loss=float(np.mean(critic_losses)),
            batch_size=len(batch),
        )
