from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .networks import VariableTaskActorNetwork
from .networks import VariableTaskCriticNetwork


@dataclass(frozen=True)
class AgentHyperParameters:
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-2
    critic_lr: float = 1e-3
    batch_size: int = 64
    ou_mean: float = 0.0
    ou_std: float = 0.3
    ou_volatility: float = 0.15
    actor_grad_clip_norm: float = 1.0
    critic_grad_clip_norm: float = 1.0
    critic_loss_name: str = "smooth_l1"
    use_actor_self_attention: bool = False
    use_actor_resource_awareness: bool = False
    actor_attention_embed_dim: int = 64
    actor_attention_heads: int = 4


class OrnsteinUhlenbeckNoise:
    """Stateful OU exploration process used by the paper."""

    def __init__(
        self,
        size: int,
        *,
        mean: float = 0.0,
        std: float = 0.3,
        volatility: float = 0.15,
    ) -> None:
        self.size = int(size)
        self.mean = float(mean)
        self.std = float(std)
        self.volatility = float(volatility)
        self.state = np.full(self.size, self.mean, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(self.mean)

    def sample(self, size: int | None = None) -> np.ndarray:
        requested_size = self.size if size is None else int(size)
        if requested_size != self.size:
            self.size = requested_size
            self.state = np.full(self.size, self.mean, dtype=np.float32)
        dx = self.volatility * (self.mean - self.state)
        dx += self.std * np.random.normal(size=self.size)
        self.state = (self.state + dx).astype(np.float32)
        return self.state


class MADDPGAgent:
    """Actor/Critic pair owned by one CH agent."""

    def __init__(
        self,
        per_task_state_dim: int,
        per_task_action_dim: int,
        device: torch.device,
        hyper_params: AgentHyperParameters | None = None,
    ) -> None:
        self.device = device
        self.hyper_params = hyper_params or AgentHyperParameters()
        self.per_task_state_dim = int(per_task_state_dim)
        self.per_task_action_dim = int(per_task_action_dim)
        self.exploration_noise = OrnsteinUhlenbeckNoise(
            self.per_task_action_dim,
            mean=self.hyper_params.ou_mean,
            std=self.hyper_params.ou_std,
            volatility=self.hyper_params.ou_volatility,
        )

        self.actor = VariableTaskActorNetwork(
            self.per_task_state_dim,
            self.per_task_action_dim,
            use_self_attention=self.hyper_params.use_actor_self_attention,
            use_resource_awareness=self.hyper_params.use_actor_resource_awareness,
            attention_embed_dim=self.hyper_params.actor_attention_embed_dim,
            attention_heads=self.hyper_params.actor_attention_heads,
        ).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.critic = VariableTaskCriticNetwork(
            self.per_task_state_dim,
            self.per_task_action_dim,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.hyper_params.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.hyper_params.critic_lr)
        if self.hyper_params.critic_loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.hyper_params.critic_loss_name == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(
                f"Unsupported critic_loss_name: {self.hyper_params.critic_loss_name}"
            )

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size % self.per_task_state_dim != 0:
            raise ValueError(
                f"State length {state_array.size} is not divisible by "
                f"per-task dimension {self.per_task_state_dim}."
            )
        state_tensor = torch.as_tensor(
            state_array.reshape(-1, self.per_task_state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().reshape(-1)
        if add_noise:
            action = action + self.exploration_noise.sample(action.size)
        return action.astype(np.float32)

    def reset_noise(self) -> None:
        self.exploration_noise.reset()

    def soft_update(self) -> None:
        tau = self.hyper_params.tau
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
