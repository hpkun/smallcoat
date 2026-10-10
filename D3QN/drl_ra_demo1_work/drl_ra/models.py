from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP Q-network with optional dueling value/advantage heads (paper Eq. 33)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64),
        dueling: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("at least one hidden layer is required")
        layers: list[nn.Module] = []
        previous = state_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(previous, 1)
            self.advantage = nn.Linear(previous, action_dim)
        else:
            self.q_head = nn.Linear(previous, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        if not self.dueling:
            return self.q_head(features)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class JointQNetwork(nn.Module):
    """Shared encoder with autoregressive D3QN heads for a joint action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_replicas: int,
        resource_levels: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64),
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("at least one hidden layer is required")
        layers: list[nn.Module] = []
        previous = state_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.encoder = nn.Sequential(*layers)
        dimensions = {
            "primary": action_dim,
            "replica_count": max_replicas,
            "replica_node": action_dim,
            "resource": resource_levels,
        }
        self.values = nn.ModuleDict({name: nn.Linear(previous, 1) for name in dimensions})
        self.advantages = nn.ModuleDict(
            {name: nn.Linear(previous, width) for name, width in dimensions.items()}
        )

    def forward(self, state: torch.Tensor, head: str) -> torch.Tensor:
        if head not in self.advantages:
            raise KeyError(f"unknown joint action head: {head}")
        features = self.encoder(state)
        value = self.values[head](features)
        advantage = self.advantages[head](features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Exclude unavailable UAV/satellite actions from selection."""
    return q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)


class MultiHeadPPOActor(nn.Module):
    """NTL actor with a shared encoder and three categorical action heads."""

    def __init__(
        self,
        observation_dim: int,
        air_action_dim: int,
        space_action_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.mode_head = nn.Linear(previous, 4)
        self.air_head = nn.Linear(previous, air_action_dim)
        self.space_head = nn.Linear(previous, space_action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(observation)
        return (
            self.mode_head(features),
            self.air_head(features),
            self.space_head(features),
        )


class CentralValueNetwork(nn.Module):
    """Training-only value network over the full hierarchical state."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = state_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)
