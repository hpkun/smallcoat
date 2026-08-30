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


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Exclude unavailable UAV/satellite actions from selection."""
    return q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)

