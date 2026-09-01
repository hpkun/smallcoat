from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import QNetwork, masked_q_values
from .replay import ReplayBuffer


class D3QNAgent:
    """Dueling Double DQN with the paper's sliding-window Lagrangian update."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict[str, Any],
        seed: int,
        device: str | torch.device = "cpu",
        dueling: bool = True,
        double_q: bool = True,
        constrained: bool = True,
    ) -> None:
        train = config["training"]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        hidden = train["hidden_sizes"]
        self.online = QNetwork(state_dim, action_dim, hidden, dueling=dueling).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden, dueling=dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(train["learning_rate"]))
        self.replay = ReplayBuffer(int(train["replay_capacity"]), seed)
        self.gamma = float(train["gamma"])
        self.batch_size = int(train["batch_size"])
        self.target_update_steps = int(train["target_update_steps"])
        self.gradient_clip = float(train.get("gradient_clip", 10.0))
        self.double_q = bool(double_q)
        self.constrained = bool(constrained)
        self.lagrange = float(train["lambda_initial"]) if constrained else 0.0
        self.cost_budget = float(train["cost_budget"])
        self.lagrange_lr = float(train["lagrange_learning_rate"])
        self.lagrange_update_steps = int(train["lagrange_update_steps"])
        self.costs: deque[float] = deque(maxlen=int(train["cost_window"]))
        self.training_steps = 0

    def act(self, state: np.ndarray, action_mask: np.ndarray, epsilon: float = 0.0) -> int:
        available = np.flatnonzero(action_mask)
        if len(available) == 0:
            raise RuntimeError("environment supplied an empty action mask")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(available))
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(masked_q_values(self.online(tensor), mask).argmax(dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        base_reward: float,
        cost: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> float | None:
        lagrangian_reward = float(base_reward - self.lagrange * cost)
        self.replay.add(state, action, lagrangian_reward, next_state, done, next_mask)
        self.costs.append(float(cost))
        self.training_steps += 1
        if self.constrained and self.training_steps % self.lagrange_update_steps == 0:
            average_cost = float(np.mean(self.costs))
            self.lagrange = max(0.0, self.lagrange + self.lagrange_lr * (average_cost - self.cost_budget))
        loss = self.learn()
        if self.training_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)
        masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=self.device)
        q_values = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._next_q_values(next_states, masks)
            targets = rewards + self.gamma * (1.0 - dones) * next_q
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def _next_q_values(self, next_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute masked Double-Q bootstrap values (paper Eq. 34)."""
        if self.double_q:
            next_actions = masked_q_values(self.online(next_states), masks).argmax(dim=1, keepdim=True)
            return self.target(next_states).gather(1, next_actions).squeeze(1)
        return masked_q_values(self.target(next_states), masks).max(dim=1).values

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lagrange": self.lagrange,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "metadata": metadata or {},
            },
            output,
        )

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload.get("target", payload["online"]))
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.lagrange = float(payload.get("lagrange", self.lagrange))
        return dict(payload.get("metadata", {}))
