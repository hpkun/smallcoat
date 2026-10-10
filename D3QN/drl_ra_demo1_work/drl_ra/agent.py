from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import JointQNetwork, QNetwork, masked_q_values
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
        self.last_q_max = float("nan")
        self.last_q_mean = float("nan")

    def act(self, state: np.ndarray, action_mask: np.ndarray, epsilon: float = 0.0) -> int:
        available = np.flatnonzero(action_mask)
        if len(available) == 0:
            raise RuntimeError("environment supplied an empty action mask")
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = masked_q_values(self.online(tensor), mask)
            self.last_q_max = float(q_values.max().item())
            self.last_q_mean = float(q_values[q_values > -1e8].mean().item())
            greedy_action = int(q_values.argmax(dim=1).item())
        if self.rng.random() < epsilon:
            return int(self.rng.choice(available))
        return greedy_action

    def observe(
        self,
        state: np.ndarray,
        action: int,
        base_reward: float,
        cost: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
        *,
        discount: float | None = None,
    ) -> float | None:
        lagrangian_reward = float(base_reward - self.lagrange * cost)
        self.replay.add(
            state,
            action,
            lagrangian_reward,
            next_state,
            done,
            next_mask,
            self.gamma if discount is None else float(discount),
        )
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
        discounts = torch.as_tensor(batch.discounts, device=self.device)
        q_values = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._next_q_values(next_states, masks)
            targets = rewards + discounts * (1.0 - dones) * next_q
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


class ReplicaD3QNAgent(D3QNAgent):
    """Independent D3QN selector for within-task replica placement."""

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any], seed: int, device: str | torch.device = "cpu") -> None:
        replica_config = dict(config)
        replica_config["training"] = dict(config["training"])
        learned = config.get("learned_replica", {})
        replica_config["training"].update(learned.get("training", {}))
        super().__init__(
            state_dim,
            action_dim,
            replica_config,
            seed,
            device=device,
            dueling=True,
            double_q=True,
            constrained=False,
        )
        self.gamma_intra = float(learned.get("gamma_intra", 1.0))
        self.gamma_inter = float(learned.get("gamma_inter", replica_config["training"]["gamma"]))
        self.gamma = self.gamma_intra
        self.lagrange = 0.0

    def behavior_clone(
        self,
        states: np.ndarray,
        action_masks: np.ndarray,
        teacher_actions: np.ndarray,
        epochs: int,
        batch_size: int | None = None,
    ) -> list[float]:
        """Pretrain only the replica online network on masked teacher actions."""
        states = np.asarray(states, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=bool)
        teacher_actions = np.asarray(teacher_actions, dtype=np.int64)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("BC states have the wrong shape")
        if action_masks.shape != (len(states), self.action_dim):
            raise ValueError("BC action masks have the wrong shape")
        if teacher_actions.shape != (len(states),):
            raise ValueError("BC teacher actions have the wrong shape")
        if len(states) == 0:
            raise ValueError("BC dataset is empty")
        if np.any(teacher_actions < 0) or np.any(teacher_actions >= self.action_dim):
            raise ValueError("BC teacher action is out of range")
        if not np.all(action_masks[np.arange(len(states)), teacher_actions]):
            raise ValueError("BC teacher action is masked out")
        if epochs < 1:
            raise ValueError("BC epochs must be positive")
        batch_size = int(batch_size or self.batch_size)
        if batch_size < 1:
            raise ValueError("BC batch size must be positive")

        self.online.train()
        losses: list[float] = []
        indices = np.arange(len(states))
        for _ in range(int(epochs)):
            self.rng.shuffle(indices)
            epoch_losses: list[float] = []
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                state_tensor = torch.as_tensor(states[batch], dtype=torch.float32, device=self.device)
                mask_tensor = torch.as_tensor(action_masks[batch], dtype=torch.bool, device=self.device)
                action_tensor = torch.as_tensor(teacher_actions[batch], dtype=torch.long, device=self.device)
                q = self.online(state_tensor)
                q = q.masked_fill(~mask_tensor, -1e9)
                loss = nn.functional.cross_entropy(q, action_tensor)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
                self.optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
        self.target.load_state_dict(self.online.state_dict())
        self.online.eval()
        return losses

    def observe_sequence(
        self,
        transitions: list[tuple[np.ndarray, int, np.ndarray, bool, np.ndarray]],
        final_reward: float,
        *,
        episode_done: bool = True,
        bootstrap_state: np.ndarray | None = None,
        bootstrap_mask: np.ndarray | None = None,
    ) -> list[float]:
        """Propagate a task outcome through its replica decisions."""
        if not transitions:
            return []
        if not episode_done and (bootstrap_state is None or bootstrap_mask is None):
            raise ValueError("a continuing episode requires the next task state and action mask")
        losses = []
        for index, (state, action, successor, _, mask) in enumerate(transitions):
            last = index == len(transitions) - 1
            if last and not episode_done:
                successor, mask = bootstrap_state, bootstrap_mask
            loss = self.observe(state, action, final_reward if last else 0.0, 0.0,
                                successor, episode_done if last else False, mask,
                                discount=self.gamma_inter if last else self.gamma_intra)
            if loss is not None:
                losses.append(loss)
        return losses


class JointD3QNAgent:
    """Autoregressive joint D3QN with one encoder and four action heads."""

    HEADS = ("primary", "replica_count", "replica_node", "resource")

    def __init__(self, env: Any, config: dict[str, Any], seed: int, device: str | torch.device = "cpu", analytic_replicas: bool = False) -> None:
        train = config["training"]
        self.state_dim = int(env.joint_state_dim)
        self.action_dim = int(env.action_dim)
        self.max_replicas = int(env.env_cfg["max_replicas"])
        self.analytic_replicas = analytic_replicas
        self.resource_levels = tuple(float(item) for item in config.get("joint", {}).get("resource_levels", (0.25, 0.5, 0.75, 1.0)))
        if not self.resource_levels or any(not 0 < level <= 1 for level in self.resource_levels):
            raise ValueError("joint resource levels must be in (0, 1]")
        if 1.0 not in self.resource_levels:
            raise ValueError("joint resource levels must include 1.0")
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = JointQNetwork(self.state_dim, self.action_dim, self.max_replicas, len(self.resource_levels), train["hidden_sizes"]).to(self.device)
        self.target = JointQNetwork(self.state_dim, self.action_dim, self.max_replicas, len(self.resource_levels), train["hidden_sizes"]).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(train["learning_rate"]))
        self.gamma = float(train["gamma"])
        self.batch_size = int(train["batch_size"])
        self.target_update_steps = int(train["target_update_steps"])
        self.gradient_clip = float(train.get("gradient_clip", 10.0))
        pairs = (("primary", "resource"), ("primary", "replica_count"), ("replica_count", "replica_node"),
                 ("replica_count", "resource"), ("replica_node", "replica_node"),
                 ("replica_node", "resource"), ("resource", "primary"))
        self.buffers = {pair: ReplayBuffer(int(train["replay_capacity"]), seed + i)
                        for i, pair in enumerate(pairs)}
        self.lagrange = float(train["lambda_initial"])
        self.costs: deque[float] = deque(maxlen=int(train["cost_window"]))
        self.cost_budget = float(train["cost_budget"])
        self.lagrange_lr = float(train["lagrange_learning_rate"])
        self.lagrange_update_steps = int(train["lagrange_update_steps"])
        self.training_steps = 0
        self.last_q_max = float("nan")

    def _act(self, state: np.ndarray, mask: np.ndarray, head: str, epsilon: float) -> int:
        available = np.flatnonzero(mask)
        if len(available) == 0:
            raise RuntimeError(f"empty action mask for joint head {head}")
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = masked_q_values(self.online(state_tensor, head), action_mask)
            self.last_q_max = float(values.max().item())
            greedy = int(values.argmax(dim=1).item())
        return int(self.rng.choice(available)) if self.rng.random() < epsilon else greedy

    def _count_mask(self, env: Any, primary: int) -> np.ndarray:
        node_mask = self._node_mask(env)
        feasible = int(node_mask.sum()) - int(node_mask[primary])
        upper = min(self.max_replicas, 1 + feasible)
        return np.asarray([count <= upper for count in range(1, self.max_replicas + 1)], dtype=bool)

    def _node_mask(self, env: Any) -> np.ndarray:
        return np.asarray([item.available and env.joint_resource_mask([item.action], self.resource_levels).any()
                           for item in env.candidates], dtype=bool)

    def decide(
        self, env: Any, epsilon: float = 0.0
    ) -> tuple[int, list[int], float, list[tuple[str, np.ndarray, int, np.ndarray, np.ndarray]]]:
        transitions: list[tuple[str, np.ndarray, int, np.ndarray, np.ndarray]] = []
        primary_state = env.joint_observation([], 1)
        primary_mask = self._node_mask(env)
        primary = self._act(primary_state, primary_mask, "primary", epsilon)
        if self.analytic_replicas:
            selected = [item.action for item in env._replica_plan(env.candidates[primary])
                        if primary_mask[item.action]]
            resource_state = env.joint_observation(selected, len(selected))
            resource_mask = env.joint_resource_mask(selected, self.resource_levels)
            resource_action = self._act(resource_state, resource_mask, "resource", epsilon)
            transitions = [("primary", primary_state, primary, resource_state, primary_mask),
                           ("resource", resource_state, resource_action, resource_state.copy(), resource_mask)]
            return primary, selected, self.resource_levels[resource_action], transitions
        count_state = env.joint_observation([primary], 1)
        count_mask = self._count_mask(env, primary)
        count_action = self._act(count_state, count_mask, "replica_count", epsilon)
        target_count = count_action + 1
        selected = [primary]
        node_state = env.joint_observation(selected, target_count)
        transitions.append(("primary", primary_state, primary, count_state, primary_mask))
        transitions.append(("replica_count", count_state, count_action, node_state, count_mask))
        while len(selected) < target_count:
            node_mask = primary_mask.copy()
            node_mask[selected] = False
            node_action = self._act(node_state, node_mask, "replica_node", epsilon)
            selected.append(node_action)
            next_state = env.joint_observation(selected, target_count)
            transitions.append(("replica_node", node_state, node_action, next_state, node_mask))
            node_state = next_state
        resource_state = env.joint_observation(selected, target_count)
        resource_mask = env.joint_resource_mask(selected, self.resource_levels)
        resource_action = self._act(resource_state, resource_mask, "resource", epsilon)
        transitions.append(("resource", resource_state, resource_action, resource_state.copy(), resource_mask))
        return primary, selected, self.resource_levels[resource_action], transitions

    def observe_sequence(
        self,
        transitions: list[tuple[str, np.ndarray, int, np.ndarray, np.ndarray]],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
        cost: float = 0.0,
    ) -> list[float]:
        """Bootstrap into the next head; only the resource step advances task time."""
        losses: list[float] = []
        constrained_reward = float(reward - self.lagrange * cost)
        for index, (head, state, action, successor, mask) in enumerate(transitions):
            last = index == len(transitions) - 1
            if last:
                successor = next_state
                terminal = done
                next_head = "primary"
                successor_mask = next_mask
            else:
                terminal = False
                next_head, successor, _, _, successor_mask = transitions[index + 1]
            pair = (head, next_head)
            self.buffers[pair].add(state, action, constrained_reward if last else 0.0,
                                   successor, terminal, successor_mask, self.gamma if last else 1.0)
            loss = self._learn_head(pair)
            if loss is not None:
                losses.append(loss)
        self.training_steps += 1
        self.costs.append(float(cost))
        if self.training_steps % self.lagrange_update_steps == 0:
            self.lagrange = max(0.0, self.lagrange + self.lagrange_lr * (float(np.mean(self.costs)) - self.cost_budget))
        if self.training_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return losses

    def _learn_head(self, pair: tuple[str, str]) -> float | None:
        head, next_head = pair
        buffer = self.buffers[pair]
        if len(buffer) < self.batch_size:
            return None
        batch = buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)
        masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=self.device)
        q_values = self.online(states, head).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = masked_q_values(self.online(next_states, next_head), masks).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states, next_head).gather(1, next_actions).squeeze(1)
            discounts = torch.as_tensor(batch.discounts, device=self.device)
            target = rewards + discounts * (1.0 - dones) * next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(), "state_dim": self.state_dim, "action_dim": self.action_dim, "max_replicas": self.max_replicas, "resource_levels": self.resource_levels, "analytic_replicas": self.analytic_replicas, "lagrange": self.lagrange, "training_steps": self.training_steps, "metadata": metadata or {}}, output)

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if int(payload["state_dim"]) != self.state_dim or int(payload["action_dim"]) != self.action_dim:
            raise ValueError("joint checkpoint dimensions do not match the environment")
        if tuple(payload["resource_levels"]) != self.resource_levels or int(payload["max_replicas"]) != self.max_replicas:
            raise ValueError("joint checkpoint action semantics do not match the configuration")
        if bool(payload.get("analytic_replicas", False)) != self.analytic_replicas:
            raise ValueError("checkpoint replica policy does not match the method")
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload.get("target", payload["online"]))
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.lagrange = float(payload.get("lagrange", self.lagrange))
        self.training_steps = int(payload.get("training_steps", 0))
        return dict(payload.get("metadata", {}))
