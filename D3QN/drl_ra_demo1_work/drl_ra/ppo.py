from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import NTLAction, NTL_AIR, NTL_BOTH, NTL_NONE, NTL_SPACE
from .models import CentralValueNetwork, MultiHeadPPOActor


@dataclass
class PPOTransition:
    observation: np.ndarray
    critic_state: np.ndarray
    air_mask: np.ndarray
    space_mask: np.ndarray
    mode_mask: np.ndarray
    action: NTLAction
    log_probability: float
    value: float
    reward: float
    bootstrap_value: float
    bootstrap_discount: float
    trace_discount: float


class NTLPPOAgent:
    """Single-agent PPO for the factorized UAV/LEO/redundancy action."""

    def __init__(self, env: Any, config: dict[str, Any], seed: int, device: str = "cpu") -> None:
        cfg = config["ppo_training"]
        self.device = torch.device(device)
        self.observation_dim = int(env.ntl_state_dim)
        self.critic_state_dim = int(env.critic_state_dim)
        torch.manual_seed(seed)
        hidden = tuple(cfg.get("hidden_sizes", (256, 128)))
        critic_hidden = tuple(cfg.get("critic_hidden_sizes", (256, 128)))
        self.actor = MultiHeadPPOActor(
            self.observation_dim,
            env.num_uavs + 1,
            env.num_satellites + 1,
            hidden,
        ).to(self.device)
        self.critic = CentralValueNetwork(self.critic_state_dim, critic_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=float(cfg["learning_rate"]),
        )
        self.gamma = float(cfg["gamma"])
        self.gae_lambda = float(cfg["gae_lambda"])
        self.clip_ratio = float(cfg["clip_ratio"])
        self.entropy_coefficient = float(cfg["entropy_coefficient"])
        self.value_coefficient = float(cfg["value_coefficient"])
        self.update_epochs = int(cfg["update_epochs"])
        self.minibatch_size = int(cfg["minibatch_size"])
        self.gradient_clip = float(cfg.get("gradient_clip", 0.5))

    def _tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _conditional_masks(mode: int, air_base: np.ndarray, space_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        air = np.zeros_like(air_base, dtype=bool)
        space = np.zeros_like(space_base, dtype=bool)
        if mode in (NTL_AIR, NTL_BOTH):
            air[:] = air_base
            air[0] = False
        else:
            air[0] = True
        if mode in (NTL_SPACE, NTL_BOTH):
            space[:] = space_base
            space[0] = False
        else:
            space[0] = True
        return air, space

    @staticmethod
    def _choice(distribution: Categorical, deterministic: bool) -> torch.Tensor:
        return distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()

    @staticmethod
    def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)

    def select_action(
        self,
        context: dict[str, Any],
        deterministic: bool = False,
        active: bool | None = None,
        include_value: bool = True,
    ) -> tuple[NTLAction, float, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        active = bool(context["gate"]) if active is None else bool(active)
        observation = self._tensor(context["ntl_observation"]).unsqueeze(0)
        critic_state = self._tensor(context["critic_state"]).unsqueeze(0) if include_value else None
        mode_mask = np.asarray(context["mode_mask"], dtype=bool).copy()
        if not active:
            mode_mask[:] = False
            mode_mask[NTL_NONE] = True
        with torch.no_grad():
            mode_logits, air_logits, space_logits = self.actor(observation)
            value = self.critic(critic_state) if critic_state is not None else torch.zeros(1, device=self.device)
            mode_distribution = Categorical(
                logits=self._masked(mode_logits, self._tensor(mode_mask, torch.bool).unsqueeze(0))
            )
            mode_tensor = self._choice(mode_distribution, deterministic)
            mode = int(mode_tensor.item())
            air_mask, space_mask = self._conditional_masks(
                mode,
                np.asarray(context["air_mask"], dtype=bool),
                np.asarray(context["space_mask"], dtype=bool),
            )
            air_distribution = Categorical(
                logits=self._masked(air_logits, self._tensor(air_mask, torch.bool).unsqueeze(0))
            )
            space_distribution = Categorical(
                logits=self._masked(space_logits, self._tensor(space_mask, torch.bool).unsqueeze(0))
            )
            air_tensor = self._choice(air_distribution, deterministic)
            space_tensor = self._choice(space_distribution, deterministic)
            log_probability = (
                mode_distribution.log_prob(mode_tensor)
                + air_distribution.log_prob(air_tensor)
                + space_distribution.log_prob(space_tensor)
            )
        action = NTLAction(mode, int(air_tensor.item()), int(space_tensor.item()))
        return action, float(log_probability.item()), float(value.item()), (air_mask, space_mask, mode_mask)

    def update(self, rollout: list[PPOTransition]) -> dict[str, float]:
        if not rollout:
            return {}
        rewards = np.asarray([item.reward for item in rollout], dtype=np.float32)
        old_values = np.asarray([item.value for item in rollout], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rollout))):
            item = rollout[index]
            delta = (
                float(rewards[index])
                + float(item.bootstrap_discount) * float(item.bootstrap_value)
                - float(old_values[index])
            )
            gae = delta + float(item.trace_discount) * gae
            advantages[index] = gae
        returns = advantages + old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)

        observations = self._tensor(np.stack([item.observation for item in rollout]))
        critic_states = self._tensor(np.stack([item.critic_state for item in rollout]))
        air_masks = self._tensor(np.stack([item.air_mask for item in rollout]), torch.bool)
        space_masks = self._tensor(np.stack([item.space_mask for item in rollout]), torch.bool)
        mode_masks = self._tensor(np.stack([item.mode_mask for item in rollout]), torch.bool)
        air_actions = self._tensor(np.asarray([item.action.air for item in rollout]), torch.long)
        space_actions = self._tensor(np.asarray([item.action.space for item in rollout]), torch.long)
        mode_actions = self._tensor(np.asarray([item.action.mode for item in rollout]), torch.long)
        old_log_probabilities = self._tensor(np.asarray([item.log_probability for item in rollout], dtype=np.float32))
        advantage_tensor = self._tensor(advantages)
        return_tensor = self._tensor(returns)

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        batch_size = len(rollout)
        for _ in range(self.update_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.minibatch_size):
                batch = order[start : start + self.minibatch_size]
                mode_logits, air_logits, space_logits = self.actor(observations[batch])
                values = self.critic(critic_states[batch])
                mode_distribution = Categorical(logits=self._masked(mode_logits, mode_masks[batch]))
                air_distribution = Categorical(logits=self._masked(air_logits, air_masks[batch]))
                space_distribution = Categorical(logits=self._masked(space_logits, space_masks[batch]))
                log_probability = (
                    mode_distribution.log_prob(mode_actions[batch])
                    + air_distribution.log_prob(air_actions[batch])
                    + space_distribution.log_prob(space_actions[batch])
                )
                ratio = torch.exp(log_probability - old_log_probabilities[batch])
                advantage = advantage_tensor[batch]
                policy_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage,
                ).mean()
                entropy = (
                    mode_distribution.entropy()
                    + air_distribution.entropy()
                    + space_distribution.entropy()
                ).mean()
                value_loss = nn.functional.mse_loss(values, return_tensor[batch])
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()],
                    self.gradient_clip,
                )
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
        return {
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "active_steps": float(len(rollout)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any], load_optimizer: bool = False) -> None:
        if "actor" in payload:
            self.actor.load_state_dict(payload["actor"])
            if "critic" in payload:
                self.critic.load_state_dict(payload["critic"])
        elif "policy" in payload:
            # Older D3QN-PPO checkpoints used the actor encoder for the value head.
            self.actor.load_state_dict(payload["policy"], strict=False)
        else:
            raise KeyError("PPO checkpoint contains neither 'actor' nor legacy 'policy' weights")
        if load_optimizer and "optimizer" in payload and "actor" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
