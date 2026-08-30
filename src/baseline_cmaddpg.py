from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .baseline_action_space import BaselineActionCodec
from .cmaddpg import CMADDPGSystem


BASELINE_ARCHITECTURE = "single_copy_multi_actor_global_critic_v1"


class BaselineCMADDPGSystem(CMADDPGSystem):
    """CMADDPG with a K+1 single-copy action per task."""

    def decode_actions(self, raw_actions):
        env_actions = {}
        critic_actions = {}
        for agent_id, raw_action in raw_actions.items():
            codec = BaselineActionCodec(self.action_specs[agent_id])
            decoded = codec.decode_numpy(raw_action)
            env_actions[agent_id] = decoded.to_multi_task_action()
            critic_actions[agent_id] = codec.encode_for_critic(
                decoded.slot_target_indices,
                decoded.slot_priority_etas,
            )
        return env_actions, critic_actions

    @staticmethod
    def _actor_to_critic(raw: torch.Tensor, target_count: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.softmax(raw[..., :target_count], dim=-1),
                torch.sigmoid(raw[..., -1:]),
            ],
            dim=-1,
        )

    def save(self, output_path: str | Path) -> Path:
        if (
            self.global_critic is None
            or self.target_global_critic is None
            or self.global_critic_optimizer is None
        ):
            raise RuntimeError("Cannot save baseline before the global Critic is initialized.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "cmaddpg-baseline",
                "architecture": BASELINE_ARCHITECTURE,
                "observation_profile": "baseline",
                "action_profile": "single-copy",
                "reward_profile": "profit-only",
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
        if checkpoint.get("algorithm") != "cmaddpg-baseline":
            raise ValueError(f"Unsupported baseline checkpoint algorithm in {path}")
        if checkpoint.get("architecture") != BASELINE_ARCHITECTURE:
            raise ValueError(f"Unsupported baseline checkpoint architecture in {path}")
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
                actor.actor_optimizer.load_state_dict(
                    state["actor_optimizer_state_dict"]
                )
        if (
            self.global_critic is None
            or self.target_global_critic is None
            or self.global_critic_optimizer is None
        ):
            raise RuntimeError("Initialize the baseline Actor pool before loading.")
        critic_state = checkpoint.get("global_critic")
        if not isinstance(critic_state, dict):
            raise ValueError("Baseline checkpoint is missing global Critic state.")
        self.global_critic.load_state_dict(critic_state["critic_state_dict"])
        self.target_global_critic.load_state_dict(
            critic_state["target_critic_state_dict"]
        )
        if "optimizer_state_dict" in critic_state:
            self.global_critic_optimizer.load_state_dict(
                critic_state["optimizer_state_dict"]
            )
        return path
