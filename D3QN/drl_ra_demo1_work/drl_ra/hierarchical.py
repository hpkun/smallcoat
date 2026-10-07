from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import D3QNAgent
from .environment import NTLAction, SAGINEnv
from .ppo import NTLPPOAgent


class HierarchicalAgent:
    """Ground D3QN plus the independently deployable NTL PPO controller."""

    def __init__(self, env: SAGINEnv, config: dict[str, Any], seed: int, device: str = "cpu") -> None:
        ground_config = deepcopy(config)
        ground_config["training"].update(config.get("ground_training", {}))
        self.ground = D3QNAgent(
            env.ground_state_dim,
            env.ground_action_dim,
            ground_config,
            seed,
            device=device,
            dueling=True,
            double_q=True,
            constrained=bool(config.get("hierarchy", {}).get("ground_constrained", False)),
        )
        self.ppo = NTLPPOAgent(env, config, seed + 1, device=device)
        self.config = config
        self.seed = int(seed)
        self.device = device

    def decide(
        self,
        env: SAGINEnv,
        epsilon: float = 0.0,
        deterministic_ntl: bool = True,
    ) -> tuple[int, NTLAction, dict[str, Any], float, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ground_state = env.ground_observation()
        ground_action = self.ground.act(ground_state, env.ground_action_mask, epsilon)
        context = env.prepare_hierarchical(ground_action)
        action, log_probability, value, masks = self.ppo.select_action(
            context,
            deterministic=deterministic_ntl,
            active=bool(context["gate"]),
            include_value=False,
        )
        return ground_action, action, context, log_probability, value, masks

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ground": {
                    "online": self.ground.online.state_dict(),
                    "target": self.ground.target.state_dict(),
                    "optimizer": self.ground.optimizer.state_dict(),
                    "lagrange": self.ground.lagrange,
                    "state_dim": self.ground.state_dim,
                    "action_dim": self.ground.action_dim,
                },
                "ppo": self.ppo.state_dict(),
                "metadata": metadata or {},
            },
            output,
        )

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.ground.device, weights_only=False)
        ground = payload["ground"]
        if int(ground["state_dim"]) != self.ground.state_dim or int(ground["action_dim"]) != self.ground.action_dim:
            raise ValueError("hierarchical checkpoint dimensions do not match the environment")
        self.ground.online.load_state_dict(ground["online"])
        self.ground.target.load_state_dict(ground.get("target", ground["online"]))
        self.ground.lagrange = float(ground.get("lagrange", self.ground.lagrange))
        if load_optimizer and "optimizer" in ground:
            self.ground.optimizer.load_state_dict(ground["optimizer"])
        self.ppo.load_state_dict(payload["ppo"], load_optimizer=load_optimizer)
        return dict(payload.get("metadata", {}))

    def save_components(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> tuple[Path, Path]:
        """Save two deployment artifacts: one Ground D3QN and one NTL PPO."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        common_metadata = metadata or {}
        ground_path = output / "ground_d3qn.pt"
        ppo_path = output / "ntl_ppo.pt"
        torch.save(
            {
                "online": self.ground.online.state_dict(),
                "state_dim": self.ground.state_dim,
                "action_dim": self.ground.action_dim,
                "metadata": {**common_metadata, "component": "ground-d3qn"},
            },
            ground_path,
        )
        torch.save(
            {
                "actor": self.ppo.actor.state_dict(),
                "state_dim": self.ppo.observation_dim,
                "metadata": {**common_metadata, "component": "ntl-ppo"},
            },
            ppo_path,
        )
        return ground_path, ppo_path

    def load_components(
        self,
        ground_path: str | Path,
        ppo_path: str | Path,
        load_optimizer: bool = False,
    ) -> dict[str, Any]:
        ground = torch.load(ground_path, map_location=self.ground.device, weights_only=False)
        ppo = torch.load(ppo_path, map_location=self.ground.device, weights_only=False)
        if int(ground["state_dim"]) != self.ground.state_dim or int(ground["action_dim"]) != self.ground.action_dim:
            raise ValueError("Ground D3QN checkpoint dimensions do not match the environment")
        if int(ppo["state_dim"]) != self.ppo.observation_dim:
            raise ValueError("NTL PPO checkpoint dimensions do not match the environment")
        self.ground.online.load_state_dict(ground["online"])
        self.ground.target.load_state_dict(ground.get("target", ground["online"]))
        self.ground.lagrange = float(ground.get("lagrange", 0.0))
        if load_optimizer and "optimizer" in ground:
            self.ground.optimizer.load_state_dict(ground["optimizer"])
        self.ppo.load_state_dict(ppo, load_optimizer=load_optimizer)
        return dict(ground.get("metadata", {}))
