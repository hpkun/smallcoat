from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .agent import D3QNAgent
from .environment import NTLAction, NTL_NONE, SAGINEnv
from .hierarchical import HierarchicalAgent
from .ppo import PPOTransition


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def method_options(method: str) -> dict[str, bool]:
    options = {
        "dueling": True,
        "double_q": True,
        "constrained": True,
        "redundancy": True,
    }
    if method == "d3qn":
        options.update(constrained=False, redundancy=False)
    elif method == "dqn":
        options.update(dueling=False, double_q=False, constrained=False, redundancy=False)
    elif method == "no-dueling":
        options["dueling"] = False
    elif method == "no-double":
        options["double_q"] = False
    elif method == "no-redundancy":
        options["redundancy"] = False
    elif method != "drl-ra":
        raise ValueError(f"unknown learning method: {method}")
    return options


def build_agent(method: str, env: SAGINEnv, config: dict[str, Any], seed: int, device: str) -> D3QNAgent:
    options = method_options(method)
    config["environment"]["enable_redundancy"] = options["redundancy"]
    return D3QNAgent(
        env.state_dim,
        env.action_dim,
        config,
        seed,
        device=device,
        dueling=options["dueling"],
        double_q=options["double_q"],
        constrained=options["constrained"],
    )


def build_hierarchical_agent(env: SAGINEnv, config: dict[str, Any], seed: int, device: str) -> HierarchicalAgent:
    return HierarchicalAgent(env, config, seed, device=device)


def train_agent(
    config: dict[str, Any],
    method: str,
    seed: int,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[D3QNAgent, list[dict[str, float]]]:
    seed_everything(seed)
    env = SAGINEnv(config, seed=seed)
    agent = build_agent(method, env, config, seed, device)
    train_cfg = config["training"]
    epsilon = float(train_cfg["epsilon_start"])
    epsilon_end = float(train_cfg["epsilon_end"])
    epsilon_decay = float(train_cfg["epsilon_decay"])
    episodes = int(train_cfg["episodes"])
    history: list[dict[str, float]] = []
    for episode in range(episodes):
        state, reset_info = env.reset(seed=seed * 10_000 + episode)
        mask = reset_info["action_mask"]
        losses: list[float] = []
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, mask, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            loss = agent.observe(state, action, reward, float(info["cost"]), next_state, done, info["action_mask"])
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state, mask = next_state, info["action_mask"]
        summary = env.summary()
        row = {
            "episode": float(episode + 1),
            "reward": total_reward,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epsilon": epsilon,
            "lagrange": agent.lagrange,
            **summary,
        }
        history.append(row)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if progress and ((episode + 1) == 1 or (episode + 1) % max(1, episodes // 10) == 0):
            print(
                f"episode={episode + 1}/{episodes} reward={total_reward:.2f} "
                f"TCR={summary['tcr']:.1f}% SR={summary['reliability_pct']:.1f}% "
                f"cost={summary['expected_cost']:.4f} lambda={agent.lagrange:.3f}"
            )
    return agent, history


def train_hierarchical_agent(
    config: dict[str, Any],
    seed: int,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[HierarchicalAgent, list[dict[str, Any]]]:
    """Jointly train the two-level system from one stream of task outcomes."""
    seed_everything(seed)
    env = SAGINEnv(config, seed=seed)
    agent = build_hierarchical_agent(env, config, seed, device)
    train_cfg = config["training"]
    epsilon = float(train_cfg["epsilon_start"])
    epsilon_end = float(train_cfg["epsilon_end"])
    epsilon_decay = float(train_cfg["epsilon_decay"])
    episodes = int(train_cfg["episodes"])
    history: list[dict[str, Any]] = []

    for episode in range(episodes):
        env.reset(seed=seed * 10_000 + episode)
        ground_state = env.ground_observation()
        ground_mask = env.ground_action_mask
        rollout: list[PPOTransition] = []
        pending_ppo: dict[str, Any] | None = None
        ground_losses: list[float] = []
        total_reward = 0.0
        done = False

        while not done:
            ground_action = agent.ground.act(ground_state, ground_mask, epsilon)
            context = env.prepare_hierarchical(ground_action)
            if context["gate"]:
                ntl_action, log_probability, value, stored_masks = agent.ppo.select_action(
                    context,
                    deterministic=False,
                    active=True,
                )
                if pending_ppo is not None:
                    rollout.append(
                        PPOTransition(
                            **pending_ppo["transition"],
                            reward=float(pending_ppo["reward"]),
                            bootstrap_value=value,
                            bootstrap_discount=agent.ppo.gamma ** int(pending_ppo["steps"]),
                            trace_discount=(agent.ppo.gamma * agent.ppo.gae_lambda)
                            ** int(pending_ppo["steps"]),
                        )
                    )
                pending_ppo = {
                    "transition": {
                        "observation": np.asarray(context["ntl_observation"], dtype=np.float32),
                        "air_mask": stored_masks[0],
                        "space_mask": stored_masks[1],
                        "mode_mask": stored_masks[2],
                        "action": ntl_action,
                        "log_probability": log_probability,
                        "value": value,
                    },
                    "reward": 0.0,
                    "steps": 0,
                }
            else:
                ntl_action = NTLAction(NTL_NONE)
            next_ground_state, reward, terminated, truncated, info = env.step_hierarchical(
                ground_action,
                ntl_action,
                context=context,
            )
            done = terminated or truncated
            next_ground_mask = info["ground_action_mask"]
            loss = agent.ground.observe(
                ground_state,
                ground_action,
                reward,
                float(info["cost"]),
                next_ground_state,
                done,
                next_ground_mask,
            )
            if loss is not None:
                ground_losses.append(loss)
            if pending_ppo is not None:
                steps = int(pending_ppo["steps"])
                pending_ppo["reward"] += (agent.ppo.gamma**steps) * reward
                pending_ppo["steps"] = steps + 1
            if done and pending_ppo is not None:
                rollout.append(
                    PPOTransition(
                        **pending_ppo["transition"],
                        reward=float(pending_ppo["reward"]),
                        bootstrap_value=0.0,
                        bootstrap_discount=0.0,
                        trace_discount=0.0,
                    )
                )
                pending_ppo = None
            total_reward += reward
            ground_state, ground_mask = next_ground_state, next_ground_mask

        ppo_stats = agent.ppo.update(rollout)
        summary = env.summary()
        row: dict[str, Any] = {
            "episode": float(episode + 1),
            "phase": "joint-training",
            "reward": total_reward,
            "ground_loss": float(np.mean(ground_losses)) if ground_losses else float("nan"),
            "ppo_loss": ppo_stats.get("loss", float("nan")),
            "ppo_policy_loss": ppo_stats.get("policy_loss", float("nan")),
            "ppo_value_loss": ppo_stats.get("value_loss", float("nan")),
            "ppo_entropy": ppo_stats.get("entropy", float("nan")),
            "ppo_active_steps": ppo_stats.get("active_steps", 0.0),
            "epsilon": epsilon,
            **summary,
        }
        history.append(row)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if progress and ((episode + 1) == 1 or (episode + 1) % max(1, episodes // 10) == 0):
            print(
                f"episode={episode + 1}/{episodes} phase=joint-training reward={total_reward:.2f} "
                f"TCR={summary['tcr']:.1f}% gate={summary['gate_rate_pct']:.1f}% "
                f"replicas={summary['mean_replicas']:.2f}",
                flush=True,
            )
    return agent, history


def evaluate_callable(
    config: dict[str, Any],
    policy: Callable[[np.ndarray, SAGINEnv, np.random.Generator], int],
    seeds: list[int],
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        seed_everything(seed)
        env = SAGINEnv(config, seed=seed)
        state, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        decision_times: list[float] = []
        done = False
        while not done:
            start = time.perf_counter_ns()
            action = policy(state, env, rng)
            decision_times.append((time.perf_counter_ns() - start) / 1e6)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        row = {"seed": float(seed), **env.summary(), "decision_latency_ms": float(np.mean(decision_times))}
        rows.append(row)
    keys = [key for key in rows[0] if key != "seed"]
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for key in keys
    }
    return rows, aggregate


def evaluate_hierarchical_agent(
    config: dict[str, Any],
    agent: HierarchicalAgent,
    seeds: list[int],
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        seed_everything(seed)
        env = SAGINEnv(config, seed=seed)
        env.reset(seed=seed)
        decision_times: list[float] = []
        done = False
        while not done:
            start = time.perf_counter_ns()
            ground_action, ntl_action, context, _, _, _ = agent.decide(
                env,
                epsilon=0.0,
                deterministic_ntl=True,
            )
            decision_times.append((time.perf_counter_ns() - start) / 1e6)
            _, _, terminated, truncated, _ = env.step_hierarchical(ground_action, ntl_action, context=context)
            done = terminated or truncated
        row = {"seed": float(seed), **env.summary(), "decision_latency_ms": float(np.mean(decision_times))}
        rows.append(row)
    keys = [key for key in rows[0] if key != "seed"]
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for key in keys
    }
    return rows, aggregate


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
