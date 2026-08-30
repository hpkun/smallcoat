from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .agent import D3QNAgent
from .environment import SAGINEnv


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


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)

