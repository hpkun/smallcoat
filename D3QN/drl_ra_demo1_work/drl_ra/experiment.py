from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .agent import D3QNAgent, ReplicaD3QNAgent
from .environment import NTLAction, NTL_NONE, SAGINEnv
from .hierarchical import HierarchicalAgent
from .ppo import PPOTransition


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reliability_constrained_reward(base_reward: float, cost: float, lagrange: float) -> float:
    return float(base_reward - lagrange * cost)


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
    elif method == "drl-ra-learned-replica":
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


class LearnedReplicaAgent:
    """Two independent D3QN policies sharing one task-level environment."""

    def __init__(self, primary: D3QNAgent, replica: ReplicaD3QNAgent, seed: int) -> None:
        self.primary = primary
        self.replica = replica
        self.rng = np.random.default_rng(seed)
        self.last_stop_reason = "not_applicable"
        self.last_replica_initial_state: np.ndarray | None = None
        self.last_replica_initial_mask: np.ndarray | None = None

    @property
    def lagrange(self) -> float:
        return self.primary.lagrange

    def decide(
        self,
        env: SAGINEnv,
        primary_epsilon: float = 0.0,
        replica_epsilon: float | None = None,
    ) -> tuple[int, list[int], list[tuple[np.ndarray, int, np.ndarray, bool, np.ndarray]]]:
        replica_epsilon = primary_epsilon if replica_epsilon is None else replica_epsilon
        state = env._state_from_candidates(env.current_task, env.candidates)
        mask = np.asarray([candidate.available for candidate in env.candidates], dtype=bool)
        primary = self.primary.act(state, mask, epsilon=primary_epsilon)
        if not bool(env.candidates[primary].available):
            primary = 0
        selected = [int(primary)]
        self.last_replica_initial_state = env.learned_redundancy_observation(selected).copy()
        self.last_replica_initial_mask = env.learned_redundancy_action_mask(selected).copy()
        transitions: list[tuple[np.ndarray, int, np.ndarray, bool, np.ndarray]] = []
        max_replicas = int(env.env_cfg["max_replicas"])
        self.last_stop_reason = "not_applicable"
        while len(selected) < max_replicas:
            replica_state = env.learned_redundancy_observation(selected)
            replica_mask = env.learned_redundancy_action_mask(selected)
            action = self.replica.act(replica_state, replica_mask, epsilon=replica_epsilon)
            stop_action = env.action_dim
            if action == stop_action:
                transitions.append((replica_state, action, replica_state.copy(), True, replica_mask.copy()))
                self.last_stop_reason = (
                    "active_stop" if bool(replica_mask[:stop_action].any()) else "no_feasible_candidate_stop"
                )
                break
            next_selected = selected + [int(action)]
            terminal = len(next_selected) >= max_replicas
            next_state = env.learned_redundancy_observation(next_selected)
            next_mask = env.learned_redundancy_action_mask(next_selected)
            transitions.append((replica_state, action, next_state, terminal, next_mask))
            selected = next_selected
            if terminal:
                self.last_stop_reason = "max_replica_stop"
                break
        return primary, selected, transitions

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "primary": {
                    "online": self.primary.online.state_dict(),
                    "target": self.primary.target.state_dict(),
                    "optimizer": self.primary.optimizer.state_dict(),
                    "lagrange": self.primary.lagrange,
                },
                "replica": {
                    "online": self.replica.online.state_dict(),
                    "target": self.replica.target.state_dict(),
                    "optimizer": self.replica.optimizer.state_dict(),
                },
                "metadata": metadata or {},
            },
            output,
        )

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.primary.device, weights_only=False)
        for agent, data in ((self.primary, payload["primary"]), (self.replica, payload["replica"])):
            agent.online.load_state_dict(data["online"])
            agent.target.load_state_dict(data.get("target", data["online"]))
            if load_optimizer and "optimizer" in data:
                agent.optimizer.load_state_dict(data["optimizer"])
        self.primary.lagrange = float(payload["primary"].get("lagrange", self.primary.lagrange))
        return dict(payload.get("metadata", {}))


def build_learned_replica_agent(env: SAGINEnv, config: dict[str, Any], seed: int, device: str) -> LearnedReplicaAgent:
    primary_config = deepcopy(config)
    primary_config["environment"]["enable_redundancy"] = False
    primary = D3QNAgent(env.state_dim, env.action_dim, primary_config, seed, device=device)
    primary_checkpoint = config.get("learned_replica", {}).get("primary_checkpoint")
    if primary_checkpoint:
        primary.load(primary_checkpoint)
    replica = ReplicaD3QNAgent(env.learned_redundancy_state_dim, env.learned_redundancy_action_dim, config, seed + 1, device=device)
    return LearnedReplicaAgent(primary, replica, seed)


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
                f"CVR={summary['cvr']:.1f}% cost={summary['expected_cost']:.4f} "
                f"lambda={agent.lagrange:.3f}"
            )
    return agent, history


def train_learned_replica_agent(
    config: dict[str, Any], seed: int, device: str = "cpu", progress: bool = True
) -> tuple[LearnedReplicaAgent, list[dict[str, Any]]]:
    seed_everything(seed)
    env = SAGINEnv(config, seed=seed)
    agent = build_learned_replica_agent(env, config, seed, device)
    train_cfg = config["training"]
    epsilon = float(train_cfg["epsilon_start"])
    episodes = int(train_cfg["episodes"])
    epsilon_end = float(train_cfg["epsilon_end"])
    epsilon_decay = float(train_cfg["epsilon_decay"])
    replica_cfg = config.get("learned_replica", {})
    replica_epsilon = float(replica_cfg.get("epsilon_start", epsilon))
    replica_epsilon_end = float(replica_cfg.get("epsilon_end", epsilon_end))
    replica_epsilon_decay = float(replica_cfg.get("epsilon_decay", epsilon_decay))
    freeze_primary_episodes = int(replica_cfg.get("freeze_primary_episodes", 0))
    if freeze_primary_episodes > 0 and not replica_cfg.get("primary_checkpoint"):
        raise ValueError("freeze_primary_episodes requires learned_replica.primary_checkpoint")
    history: list[dict[str, Any]] = []
    for episode in range(episodes):
        env.reset(seed=seed * 10_000 + episode)
        total_reward = 0.0
        primary_losses: list[float] = []
        replica_losses: list[float] = []
        primary_q_values: list[float] = []
        replica_q_values: list[float] = []
        active_stops = 0
        max_replica_stops = 0
        no_feasible_candidate_stops = 0
        done = False
        pending_replica: tuple[list[tuple[np.ndarray, int, np.ndarray, bool, np.ndarray]], float] | None = None
        while not done:
            state = env._state_from_candidates(env.current_task, env.candidates)
            primary, selected, transitions = agent.decide(
                env, primary_epsilon=epsilon, replica_epsilon=replica_epsilon
            )
            primary_q_values.append(agent.primary.last_q_max)
            if transitions:
                replica_q_values.append(agent.replica.last_q_max)
            if pending_replica is not None:
                if agent.last_replica_initial_state is None or agent.last_replica_initial_mask is None:
                    raise RuntimeError("the next task's initial replica state was not captured")
                replica_losses.extend(
                    agent.replica.observe_sequence(
                        pending_replica[0],
                        pending_replica[1],
                        episode_done=False,
                        bootstrap_state=agent.last_replica_initial_state,
                        bootstrap_mask=agent.last_replica_initial_mask,
                    )
                )
                pending_replica = None
            active_stops += int(agent.last_stop_reason == "active_stop")
            max_replica_stops += int(agent.last_stop_reason == "max_replica_stop")
            no_feasible_candidate_stops += int(agent.last_stop_reason == "no_feasible_candidate_stop")
            next_state, reward, terminated, truncated, info = env.step_with_replicas(
                selected, stop_reason=agent.last_stop_reason
            )
            done = terminated or truncated
            constrained_reward = reliability_constrained_reward(
                reward, float(info["cost"]), agent.primary.lagrange
            )
            primary_loss = None
            if episode >= freeze_primary_episodes:
                primary_loss = agent.primary.observe(
                    state, primary, reward, float(info["cost"]), next_state, done, info["action_mask"]
                )
            if primary_loss is not None:
                primary_losses.append(primary_loss)
            if done:
                replica_losses.extend(
                    agent.replica.observe_sequence(
                        transitions,
                        constrained_reward,
                        episode_done=True,
                    )
                )
            else:
                pending_replica = (transitions, constrained_reward)
            total_reward += reward
        summary = env.summary()
        summary_aliases = {
            "TCR": summary.get("tcr", float("nan")),
            "CVR": summary.get("cvr", float("nan")),
            "replica_1_rate": summary.get("replica_1_pct", float("nan")),
            "replica_2_rate": summary.get("replica_2_pct", float("nan")),
            "replica_3_rate": summary.get("replica_3_pct", float("nan")),
        }
        history.append(
            {
                "episode": float(episode + 1),
                "reward": total_reward,
                "loss": float(np.mean(primary_losses)) if primary_losses else float("nan"),
                "replica_loss": float(np.mean(replica_losses)) if replica_losses else float("nan"),
                "epsilon": epsilon,
                "replica_epsilon": replica_epsilon,
                "lagrange": agent.primary.lagrange,
                "primary_d3qn_loss": float(np.mean(primary_losses)) if primary_losses else float("nan"),
                "replica_d3qn_loss": float(np.mean(replica_losses)) if replica_losses else float("nan"),
                "primary_q_max_mean": float(np.mean(primary_q_values)) if primary_q_values else float("nan"),
                "primary_q_max_std": float(np.std(primary_q_values)) if primary_q_values else float("nan"),
                "replica_q_max_mean": float(np.mean(replica_q_values)) if replica_q_values else float("nan"),
                "replica_q_max_std": float(np.std(replica_q_values)) if replica_q_values else float("nan"),
                "active_stop_rate": float(100.0 * active_stops / max(len(env._metrics), 1)),
                "max_replica_stop_rate": float(100.0 * max_replica_stops / max(len(env._metrics), 1)),
                "no_feasible_candidate_stop_rate": float(100.0 * no_feasible_candidate_stops / max(len(env._metrics), 1)),
                "forced_stop_rate": float(100.0 * max_replica_stops / max(len(env._metrics), 1)),
                **summary,
                **summary_aliases,
            }
        )
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        replica_epsilon = max(
            replica_epsilon_end, replica_epsilon * replica_epsilon_decay
        )
        if progress and ((episode + 1) == 1 or (episode + 1) % max(1, episodes // 10) == 0):
            print(
                f"episode={episode + 1}/{episodes} learned-replica reward={total_reward:.2f} "
                f"TCR={summary['tcr']:.1f}% CVR={summary['cvr']:.1f}% replicas={summary['mean_replicas']:.2f}",
                flush=True,
            )
    return agent, history


def evaluate_learned_replica_agent(
    config: dict[str, Any], agent: LearnedReplicaAgent, seeds: list[int]
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
            _, selected, _ = agent.decide(env, primary_epsilon=0.0, replica_epsilon=0.0)
            decision_times.append((time.perf_counter_ns() - start) / 1e6)
            _, _, terminated, truncated, _ = env.step_with_replicas(selected, stop_reason=agent.last_stop_reason)
            done = terminated or truncated
        rows.append({"seed": float(seed), **env.summary(), "decision_latency_ms": float(np.mean(decision_times))})
    keys = [key for key in rows[0] if key != "seed"]
    aggregate = {key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0} for key in keys}
    return rows, aggregate


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
                        "critic_state": np.asarray(context["critic_state"], dtype=np.float32),
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
            ppo_reward = reliability_constrained_reward(
                reward,
                float(info["cost"]),
                agent.ground.lagrange,
            )
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
                pending_ppo["reward"] += (agent.ppo.gamma**steps) * ppo_reward
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
            "ground_lagrange": agent.ground.lagrange,
            **summary,
        }
        history.append(row)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if progress and ((episode + 1) == 1 or (episode + 1) % max(1, episodes // 10) == 0):
            print(
                f"episode={episode + 1}/{episodes} phase=joint-training reward={total_reward:.2f} "
                f"TCR={summary['tcr']:.1f}% CVR={summary['cvr']:.1f}% "
                f"gate={summary['gate_rate_pct']:.1f}% "
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
