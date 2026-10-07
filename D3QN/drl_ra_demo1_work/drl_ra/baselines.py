from __future__ import annotations

from typing import Callable, Any

import numpy as np


Policy = Callable[[np.ndarray, list, Any, np.random.Generator], int]


def random_policy(mask: np.ndarray, candidates: list, task: Any, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(mask)))


def nearest_policy(mask: np.ndarray, candidates: list, task: Any, rng: np.random.Generator) -> int:
    feasible = [candidate for candidate in candidates if mask[candidate.action] and candidate.layer != "local"]
    if not feasible:
        feasible = [candidate for candidate in candidates if mask[candidate.action]]
    return int(min(feasible, key=lambda item: item.distance_km).action)


def reliable_policy(mask: np.ndarray, candidates: list, task: Any, rng: np.random.Generator) -> int:
    feasible = [candidate for candidate in candidates if mask[candidate.action]]
    return int(max(feasible, key=lambda item: (item.reliability, -item.delay_s)).action)


def ra_opt_policy(mask: np.ndarray, candidates: list, task: Any, rng: np.random.Generator) -> int:
    """RA-Opt-inspired single-task enumerative approximation.

    The published MIP jointly allocates CPU across a batch of tasks. This
    baseline evaluates the feasible one-task subproblem, so it is a reference
    policy rather than a full CVXPY/ECOS_BB reproduction.
    """
    feasible = [candidate for candidate in candidates if mask[candidate.action]]
    reliable = [candidate for candidate in feasible if candidate.reliability >= task.reliability_required]
    pool = reliable or feasible
    return int(min(pool, key=lambda item: 0.6 * item.delay_s + 0.4 * item.energy_mj / 1000.0).action)


POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "greedy-nearest": nearest_policy,
    "greedy-reliability": reliable_policy,
    "ra-opt": ra_opt_policy,
}
