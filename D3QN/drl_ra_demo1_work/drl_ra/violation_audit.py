from __future__ import annotations

from typing import Any, Sequence

from .redundancy import ReplicaCandidate, combined_reliability


REQUESTED_CATEGORIES = (
    "no_backup_infeasible",
    "max3_intrinsic_infeasible",
    "max3_placement_failure",
)
PRIMARY_BOTTLENECK = "max3_primary_bottleneck"
ALL_CATEGORIES = (*REQUESTED_CATEGORIES, PRIMARY_BOTTLENECK)


def classify_violation(
    candidates: Sequence[ReplicaCandidate],
    selected_actions: Sequence[int],
    required: float,
    stop_reason: str,
) -> dict[str, Any]:
    """Classify against the immutable snapshot used to select this task's nodes.

    Independent replica reliability increases monotonically with every node's
    reliability, so the top three nodes (or primary plus the top two backups)
    give the exact reliability optimum. No reward oracle is needed.
    """
    if not 1 <= len(selected_actions) <= 3 or len(set(selected_actions)) != len(selected_actions):
        raise ValueError("expected a set of one to three distinct replica actions")
    by_action = {item.action: item for item in candidates}
    selected = [by_action[action] for action in selected_actions]
    if not all(item.available for item in selected):
        raise ValueError("selected replicas must be available in the decision snapshot")
    achieved = combined_reliability(selected)
    row: dict[str, Any] = {
        "replica_actions": list(selected_actions), "primary_action": selected_actions[0],
        "replicas": len(selected), "required_reliability": required,
        "reliability": achieved, "violation": int(achieved < required),
        "stop_reason": stop_reason, "violation_category": None,
    }
    if not row["violation"]:
        return row
    feasible = sorted((item for item in candidates if item.available), key=lambda item: (-item.reliability, item.action))
    primary = selected[0]
    best_all = feasible[:3]
    best_primary = [primary, *[item for item in feasible if item.action != primary.action][:2]]
    max_all = combined_reliability(best_all)
    max_primary = combined_reliability(best_primary)
    remaining = [item for item in feasible if item.action not in selected_actions]
    row.update(
        num_feasible_candidates=len(feasible), num_remaining_backups=len(remaining),
        max_reliability_all=max_all, max_reliability_given_primary=max_primary,
        best_all_actions=[item.action for item in best_all],
        best_given_primary_actions=[item.action for item in best_primary],
        candidate_snapshot=[
            {"action": item.action, "available": bool(item.available), "reliability": item.reliability}
            for item in candidates
        ],
    )
    if len(selected) < 3:
        if remaining or stop_reason != "infeasible":
            raise RuntimeError("unreliable early STOP despite a legal backup, or inconsistent stop reason")
        category = "no_backup_infeasible"
    elif stop_reason != "max_replica_stop":
        raise RuntimeError("three replicas must finish with max_replica_stop")
    elif max_all < required:
        category = "max3_intrinsic_infeasible"
    elif max_primary >= required:
        category = "max3_placement_failure"
    else:
        # The user's three strict definitions do not cover this case.
        category = PRIMARY_BOTTLENECK
    row["violation_category"] = category
    return row


def summarize_violations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [row for row in rows if row["violation"]]
    counts = {category: sum(row["violation_category"] == category for row in violations) for category in ALL_CATEGORIES}
    if sum(counts.values()) != len(violations):
        raise RuntimeError("violation categories do not partition all violations")
    requested_total = sum(counts[category] for category in REQUESTED_CATEGORIES)
    return {
        "tasks": len(rows), "violation_tasks": len(violations),
        "cvr_pct": 100.0 * len(violations) / max(len(rows), 1),
        "counts": counts,
        "pct_all_tasks": {category: 100.0 * count / max(len(rows), 1) for category, count in counts.items()},
        "pct_violations": {category: 100.0 * count / max(len(violations), 1) for category, count in counts.items()},
        "requested_three_cover_all": requested_total == len(violations),
        "uncovered_by_requested_three": len(violations) - requested_total,
        "partition_verified": True,
    }
