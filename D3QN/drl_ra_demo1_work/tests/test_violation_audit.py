from dataclasses import dataclass
from itertools import combinations
import unittest

import numpy as np

from drl_ra.redundancy import combined_reliability
from drl_ra.violation_audit import classify_violation, summarize_violations


@dataclass(frozen=True)
class Node:
    action: int
    reliability: float
    available: bool = True
    energy_mj: float = 0.0


class ViolationAuditTests(unittest.TestCase):
    def test_no_backup_below_three_replicas(self):
        nodes = [Node(0, 0.1), Node(1, 0.2), Node(2, 1.0, False)]
        row = classify_violation(nodes, [0, 1], 0.95, "infeasible")
        self.assertEqual(row["violation_category"], "no_backup_infeasible")
        self.assertEqual(row["num_remaining_backups"], 0)
        self.assertAlmostEqual(row["max_reliability_all"], 0.28)

    def test_three_replica_intrinsic_infeasibility(self):
        row = classify_violation([Node(i, 0.5) for i in range(4)], [0, 1, 2], 0.95, "max_replica_stop")
        self.assertEqual(row["violation_category"], "max3_intrinsic_infeasible")
        self.assertAlmostEqual(row["max_reliability_all"], 0.875)

    def test_three_replica_placement_failure_has_a_primary_preserving_witness(self):
        nodes = [Node(0, 0.1), Node(1, 0.2), Node(2, 0.8), Node(3, 0.8)]
        row = classify_violation(nodes, [0, 1, 2], 0.95, "max_replica_stop")
        self.assertEqual(row["violation_category"], "max3_placement_failure")
        self.assertEqual(row["best_given_primary_actions"], [0, 2, 3])
        self.assertGreaterEqual(row["max_reliability_given_primary"], 0.95)

    def test_global_solution_without_primary_solution_is_not_placement_failure(self):
        nodes = [Node(0, 0.1), Node(1, 0.8), Node(2, 0.8), Node(3, 0.8)]
        row = classify_violation(nodes, [0, 1, 2], 0.99, "max_replica_stop")
        self.assertEqual(row["violation_category"], "max3_primary_bottleneck")
        self.assertGreaterEqual(row["max_reliability_all"], 0.99)
        self.assertLess(row["max_reliability_given_primary"], 0.99)
        summary = summarize_violations([row])
        self.assertFalse(summary["requested_three_cover_all"])
        self.assertEqual(summary["uncovered_by_requested_three"], 1)

    def test_exact_requirement_is_not_a_violation(self):
        row = classify_violation([Node(i, 0.5) for i in range(3)], [0, 1, 2], 0.875, "max_replica_stop")
        self.assertEqual(row["violation"], 0)
        self.assertIsNone(row["violation_category"])

    def test_unreliable_early_stop_with_backup_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "early STOP"):
            classify_violation([Node(0, 0.1), Node(1, 0.8)], [0], 0.95, "infeasible")

    def test_sorted_reliability_optima_match_exhaustive_sets(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            nodes = [Node(i, float(value)) for i, value in enumerate(rng.uniform(0.0, 0.7, 5))]
            row = classify_violation(nodes, [0, 1, 2], 0.999, "max_replica_stop")
            sets = [combo for n in (1, 2, 3) for combo in combinations(nodes, n)]
            self.assertAlmostEqual(row["max_reliability_all"], max(combined_reliability(combo) for combo in sets))
            self.assertAlmostEqual(row["max_reliability_given_primary"], max(combined_reliability(combo) for combo in sets if nodes[0] in combo))

    def test_summary_uses_both_denominators_and_checks_exhaustiveness(self):
        rows = [
            {"violation": 1, "violation_category": "no_backup_infeasible"},
            {"violation": 1, "violation_category": "max3_intrinsic_infeasible"},
            {"violation": 1, "violation_category": "max3_placement_failure"},
            {"violation": 0, "violation_category": None},
        ]
        summary = summarize_violations(rows)
        self.assertTrue(summary["requested_three_cover_all"])
        self.assertEqual(summary["cvr_pct"], 75.0)
        self.assertEqual(summary["pct_all_tasks"]["no_backup_infeasible"], 25.0)
        self.assertAlmostEqual(summary["pct_violations"]["no_backup_infeasible"], 100.0 / 3)
        rows.append({"violation": 1, "violation_category": "unknown"})
        with self.assertRaisesRegex(RuntimeError, "partition"):
            summarize_violations(rows)


if __name__ == "__main__":
    unittest.main()
