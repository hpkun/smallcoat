"""验证业务规格图与资源规划器的输入契约。"""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from backend.services.tasks import TaskService
from matching import match_task
from resources import HardwareProfile
from resources.estimator import ResourceEstimator
from resources.planner import ResourcePlanner, ResourceUsage
from scheduler import TaskScheduler
from task_repr import parse_external_task, refine_business_task


EXAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests"
           / "External" / "07_spectrum_analysis.json")
LEGACY_EXAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"
                  / "signal_analysis_task.json")


def business_task(deadline_ms=None):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if deadline_ms is not None:
        raw["deadline_ms"] = deadline_ms
    refinement = refine_business_task(parse_external_task(raw))
    return raw, refinement.task, match_task(refinement)


class CalibratedEstimator(ResourceEstimator):
    def estimate(self, definition, parameters, hardware=None):
        return replace(super().estimate(definition, parameters, hardware), runtime_ms=2.0)


class SpecificationResourcePlanTests(unittest.TestCase):
    def test_matched_business_dag_reports_capacity_without_inventing_time(self):
        _, task, match = business_task()
        self.assertEqual(match.status, "MATCHED")
        plan = ResourcePlanner().plan(task, match)
        self.assertEqual(plan.status, "UNVERIFIED_TIME")
        self.assertEqual(len(plan.allocations), 3)
        self.assertIsNone(plan.estimated_serial_ms)
        self.assertIsNone(plan.time_limit_ms)
        self.assertIn("task has no explicit completion deadline", plan.issues)

    def test_explicit_deadline_and_runtime_estimates_are_checked(self):
        _, task, match = business_task(deadline_ms=8)
        uncalibrated = ResourcePlanner().plan(task, match)
        self.assertEqual(uncalibrated.status, "UNVERIFIED_TIME")
        self.assertEqual(uncalibrated.time_limit_ms, 8)
        self.assertIsNone(uncalibrated.estimated_serial_ms)
        planner = ResourcePlanner(estimator=CalibratedEstimator())
        plan = planner.plan(task, match)
        self.assertEqual(plan.status, "FEASIBLE_ESTIMATE")
        self.assertEqual((plan.estimated_serial_ms, plan.time_limit_ms), (6.0, 8))
        short = replace(task, X={**task.X, "remaining_deadline_ms": 5})
        plan = planner.plan(short, match)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertIn("estimated serial runtime exceeds task time limit", plan.issues)

    def test_expired_deadline_is_infeasible_without_runtime_estimate(self):
        _, task, match = business_task(deadline_ms=8)
        expired = replace(task, X={**task.X, "remaining_deadline_ms": 0,
                                   "observed_at_ms": 8})
        plan = ResourcePlanner().plan(expired, match)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertIn("task completion deadline has expired", plan.issues)

    def test_current_resource_usage_reduces_available_capacity(self):
        _, task, match = business_task()
        plan = ResourcePlanner().plan(task, match,
                                      HardwareProfile(cpu_capacity=10),
                                      ResourceUsage(cpu=5))
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertEqual(plan.available.cpu, 5)
        self.assertIn("cpu demand exceeds available normalized capacity", plan.issues)

    def test_matched_status_requires_bindings_for_every_specification_node(self):
        _, task, match = business_task()
        incomplete = replace(match, bindings={})
        with self.assertRaisesRegex(ValueError, "one binding for every task node"):
            ResourcePlanner().plan(task, incomplete)

    def test_service_accepts_external_and_normalized_business_task(self):
        raw, task, _ = business_task()
        service = TaskService(Mock(), scheduler=TaskScheduler())
        for document in (raw, parse_external_task(raw)):
            with self.subTest(schema=document["schema_version"]):
                plan = service.plan(document)
                self.assertEqual(plan["status"], "UNVERIFIED_TIME")
                self.assertEqual(len(plan["allocations"]), 3)
        self.assertEqual(service.simulation_state()["active_tasks"], [])
        self.assertEqual(service.plan({**raw, "task_type": "spectrum_analysis",
                                       "request": {**raw["request"], "processing": {
                                           **raw["request"]["processing"],
                                           "profile": "unknown"}}})["status"], "UNMATCHED")

    def test_service_uses_current_scheduler_resource_state(self):
        raw, _, _ = business_task()
        scheduler = TaskScheduler(HardwareProfile(cpu_capacity=10))
        service = TaskService(Mock(), scheduler=scheduler)
        legacy = json.loads(LEGACY_EXAMPLE.read_text(encoding="utf-8"))
        service.simulate_schedule(legacy)
        self.assertEqual(service.simulation_state()["used"]["cpu"], 10)
        plan = service.plan(raw)
        self.assertEqual(plan["status"], "INFEASIBLE")
        self.assertEqual(plan["available"]["cpu"], 0)
        self.assertIn("cpu demand exceeds available normalized capacity", plan["issues"])


if __name__ == "__main__":
    unittest.main()
