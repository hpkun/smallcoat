import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier
import unittest

from matching import match_task
from resources import HardwareProfile
from resources.planner import ResourcePlanner
from scheduler import SchedulingError, TaskScheduler
from task_repr import represent_task


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "TaskDemos" / "signal_analysis_task.json"


class PlanningAndSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.task = represent_task(self.raw)
        self.match = match_task(self.task)
        self.assertEqual(self.match.status, "MATCHED")

    def test_cpu_reference_plan_uses_sequential_peak_and_total_memory(self):
        plan = ResourcePlanner().plan(self.task, self.match)
        self.assertEqual(plan.status, "FEASIBLE_ESTIMATE")
        self.assertEqual((plan.total_demand.cpu, plan.total_demand.memory,
                          plan.total_demand.bandwidth), (10, 35, 50))
        self.assertEqual(plan.estimated_serial_ms, 4)
        self.assertEqual(plan.time_limit_ms, 80)
        self.assertEqual(len(plan.allocations), 3)

    def test_capacity_and_time_constraints_reject_plan(self):
        short = replace(self.task, Q=replace(self.task.Q, max_latency_ms=1))
        plan = ResourcePlanner().plan(
            short, self.match, HardwareProfile(cpu_capacity=5))
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertEqual(len(plan.issues), 2)

    def test_scheduler_reuses_warm_instances_and_retains_memory(self):
        scheduler = TaskScheduler()
        first = scheduler.submit(self.task, self.match)
        self.assertEqual(first.reused, ())
        self.assertEqual(scheduler.snapshot()["used"]["memory"], 35)
        scheduler.complete(self.task.task_id)
        self.assertEqual(scheduler.snapshot()["used"],
                         {"cpu": 0.0, "memory": 35.0, "bandwidth": 0.0})
        second = scheduler.submit(self.task, self.match)
        self.assertEqual(set(second.reused), set(first.instance_ids))
        self.assertEqual(second.instance_ids, first.instance_ids)

    def test_failed_submit_preserves_instance_state(self):
        scheduler = TaskScheduler(HardwareProfile(cpu_capacity=5))
        before = scheduler.snapshot()
        with self.assertRaises(SchedulingError):
            scheduler.submit(self.task, self.match)
        self.assertEqual(scheduler.snapshot(), before)

    def test_unrelated_warm_instance_is_evicted_for_memory(self):
        scheduler = TaskScheduler(HardwareProfile(memory_capacity=35))
        first = scheduler.submit(self.task, self.match)
        scheduler.complete(self.task.task_id)
        raw = copy.deepcopy(self.raw)
        raw["task_id"] = "signal-analysis-002"
        raw["graph"]["nodes"][0]["parameters"]["nco_frequency"] = 200000
        second_task = represent_task(raw)
        second_match = match_task(second_task)
        second = scheduler.submit(second_task, second_match)
        self.assertEqual(second.evicted, (first.instance_ids["ddc-node"],))
        self.assertEqual(set(second.reused), {"filter-node", "sensing-node"})
        self.assertEqual(scheduler.snapshot()["used"]["memory"], 35)

    def test_concurrent_submissions_cannot_overbook_capacity(self):
        scheduler = TaskScheduler(HardwareProfile(cpu_capacity=10))
        barrier = Barrier(2)

        def submit(index):
            task_id = f"parallel-{index}"
            task = replace(self.task, task_id=task_id)
            match = replace(self.match, task_id=task_id)
            barrier.wait()
            try:
                scheduler.submit(task, match)
                return "accepted"
            except SchedulingError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, (1, 2)))
        self.assertEqual(sorted(outcomes), ["accepted", "rejected"])
        self.assertEqual(scheduler.snapshot()["used"]["cpu"], 10)


if __name__ == "__main__":
    unittest.main()
