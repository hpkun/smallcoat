"""验证固定 CID 分配、依赖搜索及调度预留。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier
import unittest

from matching import match_task
from components.catalog import CapabilityMap, CapabilityMapping, default_capability_map
from resources import (ComponentMeasurement, MeasurementCatalog,
                       PhysicalHardwareProfile, PhysicalUsage)
from resources.placement import (EdgeTransfer, NetworkLink, PlacementPlanner,
                                 PlacementState, ProcessingNode)
from scheduler import SchedulingError, TaskScheduler
from task_repr import parse_external_task, refine_business_task, represent_task
from task_repr.template.refinement import SpecificationEdge


EXAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests"
           / "External" / "07_spectrum_analysis.json")


def measurement(binding, hardware_id, runtime):
    return ComponentMeasurement(
        binding.cid, f"{binding.cid}-{hardware_id}", "cpu", hardware_id,
        (), binding.parameters, PhysicalUsage(100, 1024, 1000, 0, 0),
        runtime, 0, 4096, 4096, 20, "test-measurement")


class PlacementSchedulerTests(unittest.TestCase):
    def setUp(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["deadline_ms"] = 20
        self.task = refine_business_task(parse_external_task(raw)).task
        self.match = match_task(self.task)
        self.assertEqual(self.match.status, "MATCHED")
        self.ids = tuple(node.node_id for node in self.task.G.nodes)
        self.nodes = tuple(ProcessingNode(key, PhysicalHardwareProfile(
            key, PhysicalUsage(500, 8192, 10000, 0, 0))) for key in ("node-a", "node-b"))
        self.links = (NetworkLink("node-a", "node-b", 1_000_000, 1),
                      NetworkLink("node-b", "node-a", 1_000_000, 1))
        self.transfers = tuple(EdgeTransfer(edge.source, edge.target, 1000, 1_000_000)
                               for edge in self.task.G.edges)
        profiles = tuple(measurement(binding, host, 2 if host == "node-a" else 1)
                         for host in ("node-a", "node-b")
                         for binding in self.match.bindings.values())
        self.planner = PlacementPlanner(self.nodes, self.links,
                                        MeasurementCatalog(profiles), self.transfers)
        self.state = PlacementState({key: PhysicalUsage() for key in self.planner.nodes},
                                    {key: 0.0 for key in self.planner.links})

    def test_fixed_spectrum_chain_maps_component_container_and_processing_node(self):
        fixed = dict(zip(self.ids, ("node-a", "node-b", "node-b")))
        plan = self.planner.plan(self.task, self.match, self.state, fixed)
        self.assertEqual(plan.status, "FEASIBLE_ESTIMATE")
        self.assertEqual([item.cid for item in plan.allocations], ["C03", "C05", "C27"])
        self.assertEqual([item.processing_node_id for item in plan.allocations], list(fixed.values()))
        self.assertEqual(plan.allocations[0].container_unit_id, f"{self.task.task_id}:{self.ids[0]}")
        self.assertEqual(plan.edge_transfer_ms[(self.ids[0], self.ids[1])], 2)
        self.assertEqual(plan.edge_transfer_ms[(self.ids[1], self.ids[2])], 0)
        self.assertEqual(plan.estimated_completion_ms, 6)
        self.assertEqual(plan.link_demand[("node-a", "node-b")], 1_000_000)

    def test_search_compares_execution_and_transfer_costs(self):
        fixed = self.planner.plan(self.task, self.match, self.state,
                                  {key: "node-a" for key in self.ids})
        optimal = self.planner.plan(self.task, self.match, self.state)
        self.assertEqual(fixed.estimated_completion_ms, 6)
        self.assertEqual(optimal.estimated_completion_ms, 3)
        self.assertTrue(all(item.processing_node_id == "node-b" for item in optimal.allocations))
        self.assertTrue(optimal.exhaustive)

    def test_component_candidate_handoff_and_search_limits(self):
        base = default_capability_map()
        duplicate = CapabilityMapping("C03", ("digital_down_conversion",), "第二个已验证候选")
        options = match_task(self.task, CapabilityMap((duplicate,) + base.mappings),
                             max_solutions=2)
        self.assertEqual(len(options.candidate_bindings), 2)
        self.assertTrue(options.candidates_exhaustive)
        self.assertEqual(self.planner.plan(self.task, options, self.state).status,
                         "FEASIBLE_ESTIMATE")
        truncated = match_task(self.task, CapabilityMap((duplicate, duplicate) + base.mappings),
                               max_solutions=2)
        self.assertFalse(truncated.candidates_exhaustive)
        self.assertEqual(self.planner.plan(self.task, truncated, self.state).status,
                         "SEARCH_LIMIT")
        self.assertEqual(self.planner.plan(self.task, self.match, self.state,
                                            max_assignments=1).status, "SEARCH_LIMIT")

    def test_search_chooses_second_statically_valid_cid(self):
        raw = json.loads((Path(__file__).resolve().parents[1] / "examples" / "TaskDemos"
                          / "signal_analysis_task.json").read_text(encoding="utf-8"))
        raw["graph"] = {"nodes": [{
            "id": "unit", "capability": "rf_iq_receive_simulation",
            "inputs": {"in": "iq/complex"}, "outputs": {"out": "iq/complex"},
            "estimated_runtime_ms": 1,
            "parameters": {"lo_offset_hz": 100, "gain_db": 1}}], "edges": []}
        task = represent_task(raw)
        mappings = CapabilityMap((
            CapabilityMapping("C33", ("rf_iq_receive_simulation",), "测试候选"),
            CapabilityMapping("C34", ("rf_iq_receive_simulation",), "测试候选")))
        match = match_task(task, mappings, max_solutions=3)
        self.assertEqual([item["unit"].cid for item in match.candidate_bindings],
                         ["C33", "C34"])
        second = match.candidate_bindings[1]["unit"]
        planner = PlacementPlanner(self.nodes, self.links, MeasurementCatalog((
            measurement(second, "node-b", 1),)), ())
        plan = planner.plan(task, match, self.state)
        self.assertEqual(plan.status, "FEASIBLE_ESTIMATE")
        self.assertEqual(plan.allocations[0].cid, "C34")

    def test_branch_dependencies_use_longest_path_and_validate_whole_graph(self):
        root, middle, end = self.task.G.nodes
        branch = replace(middle, node_id="branch")
        joined = replace(end, node_id="join")
        edges = (SpecificationEdge(root.node_id, "out", middle.node_id, "in"),
                 SpecificationEdge(root.node_id, "out", branch.node_id, "in"),
                 SpecificationEdge(middle.node_id, "out", joined.node_id, "in"),
                 SpecificationEdge(branch.node_id, "out", joined.node_id, "in"))
        graph = replace(self.task.G, nodes=(root, middle, branch, joined), edges=edges)
        task = replace(self.task, G=graph)
        bindings = {root.node_id: self.match.bindings[self.ids[0]],
                    middle.node_id: self.match.bindings[self.ids[1]],
                    branch.node_id: self.match.bindings[self.ids[1]],
                    joined.node_id: self.match.bindings[self.ids[2]]}
        match = replace(self.match, bindings=bindings,
                        edges=tuple((edge.source, edge.target) for edge in edges))
        transfers = tuple(EdgeTransfer(edge.source, edge.target, 100, 1_000_000)
                          for edge in edges)
        planner = PlacementPlanner(self.nodes, self.links, self.planner.measurements, transfers)
        plan = planner.plan(task, match, self.state,
                            {key: "node-a" for key in bindings})
        self.assertEqual(plan.estimated_completion_ms, 6)
        self.assertEqual(plan.status, "FEASIBLE_ESTIMATE")
        with self.assertRaisesRegex(ValueError, "complete matched DAG"):
            planner.plan(task, replace(match, edges=match.edges[:-1]), self.state)
        cyclic = replace(task, G=replace(graph, edges=edges + (
            SpecificationEdge(joined.node_id, "out", root.node_id, "in"),)))
        with self.assertRaisesRegex(ValueError, "acyclic"):
            planner.plan(cyclic, match, self.state)

    def test_missing_profile_link_capacity_and_deadline_are_diagnosed(self):
        missing = PlacementPlanner(self.nodes, self.links,
                                   self.planner.measurements, self.transfers[:-1])
        self.assertEqual(missing.plan(self.task, self.match, self.state).status,
                         "NEEDS_TRANSFER_PROFILE")
        fixed = dict(zip(self.ids, ("node-a", "node-b", "node-b")))
        slow = replace(self.task, X={**self.task.X, "remaining_deadline_ms": 5})
        plan = self.planner.plan(slow, self.match, self.state, fixed)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertIn("DAG completion time exceeds task time limit", plan.issues)
        busy = replace(self.state, link_used={**self.state.link_used,
                                              ("node-a", "node-b"): 1})
        plan = self.planner.plan(self.task, self.match, busy, fixed)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertTrue(any("link capacity" in issue for issue in plan.issues))
        limited = replace(self.nodes[0], hardware=replace(
            self.nodes[0].hardware, capacity=PhysicalUsage(50, 8192, 10000, 0, 0)))
        planner = PlacementPlanner((limited, self.nodes[1]), self.links,
                                   self.planner.measurements, self.transfers)
        plan = planner.plan(self.task, self.match, self.state,
                            {key: "node-a" for key in self.ids})
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertTrue(any("node capacity" in issue for issue in plan.issues))

    def test_missing_calibration_and_invalid_edge_data_are_diagnosed(self):
        empty = PlacementPlanner(self.nodes, self.links, MeasurementCatalog(),
                                 self.transfers)
        self.assertEqual(empty.plan(self.task, self.match, self.state).status,
                         "NEEDS_CALIBRATION")
        oversized = replace(self.transfers[0], bytes_per_call=5000)
        planner = PlacementPlanner(self.nodes, self.links, self.planner.measurements,
                                   (oversized,) + self.transfers[1:])
        plan = planner.plan(self.task, self.match, self.state)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertTrue(any("measured output or input" in issue for issue in plan.issues))
        no_deadline = replace(self.task, Q={**self.task.Q, "deadline_ms": None},
                              X={**self.task.X, "remaining_deadline_ms": None})
        self.assertEqual(self.planner.plan(no_deadline, self.match, self.state).status,
                         "UNVERIFIED_TIME")

    def test_preview_submit_complete_and_warm_memory(self):
        scheduler = TaskScheduler(placement_planner=self.planner)
        before = scheduler.placement_snapshot()
        preview = scheduler.preview_placement(self.task, self.match)
        self.assertEqual(preview.status, "FEASIBLE_ESTIMATE")
        self.assertEqual(scheduler.placement_snapshot(), before)
        plan = scheduler.submit_placement(self.task, self.match)
        self.assertEqual(scheduler.placement_snapshot()["active_tasks"], [self.task.task_id])
        self.assertEqual({item["cid"] for item in scheduler.placement_snapshot()["instances"]},
                         {"C03", "C05", "C27"})
        scheduler.complete_placement(self.task.task_id, keep_warm=True)
        state = scheduler.placement_snapshot()
        self.assertEqual(state["active_tasks"], [])
        self.assertEqual(state["node_used"]["node-b"]["memory_bytes"], 3072)
        self.assertEqual(state["node_used"]["node-b"]["cpu_millicores"], 0)
        for allocation in plan.allocations:
            scheduler.evict_placement_warm(allocation.container_unit_id)
        self.assertEqual(scheduler.placement_snapshot()["node_used"], before["node_used"])
        scheduler.submit_placement(self.task, self.match)
        scheduler.complete_placement(self.task.task_id)
        self.assertEqual(scheduler.placement_snapshot()["node_used"], before["node_used"])

    def test_link_reservation_and_rejected_submit_preserve_state(self):
        scheduler = TaskScheduler(placement_planner=self.planner)
        fixed = dict(zip(self.ids, ("node-a", "node-b", "node-b")))
        first = scheduler.submit_placement(self.task, self.match, fixed)
        self.assertEqual(scheduler.placement_snapshot()["link_used"]["node-a->node-b"],
                         1_000_000)
        second = replace(self.task, task_id="second-spectrum")
        second_match = replace(self.match, task_id=second.task_id)
        before = scheduler.placement_snapshot()
        preview = scheduler.preview_placement(second, second_match, fixed)
        self.assertEqual(preview.status, "INFEASIBLE")
        self.assertTrue(any("link capacity" in issue for issue in preview.issues))
        with self.assertRaisesRegex(SchedulingError, "link capacity"):
            scheduler.submit_placement(second, second_match, fixed)
        self.assertEqual(scheduler.placement_snapshot(), before)
        scheduler.complete_placement(self.task.task_id)
        self.assertEqual(scheduler.placement_snapshot()["link_used"]["node-a->node-b"], 0)
        short = replace(self.task, task_id="short-spectrum",
                        X={**self.task.X, "remaining_deadline_ms": 5})
        short_match = replace(self.match, task_id=short.task_id)
        before = scheduler.placement_snapshot()
        with self.assertRaisesRegex(SchedulingError, "completion time"):
            scheduler.submit_placement(short, short_match, fixed)
        self.assertEqual(scheduler.placement_snapshot(), before)

    def test_concurrent_submissions_do_not_overbook(self):
        limited = tuple(replace(item, hardware=replace(
            item.hardware, capacity=replace(
                item.hardware.capacity,
                cpu_millicores=300 if item.node_id == "node-b" else 0)))
            for item in self.nodes)
        planner = PlacementPlanner(limited, self.links, self.planner.measurements,
                                   self.transfers)
        scheduler = TaskScheduler(placement_planner=planner)
        barrier = Barrier(2)

        def submit(index):
            task = replace(self.task, task_id=f"parallel-{index}")
            match = replace(self.match, task_id=task.task_id)
            barrier.wait()
            try:
                scheduler.submit_placement(task, match)
                return "accepted"
            except SchedulingError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(submit, (1, 2)))
        self.assertEqual(sorted(outcomes), ["accepted", "rejected"])
        state = scheduler.placement_snapshot()
        self.assertEqual(len(state["active_tasks"]), 1)
        self.assertEqual(state["node_used"]["node-b"]["cpu_millicores"], 300)


if __name__ == "__main__":
    unittest.main()
