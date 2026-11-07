"""验证物理资源画像的来源、适配和容量约束。"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from matching import match_task
from resources import (ComponentMeasurement, MeasurementCatalog,
                       PhysicalHardwareProfile, PhysicalResourceState, PhysicalUsage,
                       load_measurements, load_physical_hardware)
from resources.planner import ResourcePlanner
from task_repr import parse_external_task, refine_business_task


EXAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "BusinessRequests"
           / "External" / "07_spectrum_analysis.json")
EMPTY_MEASUREMENTS = (Path(__file__).resolve().parents[1] / "resources" / "data"
                      / "measurements.json")


def measurement(cid, parameters, processor="cpu", *, runtime=2,
                features=(), implementation_id=None):
    demand = PhysicalUsage(cpu_millicores=100 if processor == "cpu" else 10,
                           memory_bytes=2048, bandwidth_bytes_per_s=4096,
                           dsp_units=1 if processor == "dsp" else 0,
                           fpga_units=1 if processor == "fpga" else 0)
    return ComponentMeasurement(cid, implementation_id or f"{cid}-{processor}", processor,
                                "board-1", tuple(features), parameters, demand,
                                runtime, 1, 4096, 2048, 20, "lab-run-001")


class PhysicalResourcePlanTests(unittest.TestCase):
    def setUp(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["deadline_ms"] = 20
        self.task = refine_business_task(parse_external_task(raw)).task
        self.match = match_task(self.task)
        self.assertEqual(self.match.status, "MATCHED")
        self.hardware = PhysicalHardwareProfile(
            "board-1", PhysicalUsage(200, 8192, 10000, 1, 0), ("dsp-v1",))
        self.state = PhysicalResourceState("board-1", PhysicalUsage())
        self.profiles = tuple(measurement(binding.cid, binding.parameters)
                              for binding in self.match.bindings.values())

    def test_selects_compatible_dsp_and_uses_physical_units(self):
        first = self.task.G.nodes[0].node_id
        binding = self.match.bindings[first]
        accelerated = measurement(binding.cid, binding.parameters, "dsp",
                                  runtime=1, features=("dsp-v1",))
        profiles = MeasurementCatalog(self.profiles + (accelerated,))
        plan = ResourcePlanner().plan_physical(
            self.task, self.match, self.hardware, profiles,
            PhysicalResourceState("board-1", PhysicalUsage(cpu_millicores=50)))
        self.assertEqual(plan.status, "FEASIBLE_MEASURED_ESTIMATE")
        self.assertEqual(plan.allocations[0].measurement.processor, "dsp")
        self.assertEqual(plan.total_demand.dsp_units, 1)
        self.assertEqual(plan.total_demand.fpga_units, 0)
        self.assertEqual(plan.total_demand.memory_bytes, 6144)
        self.assertEqual(plan.available.cpu_millicores, 150)
        self.assertEqual(plan.estimated_serial_ms, 8)
        self.assertEqual(plan.to_dict()["unit_system"], "physical")
        self.assertEqual(plan.to_dict()["units"]["cpu_millicores"], "millicores")

    def test_unavailable_hardware_feature_rejects_accelerated_implementation(self):
        first = self.task.G.nodes[0].node_id
        binding = self.match.bindings[first]
        accelerated = measurement(binding.cid, binding.parameters, "dsp",
                                  features=("dsp-v2",))
        profiles = MeasurementCatalog((accelerated,) + self.profiles[1:])
        plan = ResourcePlanner().plan_physical(self.task, self.match, self.hardware, profiles, self.state)
        self.assertEqual(plan.status, "NEEDS_CALIBRATION")
        self.assertIn(first, plan.issues[0])

    def test_accelerator_capacity_selects_cpu_implementation(self):
        first = self.task.G.nodes[0].node_id
        binding = self.match.bindings[first]
        accelerated = measurement(binding.cid, binding.parameters, "dsp",
                                  runtime=1, features=("dsp-v1",))
        hardware = replace(self.hardware, capacity=replace(self.hardware.capacity, dsp_units=0))
        plan = ResourcePlanner().plan_physical(
            self.task, self.match, hardware,
            MeasurementCatalog(self.profiles + (accelerated,)), self.state)
        self.assertEqual(plan.status, "FEASIBLE_MEASURED_ESTIMATE")
        self.assertEqual(plan.allocations[0].measurement.processor, "cpu")
        self.assertEqual(plan.total_demand.dsp_units, 0)

    def test_builtin_empty_catalog_never_claims_physical_feasibility(self):
        plan = ResourcePlanner().plan_physical(
            self.task, self.match, self.hardware,
            load_measurements(EMPTY_MEASUREMENTS), self.state)
        self.assertEqual(plan.status, "NEEDS_CALIBRATION")
        self.assertEqual(len(plan.issues), len(self.task.G.nodes))

    def test_missing_exact_parameter_or_hardware_measurement_is_reported(self):
        changed = replace(self.profiles[0], parameters={"different": True})
        plan = ResourcePlanner().plan_physical(
            self.task, self.match, self.hardware,
            MeasurementCatalog((changed,) + self.profiles[1:]), self.state)
        self.assertEqual(plan.status, "NEEDS_CALIBRATION")
        self.assertIsNone(plan.total_demand)
        other = replace(self.hardware, hardware_id="board-2")
        self.assertEqual(ResourcePlanner().plan_physical(
            self.task, self.match, other,
            MeasurementCatalog(self.profiles),
            PhysicalResourceState("board-2", PhysicalUsage())).status, "NEEDS_CALIBRATION")

    def test_resource_state_must_match_hardware(self):
        with self.assertRaisesRegex(ValueError, "selected hardware"):
            ResourcePlanner().plan_physical(
                self.task, self.match, self.hardware, MeasurementCatalog(self.profiles),
                PhysicalResourceState("board-2", PhysicalUsage()))

    def test_capacity_and_deadline_are_checked_without_level_conversion(self):
        planner = ResourcePlanner()
        profiles = MeasurementCatalog(self.profiles)
        small = replace(self.hardware, capacity=PhysicalUsage(50, 8192, 10000, 1, 0))
        plan = planner.plan_physical(self.task, self.match, small, profiles, self.state)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertIn("cpu_millicores demand exceeds available physical capacity", plan.issues)
        short = replace(self.task, X={**self.task.X, "remaining_deadline_ms": 5})
        plan = planner.plan_physical(short, self.match, self.hardware, profiles, self.state)
        self.assertEqual(plan.status, "INFEASIBLE")
        self.assertIn("measured component time sum exceeds task time limit", plan.issues)

    def test_loader_requires_source_and_consistent_processor_demand(self):
        entry = self.profiles[0]
        row = {**entry.__dict__, "required_features": [], "demand": entry.demand.__dict__}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.json"
            path.write_text(json.dumps({"schema_version": 1, "profiles": [row]}), encoding="utf-8")
            self.assertEqual(len(load_measurements(path).profiles), 1)
            row["source"] = ""
            path.write_text(json.dumps({"schema_version": 1, "profiles": [row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                load_measurements(path)
            row["source"] = "lab-run-001"
            row["demand"]["dsp_units"] = 1
            path.write_text(json.dumps({"schema_version": 1, "profiles": [row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent processor demand"):
                load_measurements(path)

    def test_hardware_loader_uses_explicit_capacity_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware.json"
            path.write_text(json.dumps({"schema_version": 1, "hardware_id": "board-1",
                                        "features": ["dsp-v1"],
                                        "capacity": self.hardware.capacity.__dict__}), encoding="utf-8")
            self.assertEqual(load_physical_hardware(path), self.hardware)


if __name__ == "__main__":
    unittest.main()
