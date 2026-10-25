import json
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from task_executor import DAGExecutionError, DAGExecutor
from task_repr import ComponentInstance, match_task, represent_task
from task_repr.matching.matcher import ComponentBinding
from task_repr.component_models.common.digital_filter_component import DigitalFilterComponent
from task_repr.models.task import FunctionGraph
from task_repr.runtime import DataPacket


TASK_PATH = (Path(__file__).resolve().parents[1] / "examples" / "TaskDemos" /
             "cni_video_backhaul_task.json")
SIGNAL_PATH = TASK_PATH.with_name("signal_analysis_task.json")


def task_graph():
    return represent_task(json.loads(TASK_PATH.read_text(encoding="utf-8")))


def source_packet(domain="time"):
    samples = np.exp(2j * np.pi * 128_000 * np.arange(2048) / 2_048_000)
    return DataPacket(samples, 2_048_000, 100_000_000, 0, domain)


def fft_executor(task=None, binding="C08"):
    return DAGExecutor(task or task_graph(), {"video-fft": binding},
                       active_nodes={"video-sync", "video-fft"})


class DAGExecutorTests(unittest.TestCase):
    def test_existing_task_edge_feeds_c08_and_marks_scope(self):
        task = task_graph()
        executor = fft_executor(task)
        self.assertEqual(executor.status, "pending")
        result = executor.execute({"video-sync": source_packet()})
        self.assertTrue(result.completed)
        self.assertFalse(result.full_task_completed)
        self.assertEqual(result.execution_order, ("video-sync", "video-fft"))
        self.assertEqual(result.node_outputs["video-sync"].domain, "time")
        self.assertEqual(set(result.sink_outputs), {"video-fft"})
        spectrum = result.sink_outputs["video-fft"]
        self.assertEqual(spectrum.domain, "frequency")
        self.assertEqual(np.argmax(np.abs(spectrum.samples)), 128)
        self.assertEqual(result.instances["video-fft"].parameters["fft_size"], 2048)
        self.assertEqual(result.metrics["video-fft"].total_frames, 1)
        self.assertEqual(len(task.G.nodes), 6)

    def test_repeated_execution_uses_a_fresh_c08_component(self):
        executor = fft_executor()
        for _ in range(2):
            result = executor.execute({"video-sync": source_packet()})
            self.assertTrue(executor.completed)
            self.assertEqual(result.metrics["video-fft"].call_count, 1)

    def test_explicit_instance_overrides_node_parameters(self):
        task = task_graph()
        instance = ComponentInstance("custom-fft", "C08", task.task_id,
                                     {"fft_size": 2048, "shift": True})
        result = fft_executor(task, instance).execute({"video-sync": source_packet()})
        self.assertEqual(result.instances["video-fft"].instance_id, "custom-fft")
        self.assertTrue(result.sink_outputs["video-fft"].metadata["fft"]["shift"])

    def test_unselected_nodes_are_not_silently_marked_complete(self):
        task = task_graph()
        with self.assertRaisesRegex(DAGExecutionError, "video-sync needs a component binding"):
            DAGExecutor(task, {"video-fft": "C08"})
        with self.assertRaisesRegex(DAGExecutionError, "invalid active nodes"):
            DAGExecutor(task, {"video-fft": "C08"}, active_nodes={"missing"})
        fft = next(node for node in task.G.nodes if node.node_id == "video-fft")
        changed = replace(fft, inputs=fft.inputs | {"other": "iq/ofdm_symbols"})
        nodes = tuple(changed if node.node_id == fft.node_id else node for node in task.G.nodes)
        task = replace(task, G=FunctionGraph(nodes, task.G.edges))
        with self.assertRaisesRegex(DAGExecutionError, "single input/output"):
            fft_executor(task)

    def test_failure_identifies_c08_node_and_allows_retry(self):
        executor = fft_executor()
        with self.assertRaisesRegex(DAGExecutionError, "node video-fft .* failed"):
            executor.execute({"video-sync": source_packet("frequency")})
        self.assertEqual(executor.status, "failed")
        self.assertFalse(executor.completed)
        self.assertTrue(executor.execute({"video-sync": source_packet()}).completed)

    def test_missing_source_and_unknown_binding_fail_explicitly(self):
        executor = fft_executor()
        with self.assertRaisesRegex(DAGExecutionError, "missing=\\['video-sync'\\]"):
            executor.execute({})
        self.assertEqual(executor.status, "failed")
        with self.assertRaisesRegex(DAGExecutionError, "no runtime implementation"):
            fft_executor(binding="C99")

    def test_invalid_c08_parameters_fail_before_execution(self):
        task = task_graph()
        invalid = ComponentInstance("bad-fft", "C08", task.task_id, {"fft_size": 1})
        executor = fft_executor(task, invalid)
        with self.assertRaisesRegex(DAGExecutionError, "cannot create node video-fft"):
            executor.execute({"video-sync": source_packet()})
        self.assertEqual(executor.status, "failed")


def signal_task():
    return represent_task(json.loads(SIGNAL_PATH.read_text(encoding="utf-8")))


def signal_packet(*, domain="time", dtype=complex, rate=4_000_000, length=4096):
    samples = np.exp(2j * np.pi * 100_000 * np.arange(length) / rate)
    if dtype is float:
        samples = samples.real
    samples = samples.astype(dtype)
    return DataPacket(samples, rate, 100_000_000, 0, domain)


class MatchedDAGExecutorTests(unittest.TestCase):
    def make_executor(self):
        task = signal_task()
        match = match_task(task)
        self.assertEqual(match.status, "MATCHED", match.to_dict())
        return DAGExecutor.from_match(task, match)

    def test_signal_analysis_processes_three_components(self):
        executor = self.make_executor()
        result = executor.execute({"ddc-node": signal_packet()})
        self.assertTrue(result.full_task_completed)
        self.assertEqual(result.execution_order, ("ddc-node", "filter-node", "sensing-node"))
        self.assertEqual([result.instances[node].cid for node in result.execution_order],
                         ["C03", "C05", "C27"])
        self.assertEqual([result.node_outputs[node].item_count for node in result.execution_order],
                         [2048, 2048, 1024])
        self.assertEqual([result.node_outputs[node].sample_rate for node in result.execution_order],
                         [2_000_000] * 3)
        self.assertEqual([result.node_outputs[node].domain for node in result.execution_order],
                         ["time", "time", "psd"])
        self.assertEqual(set(result.metrics), set(result.execution_order))
        self.assertEqual(result.node_outputs["sensing-node"].samples.dtype.kind, "f")

    def test_rejects_source_protocol_dtype_rate_and_length(self):
        for packet, detail in (
            (signal_packet(domain="frequency"), "domain"),
            (signal_packet(dtype=float), "dtype"),
            (signal_packet(rate=2_000_000), "sample rate"),
            (signal_packet(length=2048), "block length"),
            (DataPacket(np.ones((2, 2048), dtype=complex), 4_000_000,
                        100_000_000, 0, "time"), "dimensions"),
        ):
            with self.subTest(detail=detail):
                with self.assertRaisesRegex(DAGExecutionError, f"ddc-node input: .*{detail}"):
                    self.make_executor().execute({"ddc-node": packet})

    def test_uses_normalized_match_parameters(self):
        raw = json.loads(SIGNAL_PATH.read_text(encoding="utf-8"))
        parameters = raw["graph"]["nodes"][0]["parameters"]
        parameters["nco_frequency_mhz"] = parameters.pop("nco_frequency") / 1_000_000
        task = represent_task(raw)
        match = match_task(task)
        self.assertEqual(match.status, "MATCHED", match.to_dict())
        executor = DAGExecutor.from_match(task, match)
        self.assertEqual(executor.instances["ddc-node"].parameters["nco_frequency"], 100_000)
        self.assertTrue(executor.execute({"ddc-node": signal_packet()}).completed)

    def test_rejects_partial_or_tampered_match_before_execution(self):
        task = signal_task()
        match = match_task(task)
        with self.assertRaisesRegex(DAGExecutionError, "MATCHED"):
            DAGExecutor.from_match(task, replace(match, status="PARTIAL"))
        with self.assertRaisesRegex(DAGExecutionError, "all task nodes and edges"):
            DAGExecutor.from_match(task, replace(match, edges=()))
        bad_bindings = dict(match.bindings)
        bad_bindings["ddc-node"] = ComponentBinding("C03", {"mode": "duc", "decimation": 1})
        with self.assertRaisesRegex(DAGExecutionError, "edge ddc-node->filter-node: sample rate"):
            DAGExecutor.from_match(task, replace(match, bindings=bad_bindings))

    def test_rejects_bad_component_output_at_handoff(self):
        original = DigitalFilterComponent.process
        for problem in ("domain", "dtype", "block length", "sample rate"):
            def bad_output(component, packet):
                output = original(component, packet)
                samples = output.samples
                domain = output.domain
                rate = output.sample_rate
                if problem == "domain":
                    domain = "frequency"
                elif problem == "dtype":
                    samples = samples.real
                elif problem == "block length":
                    samples = samples[:1024]
                else:
                    rate = 1_000_000
                return DataPacket(samples, rate, output.center_frequency,
                                  output.timestamp, domain, output.metadata)

            with self.subTest(problem=problem):
                executor = self.make_executor()
                with patch.object(DigitalFilterComponent, "process", bad_output):
                    with self.assertRaisesRegex(DAGExecutionError,
                                                f"filter-node output: .*{problem}"):
                        executor.execute({"ddc-node": signal_packet()})

    def test_fft_mode_ocs_is_checked_before_execution(self):
        raw = json.loads(SIGNAL_PATH.read_text(encoding="utf-8"))
        raw["graph"]["nodes"] = [{"id": "transform", "capability": "fft",
                                  "inputs": {"in": "iq/complex"},
                                  "outputs": {"out": "frequency/complex"},
                                  "estimated_runtime_ms": 1,
                                  "parameters": {"fft_size": 16, "input_block_length": 16}}]
        raw["graph"]["edges"] = []
        task = represent_task(raw)
        match = match_task(task)
        self.assertEqual(match.status, "MATCHED", match.to_dict())
        bad_bindings = {"transform": ComponentBinding("C08", {"fft_size": 16, "mode": "ifft"})}
        with self.assertRaisesRegex(DAGExecutionError, "does not declare OPS"):
            DAGExecutor.from_match(task, replace(match, bindings=bad_bindings))


if __name__ == "__main__":
    unittest.main()
