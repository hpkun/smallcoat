import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from task_executor import DAGExecutionError, DAGExecutor
from task_repr import ComponentInstance, represent_task
from task_repr.models.task import FunctionGraph
from task_repr.runtime import DataPacket


TASK_PATH = (Path(__file__).resolve().parents[1] / "examples" / "TaskDemos" /
             "cni_video_backhaul_task.json")


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

    def test_missing_source_and_non_c08_binding_fail_explicitly(self):
        executor = fft_executor()
        with self.assertRaisesRegex(DAGExecutionError, "missing=\\['video-sync'\\]"):
            executor.execute({})
        self.assertEqual(executor.status, "failed")
        with self.assertRaisesRegex(DAGExecutionError, "only C08"):
            fft_executor(binding="C05")

    def test_invalid_c08_parameters_fail_before_execution(self):
        task = task_graph()
        invalid = ComponentInstance("bad-fft", "C08", task.task_id, {"fft_size": 1})
        executor = fft_executor(task, invalid)
        with self.assertRaisesRegex(DAGExecutionError, "cannot create node video-fft"):
            executor.execute({"video-sync": source_packet()})
        self.assertEqual(executor.status, "failed")


if __name__ == "__main__":
    unittest.main()
