import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from matplotlib import image

from task_repr.visualize import _layers, draw_figure, load_result, load_results


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "examples" / "BusinessRequests" / "voice_transmit_request.json"
ISAC = ROOT / "examples" / "TaskDemos" / "isac_joint_task.json"
EXTERNAL = ROOT / "examples" / "BusinessRequests" / "External"


class TaskVisualizationTests(unittest.TestCase):
    def test_voice_request_and_saved_result_render_the_same_dag(self):
        result = load_result(VOICE)
        graph = result["task_representation"]["G"]
        self.assertEqual([len(layer) for layer in _layers(graph)], [1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "processed.json"
            saved.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_result(saved), json.loads(json.dumps(result)))
            figure = draw_figure(result)
            png = Path(directory) / "task.png"
            svg = Path(directory) / "task.svg"
            try:
                figure.savefig(png, dpi=100)
                figure.savefig(svg)
            finally:
                from matplotlib import pyplot as plt
                plt.close(figure)
            pixels = image.imread(png)
            self.assertGreater(float(pixels[:, :, :3].std()), 0.02)
            markup = svg.read_text(encoding="utf-8")
            self.assertIn("accept voice service", markup)
            self.assertIn("未绑定组件", markup)
            self.assertNotIn("C14", markup)

    def test_existing_branching_task_uses_two_nodes_in_final_layer(self):
        result = load_result(ISAC)
        self.assertEqual([len(layer) for layer in
                          _layers(result["task_representation"]["G"])], [1, 1, 2])

    def test_external_requests_render_abstract_templates(self):
        results = load_results(EXTERNAL / "all_tasks.json")
        self.assertEqual(len(results), 7)
        self.assertEqual(len({result["template_id"] for result in results}), 3)
        with tempfile.TemporaryDirectory() as directory:
            for result in results:
                graph = result["task_representation"]["G"]
                self.assertEqual(sum(map(len, _layers(graph))), len(graph["nodes"]))
                figure = draw_figure(result)
                output = Path(directory) / f"{result['task_representation']['task_id']}.svg"
                try:
                    figure.savefig(output)
                finally:
                    from matplotlib import pyplot as plt
                    plt.close(figure)
                markup = output.read_text(encoding="utf-8")
                role = graph["nodes"][0]["role"].replace("_", " ")
                for line in textwrap.wrap(role, width=25):
                    self.assertIn(line, markup)
                self.assertIn(result["template_id"], markup)


if __name__ == "__main__":
    unittest.main()
