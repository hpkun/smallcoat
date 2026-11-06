"""Render a represented task's S/Q/R/G/X and functional DAG.

Run from the project root: python -m task_repr.visualize [input.json]
"""

import argparse
import heapq
import json
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

from . import represent_external_task, represent_external_tasks, represent_task


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "examples" / "BusinessRequests" / "voice_transmit_request.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "voice_task_representation.png"
ROLE_NAMES = {
    "audio_source": "语音输入", "voice_encoding": "语音编码",
    "channel_encoding": "信道编码", "digital_modulation": "数字调制",
    "rf_transmission": "射频发送",
}
INK = "#18252f"
MUTED = "#52616b"
LINE = "#d5dde1"
ACCENT = "#0f7568"


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = next((name for name in ("Microsoft YaHei", "Noto Sans SC", "SimHei")
                      if name in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = [preferred, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["svg.fonttype"] = "none"


def load_result(path: Path) -> dict[str, Any]:
    """Accept a business request, a saved representation result, or task JSON."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("input must be a JSON object")
    if document.get("schema_version") == "external_rf_task.v1":
        return represent_external_task(document)
    if "task_representation" in document:
        return document
    if "schema_version" in document and "graph" in document:
        return {"task_representation": represent_task(document).to_dict()}
    if {"S", "Q", "R", "G", "X"} <= document.keys():
        return {"task_representation": document}
    raise ValueError("input must be a business request or a represented task")


def load_results(path: Path) -> list[dict[str, Any]]:
    """按任务族模板为批量请求生成逐任务的绘图输入。"""
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list) or (isinstance(document, dict) and "tasks" in document):
        batch = represent_external_tasks(document)
    elif isinstance(document, dict) and "task_representations" in document:
        batch = document
    else:
        return [load_result(path)]
    templates = batch["task_templates"]
    return [{"task_representation": task,
             "task_template": templates[task["S"]["family"]],
             "template_id": templates[task["S"]["family"]]["template_id"],
             "status": batch["status"]}
            for task in batch["task_representations"]]


def _layers(graph: dict) -> list[list[dict]]:
    nodes = graph["nodes"]
    node_id = lambda node: node.get("node_id", node.get("id"))
    by_id = {node_id(node): node for node in nodes}
    if len(by_id) != len(nodes) or not nodes:
        raise ValueError("DAG requires unique nodes")
    order = {node_id(node): index for index, node in enumerate(nodes)}
    indegree = {node_id: 0 for node_id in by_id}
    successors = {node_id: [] for node_id in by_id}
    depth = {node_id: 0 for node_id in by_id}
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in by_id or target not in by_id:
            raise ValueError("DAG edge references an unknown node")
        successors[source].append(target)
        indegree[target] += 1
    ready = [order[node_id] for node_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        current_id = node_id(nodes[heapq.heappop(ready)])
        visited += 1
        for target in successors[current_id]:
            depth[target] = max(depth[target], depth[current_id] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, order[target])
    if visited != len(nodes):
        raise ValueError("DAG contains a cycle")
    layers = [[] for _ in range(max(depth.values()) + 1)]
    for node in nodes:
        layers[depth[node_id(node)]].append(node)
    return layers


def _number(value: Any, unit: str = "") -> str:
    return "未指定" if value is None else f"{value:g}{unit}" if isinstance(value, (int, float)) else str(value)


def _summary_lines(task: dict) -> list[tuple[str, list[str]]]:
    s, q, r, x = (task[key] for key in ("S", "Q", "R", "X"))
    if task["G"].get("abstraction_level") == "business_function":
        constraints = [f"{item['metric']}: {item['comparator']} {item['value']} {item['unit']} ({item['scope']})"
                       for item in q["quality_constraints"]]
        periods = [f"{item['kind']}: {item['interval_ms']} ms"
                   for item in q.get("periodic_requirements", [])]
        return [
            ("S  任务语义", [f"{s['family']} / {s['task_type']} / {s['direction']}",
                             f"优先级 {s['base_priority']}  输入 {s['data_input']['kind']}"]),
            ("Q  服务约束", [f"到达 {_number(q['arrival_ms'], ' ms')}  持续 {_number(q['duration_ms'], ' ms')}  截止 {_number(q['deadline_ms'], ' ms')}",
                             *constraints, *periods]),
            ("R  资源需求", [f"估算状态 {r['estimation_status']}",
                             "CPU / 内存 / DSP / FPGA / 带宽 / RF：待估算"]),
            ("X  动态状态", [f"状态 {x['status']}  优先级 {x['current_priority']}",
                             f"进度 {_number(100 * x['progress'], '%')}  剩余时限 {_number(x['remaining_deadline_ms'], ' ms')}"]),
        ]
    nodes = task["G"]["nodes"]
    source = next((node["parameters"].get("source_ref") for node in nodes
                   if "source_ref" in node.get("parameters", {})), None)
    extension = s.get("extension") or {}
    semantics = [
        f"{s['family']} / {s['task_type']} / {s['direction']}  ·  优先级 {s['base_priority']}",
        f"场景 {s['scenario']}",
    ]
    if source:
        semantics.append(f"音频源 {source}")
    if extension.get("modulation_coding"):
        semantics.append(f"通信参数 {extension['modulation_coding']}")
    return [
        ("S  任务语义", semantics),
        ("Q  服务约束", [
            f"到达 {_number(q['arrival_ms'], ' ms')}  ·  截止 {_number(q['deadline_ms'], ' ms')}",
            f"持续 {_number(q.get('duration_ms'), ' ms')}  ·  最大时延 {_number(q.get('max_latency_ms'), ' ms')}",
            f"最低吞吐 {_number(q.get('min_throughput_mbps'), ' Mbps')}",
        ]),
        ("R  资源提示 · 演示估计", [
            f"CPU {_number(r['cpu_millicores'], ' m')}  ·  内存 {_number(r['memory_mib'], ' MiB')}",
            f"DSP {_number(r['dsp_units'])}  ·  FPGA {_number(r['fpga_units'])}  ·  RF 通道 {_number(r.get('rf_channels'))}",
            f"带宽 {_number(r['bandwidth_mbps'], ' Mbps')}",
        ]),
        ("X  动态状态", [
            f"状态 {x['status']}  ·  当前优先级 {x['current_priority']}",
            f"进度 {_number(100 * x['progress'], '%')}  ·  剩余时限 {_number(x['remaining_deadline_ms'], ' ms')}",
            f"观测时刻 {_number(x['observed_at_ms'], ' ms')}",
        ]),
    ]


def _draw_summary(ax, task: dict) -> None:
    abstract = task["G"].get("abstraction_level") == "business_function"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.5, 0.5], [0.06, 0.96], color=LINE, lw=1)
    ax.plot([0.01, 0.99], [0.5, 0.5], color=LINE, lw=1)
    for index, (heading, lines) in enumerate(_summary_lines(task)):
        x = 0.025 if index % 2 == 0 else 0.525
        y = 0.96 if index < 2 else 0.46
        ax.text(x, y, heading, ha="left", va="top", fontsize=11.5,
                fontweight="bold", color=ACCENT, transform=ax.transAxes)
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=55 if abstract else 47,
                                         break_long_words=True) or [""])
        ax.text(x, y - 0.10, "\n".join(wrapped), ha="left", va="top",
                fontsize=8.5 if abstract else 9.5, color=INK,
                linespacing=1.25 if abstract else 1.65,
                transform=ax.transAxes)


def _draw_graph(ax, graph: dict, layers: list[list[dict]]) -> None:
    max_columns = max(len(layer) for layer in layers)
    width = max(11.5, max_columns * 4.3)
    height = len(layers) * 2.15 + 0.5
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_title("G  功能依赖 DAG", loc="left", fontsize=13, fontweight="bold",
                 color=INK, pad=18)
    positions = {}
    node_width, node_height = 3.7, 1.36
    abstract = graph.get("abstraction_level") == "business_function"
    for level, layer in enumerate(layers):
        y = height - 1.25 - level * 2.15
        for column, node in enumerate(layer):
            x = width / 2 + (column - (len(layer) - 1) / 2) * 4.3
            positions[node["id"] if abstract else node["node_id"]] = (x, y)

    by_id = {(node["id"] if abstract else node["node_id"]): node
             for layer in layers for node in layer}
    for edge in graph["edges"]:
        start_x, start_y = positions[edge["source"]]
        end_x, end_y = positions[edge["target"]]
        start = (start_x, start_y - node_height / 2 - 0.04)
        end = (end_x, end_y + node_height / 2 + 0.10)
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                lw=1.5, color="#7d919b", zorder=1,
                                connectionstyle="arc3,rad=0.04")
        ax.add_patch(arrow)
        data_format = ("" if abstract else
                       by_id[edge["source"]]["outputs"].get(edge["source_port"], ""))
        label_x = (start_x + end_x) / 2 + (0.18 if start_x == end_x else 0)
        label_y = (start[1] + end[1]) / 2
        ax.text(label_x, label_y, data_format, fontsize=8.5, color=MUTED,
                ha="center" if start_x != end_x else "left", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
                zorder=3)

    for index, node in enumerate(graph["nodes"], start=1):
        identifier = node["id"] if abstract else node["node_id"]
        x, y = positions[identifier]
        is_boundary = abstract or not node["inputs"] or not node["outputs"]
        patch = FancyBboxPatch((x - node_width / 2, y - node_height / 2),
                               node_width, node_height,
                               boxstyle="round,pad=0.015,rounding_size=0.12",
                               facecolor="#f0f7f5" if is_boundary else "#ffffff",
                               edgecolor=ACCENT if is_boundary else LINE,
                               linewidth=1.35, zorder=2)
        ax.add_patch(patch)
        left = x - node_width / 2 + 0.22
        role = node.get("role") or node["node_id"]
        title = ROLE_NAMES.get(role, role.replace("_", " "))
        if abstract:
            title = textwrap.fill(title, width=25)
        ax.text(left, y + 0.38, f"{index:02d}  {title}", fontsize=9.5 if abstract else 11.5,
                fontweight="bold", color=INK, ha="left", va="center", zorder=3)
        ax.text(left, y + 0.04, "公共业务功能" if abstract else node["capability"], fontsize=9.5,
                color=ACCENT, ha="left", va="center", zorder=3)
        detail = (f"节点 {identifier}  /  未绑定组件" if abstract else
                  f"节点 {identifier}    估计 {_number(node['estimated_runtime_ms'], ' ms')}")
        ax.text(left, y - 0.33, detail,
                fontsize=8.4, color=MUTED, ha="left", va="center", zorder=3)


def draw_figure(result: dict[str, Any]):
    task = result["task_representation"]
    layers = _layers(task["G"])
    abstract = task["G"].get("abstraction_level") == "business_function"
    _configure_font()
    fig = plt.figure(figsize=(13.5, (4.9 if abstract else 4.3) + len(layers) * 1.85),
                     facecolor="#ffffff")
    grid = fig.add_gridspec(2, 1, height_ratios=[4.2 if abstract else 3.2,
                                                len(layers) * 2.05],
                            left=0.055, right=0.945, top=0.91, bottom=0.04,
                            hspace=0.12)
    fig.suptitle(f"任务表征  /  {task['task_id']}", x=0.055, y=0.975, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    template_id = result.get("template_id")
    if template_id:
        fig.text(0.945, 0.975, f"模板 {template_id}  ·  未匹配组件",
                 ha="right", va="top", fontsize=10, color=MUTED)
    summary = fig.add_subplot(grid[0])
    graph = fig.add_subplot(grid[1])
    _draw_summary(summary, task)
    _draw_graph(graph, task["G"], layers)
    fig.text(0.055, 0.015, "功能需求与资源提示；未进行组件匹配或任务执行。",
             fontsize=8.5, color=MUTED)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化任务表征及功能 DAG")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="业务请求、六任务集合、已处理结果或任务 JSON")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help="输出 PNG、SVG 或 PDF 路径")
    parser.add_argument("--show", action="store_true", help="保存后打开绘图窗口")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("output extension must be .png, .svg, or .pdf")
    results = load_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        task_id = result["task_representation"]["task_id"]
        output = (args.output.with_name(f"{args.output.stem}_{task_id}{args.output.suffix}")
                  if len(results) > 1 else args.output)
        figure = draw_figure(result)
        figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
        print(f"任务 {task_id}：图已保存到 {output}")
        if args.show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
