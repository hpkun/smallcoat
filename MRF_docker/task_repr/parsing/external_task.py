"""把 external_rf_task.v1 JSON 解析成未绑定的业务级 S/Q/R/G/X。"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from ..errors import ValidationError


BASE = Path(__file__).resolve().parent / "data"
RULES = json.loads((BASE / "business_function_rules.json").read_text(encoding="utf-8"))
TEMPLATE = json.loads((BASE / "unified_task_template.json").read_text(encoding="utf-8"))


class TaskInputError(ValidationError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise TaskInputError(message)


def _number(value: Any, name: str, *, strictly_positive=False) -> None:
    _check(type(value) in (int, float) and math.isfinite(value), f"{name} must be a finite number")
    _check(value > 0 if strictly_positive else value >= 0,
           f"{name} must be {'> 0' if strictly_positive else '>= 0'}")


def _nonempty(value: Any, name: str) -> None:
    _check(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")


def _resource_template() -> dict:
    return copy.deepcopy(TEMPLATE["R"])


def _graph(task: dict, rule: dict) -> dict:
    graph = copy.deepcopy(TEMPLATE["G"])
    direction = task["direction"]
    branches = rule["branches"]
    selected = ["main"] if "main" in branches else (["tx", "rx"] if direction == "tx_rx" else [direction])
    for branch in selected:
        steps = branches[branch]
        prev = None
        for index, role in enumerate(steps, 1):
            node_id = f"{branch}-{index:02d}"
            graph["nodes"].append({
                "id": node_id,
                "role": role,
                "canonical_role": rule.get("shared_roles", {}).get(role, role),
                "binding_status": "abstract_function_only"
            })
            if prev is not None:
                graph["edges"].append({"source": prev, "target": node_id})
            prev = node_id
    graph["generation_status"] = "abstract_only_requires_refinement"
    graph["unresolved_information"] = list(rule["open_questions"])
    if direction == "tx_rx" and task["data_input"]["kind"] == "application_audio_stream":
        # 示例只给出上行的业务音频数据引用；不能虚构下行接收源。
        graph["unresolved_information"].append("bidirectional_receive_stream_not_bound")
    return graph


def _read_path(obj: dict, path: str):
    value: Any = obj
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_voice_source(raw: dict) -> None:
    request = raw["request"]
    if raw["task_type"] != "voice_communication" or "audio_source" not in request:
        return
    audio = request["audio_source"]
    communication = request.get("communication")
    traffic = request["traffic"]
    _check(isinstance(audio, dict) and set(audio) == {
        "kind", "reference", "format", "sample_rate_hz", "channels"
    }, "request.audio_source has invalid fields")
    _check(isinstance(communication, dict) and set(communication) == {
        "data_rate_mbps", "modulation_coding", "link_quality"
    }, "request.communication has invalid fields")
    _check(isinstance(traffic, dict), "request.traffic must be an object")
    _check(audio["kind"] in {"pcm_buffer", "pcm_stream"},
           "request.audio_source.kind must be pcm_buffer or pcm_stream")
    _nonempty(audio["reference"], "request.audio_source.reference")
    _check(audio["format"] == "audio/pcm", "request.audio_source.format must be audio/pcm")
    for field in ("sample_rate_hz", "channels"):
        _number(audio[field], f"request.audio_source.{field}", strictly_positive=True)
        _check(type(audio[field]) is int, f"request.audio_source.{field} must be an integer")
    _number(traffic.get("payload_mib"), "request.traffic.payload_mib", strictly_positive=True)
    _number(communication["data_rate_mbps"], "request.communication.data_rate_mbps",
            strictly_positive=True)
    _nonempty(communication["modulation_coding"], "request.communication.modulation_coding")
    _number(communication["link_quality"], "request.communication.link_quality")
    _check(communication["link_quality"] <= 1,
           "request.communication.link_quality must be <= 1")
    data_input = raw["data_input"]
    _check(data_input["kind"] == "application_audio_stream" and
           data_input["stream_id"] == audio["reference"] and
           data_input.get("format") == audio["format"] and
           data_input.get("sample_rate_hz") == audio["sample_rate_hz"] and
           data_input.get("channels") == audio["channels"],
           "data_input must describe the same PCM source as request.audio_source")


def parse_task(raw: dict[str, Any]) -> dict[str, Any]:
    _check(isinstance(raw, dict), "task must be a JSON object")
    required = {
        "schema_version", "task_id", "family", "task_type", "arrival_ms", "priority",
        "duration_ms", "direction", "request", "performance", "data_input", "expected_output"
    }
    optional = {"deadline_ms", "scenario"}
    _check(set(raw) >= required, f"missing required fields: {sorted(required - set(raw))}")
    _check(set(raw) <= required | optional, f"unknown fields: {sorted(set(raw) - required - optional)}")
    _check(raw["schema_version"] == "external_rf_task.v1", "unsupported input schema_version")
    _nonempty(raw["task_id"], "task_id")
    task_type = raw["task_type"]
    _check(task_type in RULES["task_types"], f"unsupported task_type: {task_type}")
    rule = RULES["task_types"][task_type]
    _check(raw["family"] == rule["family"], f"{task_type} requires family={rule['family']}")
    _check(raw["direction"] in rule["valid_directions"], f"invalid direction for {task_type}")
    _number(raw["arrival_ms"], "arrival_ms")
    _number(raw["duration_ms"], "duration_ms", strictly_positive=True)
    _check(type(raw["priority"]) is int and 0 <= raw["priority"] <= 10,
           "priority must be an integer in [0,10] (current simulator convention)")
    _check(isinstance(raw["request"], dict), "request must be an object")
    _check(isinstance(raw["performance"], dict), "performance must be an object")
    _check(isinstance(raw["data_input"], dict), "data_input must be an object")
    _check(isinstance(raw["expected_output"], list) and raw["expected_output"],
           "expected_output must be a nonempty array")
    for field in ("kind", "stream_id", "binding_status"):
        _nonempty(raw["data_input"].get(field), f"data_input.{field}")
    for field in rule["required_request_fields"]:
        _check(field in raw["request"], f"{task_type}: request.{field} is required")
    _validate_voice_source(raw)
    _check(set(raw["performance"]) <= set(rule["quality_fields"]),
           f"{task_type}: unknown performance keys: {sorted(set(raw['performance']) - set(rule['quality_fields']))}")
    out = copy.deepcopy(TEMPLATE)
    out["task_id"] = raw["task_id"]
    out["S"] = {
        "family": rule["canonical_family"],
        "source_family": raw["family"],
        "task_type": task_type,
        "scenario": raw.get("scenario"),
        "direction": raw["direction"],
        "base_priority": raw["priority"],
        "business_parameters": copy.deepcopy(raw["request"]),
        "data_input": copy.deepcopy(raw["data_input"]),
        "expected_output": copy.deepcopy(raw["expected_output"])
    }
    q = out["Q"]
    q["arrival_ms"] = raw["arrival_ms"]
    q["duration_ms"] = raw["duration_ms"]
    deadline = raw.get("deadline_ms")
    if deadline is not None:
        _number(deadline, "deadline_ms")
        _check(deadline > raw["arrival_ms"], "deadline_ms must exceed arrival_ms")
        q["deadline_ms"] = deadline
        q["deadline_status"] = "specified_absolute_time"
    for key, value in raw["performance"].items():
        spec = rule["quality_fields"][key]
        if "property" in spec:
            _check(type(value) is bool, f"performance.{key} must be boolean")
            q["required_properties"][spec["property"]] = value
            continue
        if "context" in spec:
            _check(type(value) in (int, float) and math.isfinite(value), f"performance.{key} must be finite number")
            q["evaluation_conditions"][spec["context"]] = value
            continue
        _number(value, f"performance.{key}")
        if spec["unit"] == "ratio":
            _check(value <= 1, f"performance.{key} ratio must be <= 1")
        q["quality_constraints"].append({
            "metric": spec["metric"], "comparator": spec["comparator"],
            "value": value, "unit": spec["unit"], "scope": spec["scope"]
        })
    for source in rule.get("period_sources", []):
        interval = _read_path(raw, source["path"])
        if interval is not None:
            _number(interval, source["path"], strictly_positive=True)
            q["periodic_requirements"].append({
                "kind": source["kind"], "interval_ms": interval,
                "role": source["role"], "source_field": source["path"]
            })
    out["R"] = _resource_template()
    out["G"] = _graph(raw, rule)
    x = out["X"]
    x["observed_at_ms"] = raw["arrival_ms"]
    x["current_priority"] = raw["priority"]
    x["remaining_deadline_ms"] = (deadline - raw["arrival_ms"]) if deadline is not None else None
    return out


def parse_external_task(raw: dict[str, Any]) -> dict[str, Any]:
    """Return an unbound business-function representation for one external request."""
    return parse_task(raw)






