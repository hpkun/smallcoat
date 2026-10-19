"""32 个软件化波形组件的机器可读目录。

组件库把功能接口和第一版调度仿真标定放在同一条记录中，供任务模板、组件组合
和资源调度代码复用。这里保存的是轻量化组件契约，不包含 IQ 样本、固件二进制
或设备实测日志。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..models.component import ComponentDefinition, ComponentSimulationProfile
from ..errors import ValidationError
from ..models.resource import ResourceLevels


_LATENCY_LEVELS = {f"L{level}": level for level in range(6)}
_PRIMARY_RESOURCES = {"CPU", "DSP", "FPGA", "NIC"}
_SHARING_CLASSES = {"S", "P", "X"}
_PREEMPTION_CLASSES = {"Y", "C", "N"}


@dataclass(frozen=True)
class ComponentCatalog:
    """组件定义集合，提供按 CID 查询及基本一致性校验。"""

    components: tuple[ComponentDefinition, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        ids = [component.cid for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValidationError("component catalog CID values must be unique")

    def __len__(self) -> int:
        return len(self.components)

    def get(self, cid: str) -> ComponentDefinition:
        """按 CID 取得组件；未知 CID 明确报错而不是返回空值。"""
        for component in self.components:
            if component.cid == cid:
                return component
        raise KeyError(cid)

    def by_family(self, family: str) -> tuple[ComponentDefinition, ...]:
        """返回指定任务族可用的组件，并包含所有通用组件。"""
        return tuple(component for component in self.components
                     if family in component.families or "通用" in component.families)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典，便于生成实验快照。"""
        return {
            "schema_version": self.schema_version,
            "components": [_component_to_dict(component) for component in self.components],
        }


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValidationError(f"{path} must be a non-empty array of strings")
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise ValidationError(f"{path} must not contain duplicate values")
    return result


def _object(value: Any, path: str, required: set[str], optional: set[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValidationError(f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _finite_number(value: Any, path: str, *, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValidationError(f"{path} must be a finite number >= {minimum}")
    return float(value)


def _resource_levels(value: Any, path: str) -> ResourceLevels:
    entry = _object(value, path, {"cpu", "mem", "dsp", "fpga", "bw"})
    levels: dict[str, int] = {}
    for name in ("cpu", "mem", "dsp", "fpga", "bw"):
        item = entry[name]
        if type(item) is not int or not 0 <= item <= 5:
            raise ValidationError(f"{path}.{name} must be an integer in 0..5")
        levels[name] = item
    return ResourceLevels(**levels)


def _latency_level(value: Any, path: str) -> str:
    latency = _text(value, path)
    parts = latency.split("-")
    if (len(parts) not in (1, 2) or
            any(part not in _LATENCY_LEVELS for part in parts) or
            len(parts) == 2 and _LATENCY_LEVELS[parts[0]] > _LATENCY_LEVELS[parts[1]]):
        raise ValidationError(f"{path} must be L0-L5 or an ascending range such as L0-L1")
    return latency


def _simulation_profile(value: Any, path: str) -> ComponentSimulationProfile:
    entry = _object(value, path, {"resource_levels", "latency_level", "sharing_class", "preemption_class"})
    sharing = _text(entry["sharing_class"], f"{path}.sharing_class")
    if sharing not in _SHARING_CLASSES:
        raise ValidationError(f"{path}.sharing_class must be S, P, or X")
    preemption = _text(entry["preemption_class"], f"{path}.preemption_class")
    if preemption not in _PREEMPTION_CLASSES:
        raise ValidationError(f"{path}.preemption_class must be Y, C, or N")
    return ComponentSimulationProfile(
        resource_levels=_resource_levels(entry["resource_levels"], f"{path}.resource_levels"),
        latency_level=_latency_level(entry["latency_level"], f"{path}.latency_level"),
        sharing_class=sharing,
        preemption_class=preemption,
    )


def _component(value: Any, index: int) -> ComponentDefinition:
    path = f"components[{index}]"
    required = {
        "cid", "name", "short_name", "function", "input_formats", "output_formats",
        "configurable_parameters", "families", "typical_predecessors", "typical_successors",
        "primary_resources", "estimated_runtime_ms", "runtime_status", "simulation_profile",
    }
    entry = _object(value, path, required)
    cid = _text(entry["cid"], f"{path}.cid")
    if len(cid) != 3 or cid[0] != "C" or not cid[1:].isdigit():
        raise ValidationError(f"{path}.cid must use the C01 form")
    primary = _text_list(entry["primary_resources"], f"{path}.primary_resources")
    unknown_primary = set(primary) - _PRIMARY_RESOURCES
    if unknown_primary:
        raise ValidationError(f"{path}.primary_resources has unknown values {sorted(unknown_primary)}")
    runtime = entry["estimated_runtime_ms"]
    if runtime is not None:
        runtime = _finite_number(runtime, f"{path}.estimated_runtime_ms", minimum=0)
        if runtime == 0:
            raise ValidationError(f"{path}.estimated_runtime_ms must be > 0 when provided")
    return ComponentDefinition(
        cid=cid,
        name=_text(entry["name"], f"{path}.name"),
        short_name=_text(entry["short_name"], f"{path}.short_name"),
        function=_text(entry["function"], f"{path}.function"),
        input_formats=_text_list(entry["input_formats"], f"{path}.input_formats"),
        output_formats=_text_list(entry["output_formats"], f"{path}.output_formats"),
        configurable_parameters=_text_list(entry["configurable_parameters"], f"{path}.configurable_parameters"),
        families=_text_list(entry["families"], f"{path}.families"),
        typical_predecessors=_text_list(entry["typical_predecessors"], f"{path}.typical_predecessors", allow_empty=True),
        typical_successors=_text_list(entry["typical_successors"], f"{path}.typical_successors", allow_empty=True),
        primary_resources=primary,
        estimated_runtime_ms=runtime,
        runtime_status=_text(entry["runtime_status"], f"{path}.runtime_status"),
        simulation_profile=_simulation_profile(entry["simulation_profile"], f"{path}.simulation_profile"),
    )


def load_component_catalog(path: str | Path | None = None) -> ComponentCatalog:
    """读取并严格校验组件库；不传路径时读取内置 32 组件 JSON。"""
    catalog_path = (Path(path) if path is not None else
                    Path(__file__).resolve().parents[1] / "data" / "components.json")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load component catalog {catalog_path}: {exc}") from exc
    root = _object(raw, "catalog", {"schema_version", "components"})
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValidationError("catalog.schema_version must be 1")
    if not isinstance(root["components"], list) or not root["components"]:
        raise ValidationError("catalog.components must be a non-empty array")
    components = tuple(_component(item, index) for index, item in enumerate(root["components"]))
    catalog = ComponentCatalog(components, root["schema_version"])
    if len(catalog) != 32:
        raise ValidationError(f"built-in component catalog must contain 32 components, got {len(catalog)}")
    return catalog


def _component_to_dict(component: ComponentDefinition) -> dict[str, Any]:
    """把 dataclass 恢复为与 JSON 兼容的结构，便于导出版本化快照。"""
    profile = component.simulation_profile
    return {
        "cid": component.cid,
        "name": component.name,
        "short_name": component.short_name,
        "function": component.function,
        "input_formats": list(component.input_formats),
        "output_formats": list(component.output_formats),
        "configurable_parameters": list(component.configurable_parameters),
        "families": list(component.families),
        "typical_predecessors": list(component.typical_predecessors),
        "typical_successors": list(component.typical_successors),
        "primary_resources": list(component.primary_resources),
        "estimated_runtime_ms": component.estimated_runtime_ms,
        "runtime_status": component.runtime_status,
        "simulation_profile": {
            "resource_levels": {
                "cpu": profile.resource_levels.cpu,
                "mem": profile.resource_levels.mem,
                "dsp": profile.resource_levels.dsp,
                "fpga": profile.resource_levels.fpga,
                "bw": profile.resource_levels.bw,
            },
            "latency_level": profile.latency_level,
            "sharing_class": profile.sharing_class,
            "preemption_class": profile.preemption_class,
        },
    }


def default_component_catalog() -> ComponentCatalog:
    """返回内置组件库。每次调用重新读取，避免调用方修改共享全局对象。"""
    return load_component_catalog()


__all__ = [
    "ComponentCatalog",
    "ComponentDefinition",
    "ComponentSimulationProfile",
    "default_component_catalog",
    "load_component_catalog",
]
