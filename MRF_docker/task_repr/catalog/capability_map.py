"""任务功能到运行组件的候选映射，不执行参数或端口匹配。"""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ..errors import ValidationError
from .component_catalog import ComponentCatalog, default_component_catalog


_CAPABILITY_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class CapabilityMapping:
    cid: str
    capabilities: tuple[str, ...]
    match_condition: str


@dataclass(frozen=True)
class CapabilityMap:
    mappings: tuple[CapabilityMapping, ...]
    schema_version: int = 1

    def candidates_for(self, capability: str) -> tuple[CapabilityMapping, ...]:
        """按完整 capability ID 返回候选；不表示节点已经匹配成功。"""
        return tuple(mapping for mapping in self.mappings
                     if capability in mapping.capabilities)

    def by_component(self, cid: str) -> CapabilityMapping:
        for mapping in self.mappings:
            if mapping.cid == cid:
                return mapping
        raise KeyError(cid)


def load_capability_map(path: str | Path | None = None,
                        catalog: ComponentCatalog | None = None) -> CapabilityMap:
    """读取版本化映射表，并检查 CID 与组件目录的一致性。"""
    mapping_path = (Path(path) if path is not None else
                    Path(__file__).resolve().parents[1] / "data" / "capability_map.json")
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load capability map {mapping_path}: {exc}") from exc

    if not isinstance(raw, dict) or set(raw) != {"schema_version", "mappings"}:
        raise ValidationError("capability map must contain schema_version and mappings")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValidationError("capability map schema_version must be 1")
    entries = raw["mappings"]
    if not isinstance(entries, list) or not entries:
        raise ValidationError("capability map mappings must be a non-empty array")

    known_cids = {component.cid for component in
                  (catalog or default_component_catalog()).components}
    mappings = []
    seen_cids = set()
    for index, entry in enumerate(entries):
        label = f"mappings[{index}]"
        if not isinstance(entry, dict) or set(entry) != {
                "cid", "capabilities", "match_condition"}:
            raise ValidationError(f"{label} must contain cid, capabilities and match_condition")
        cid = entry["cid"]
        if not isinstance(cid, str) or cid not in known_cids:
            raise ValidationError(f"{label}.cid references an unknown component: {cid}")
        if cid in seen_cids:
            raise ValidationError(f"{label}.cid is duplicated: {cid}")
        seen_cids.add(cid)
        capabilities = entry["capabilities"]
        if (not isinstance(capabilities, list) or not capabilities or
                any(not isinstance(value, str) or not _CAPABILITY_ID.fullmatch(value)
                    for value in capabilities) or
                len(capabilities) != len(set(capabilities))):
            raise ValidationError(f"{label}.capabilities must contain unique capability IDs")
        condition = entry["match_condition"]
        if not isinstance(condition, str) or not condition.strip():
            raise ValidationError(f"{label}.match_condition must be non-empty text")
        mappings.append(CapabilityMapping(cid, tuple(capabilities), condition.strip()))
    return CapabilityMap(tuple(mappings), raw["schema_version"])


def default_capability_map() -> CapabilityMap:
    return load_capability_map()
