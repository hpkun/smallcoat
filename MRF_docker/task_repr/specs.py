"""Versioned, machine-readable protocol (OPS) and component interface (OCS) contracts."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ValidationError
from .runtime.ports import RuntimePort
from .runtime.packet import DataPacket


_DATA = Path(__file__).resolve().parent / "data"
_KINDS = set("biufc")
_PROPERTY_TYPES = {"string", "integer", "number", "boolean", "array"}


def _object(value: Any, path: str, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{path} must have fields {sorted(fields)}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string")
    return value


def _list(value: Any, path: str, allowed: set | None = None) -> tuple:
    if (not isinstance(value, list) or not value or
            any(type(item) not in (str, int) for item in value) or
            len(value) != len(set(value))):
        raise ValidationError(f"{path} must be a non-empty array without duplicates")
    if allowed is not None and not set(value) <= allowed:
        raise ValidationError(f"{path} has unsupported values")
    return tuple(value)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load specification {path}: {exc}") from exc


@dataclass(frozen=True)
class ProtocolSpec:
    id: str
    version: str
    domain: str
    dtype_kinds: tuple[str, ...]
    dimensions: tuple[int, ...]
    item_unit: str

    def check_port(self, port: RuntimePort) -> str | None:
        if (self.domain not in port.domains or
                not set(self.dtype_kinds) <= set(port.dtype_kinds) or
                not set(self.dimensions) <= set(port.dimensions) or
                self.item_unit != port.item_unit):
            return f"protocol {self.id} requires {self}, but runtime port {port.name} has {port}"
        return None

    def validate_packet(self, packet: DataPacket) -> str | None:
        """Check the OPS fields observable in a runtime packet."""
        if not isinstance(packet, DataPacket):
            return f"protocol {self.id} requires a DataPacket"
        if packet.domain != self.domain:
            return f"protocol {self.id} requires domain {self.domain}, got {packet.domain}"
        if packet.samples.dtype.kind not in self.dtype_kinds:
            return f"protocol {self.id} requires dtype kind in {self.dtype_kinds}, got {packet.samples.dtype.kind}"
        if packet.samples.ndim not in self.dimensions:
            return f"protocol {self.id} requires dimensions in {self.dimensions}, got {packet.samples.ndim}"
        if (not math.isfinite(packet.sample_rate) or packet.sample_rate <= 0 or
                not math.isfinite(packet.center_frequency) or
                not math.isfinite(packet.timestamp)):
            return f"protocol {self.id} requires finite context and a positive sample rate"
        if not np.all(np.isfinite(packet.samples)):
            return f"protocol {self.id} requires finite samples"
        return None


@dataclass(frozen=True)
class PortSpec:
    direction: str
    protocols: tuple[str, ...] = ()
    protocols_by_mode: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ComponentSpec:
    cid: str
    version: str
    ports: dict[str, PortSpec]
    properties: dict[str, str]
    required_properties: tuple[str, ...] = ()

    def check_required(self, values: dict[str, Any]) -> str | None:
        missing = set(self.required_properties) - set(values)
        return f"{self.cid} requires OCS properties {sorted(missing)}" if missing else None

    def check_property(self, name: str, value: Any) -> str | None:
        kind = self.properties.get(name)
        if kind is None:
            return f"{self.cid} has no OCS property {name}"
        valid = {
            "string": lambda: isinstance(value, str),
            "integer": lambda: type(value) is int,
            "number": lambda: type(value) in (int, float),
            "boolean": lambda: type(value) is bool,
            "array": lambda: isinstance(value, (list, tuple)),
        }[kind]()
        return None if valid else f"{self.cid}.{name} must be {kind}"

    def check_port(self, name: str, protocol_id: str, runtime_port: RuntimePort,
                   configuration: dict[str, Any] | None = None) -> str | None:
        port = self.ports.get(name)
        if port is None or port.direction != runtime_port.direction:
            return f"{self.cid} has no {runtime_port.direction} OCS port {name}"
        allowed = port.protocols
        if port.protocols_by_mode is not None:
            mode = (configuration or {}).get("mode")
            if not isinstance(mode, str) or mode not in port.protocols_by_mode:
                return f"{self.cid} has no OCS port profile for mode {mode!r}"
            allowed = port.protocols_by_mode[mode]
        if protocol_id not in allowed:
            return f"{self.cid}.{name} does not declare OPS {protocol_id}"
        return None


def load_protocol_specs(path: str | Path | None = None) -> dict[str, ProtocolSpec]:
    root = _object(_read(Path(path) if path is not None else _DATA / "ops.json"),
                   "OPS", {"schema_version", "protocols"})
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValidationError("OPS schema_version must be 1")
    if not isinstance(root["protocols"], list) or not root["protocols"]:
        raise ValidationError("OPS protocols must be a non-empty array")
    result = {}
    for index, raw in enumerate(root["protocols"]):
        path_name = f"OPS protocols[{index}]"
        entry = _object(raw, path_name,
                        {"id", "version", "domain", "dtype_kinds", "dimensions", "item_unit"})
        spec = ProtocolSpec(
            _text(entry["id"], f"{path_name}.id"),
            _text(entry["version"], f"{path_name}.version"),
            _text(entry["domain"], f"{path_name}.domain"),
            _list(entry["dtype_kinds"], f"{path_name}.dtype_kinds", _KINDS),
            _list(entry["dimensions"], f"{path_name}.dimensions", {1, 2}),
            _text(entry["item_unit"], f"{path_name}.item_unit"),
        )
        if spec.id in result:
            raise ValidationError(f"duplicate OPS id {spec.id}")
        result[spec.id] = spec
    return result


def load_component_specs(path: str | Path | None = None,
                         protocols: dict[str, ProtocolSpec] | None = None) -> dict[str, ComponentSpec]:
    protocols = protocols if protocols is not None else load_protocol_specs()
    root = _object(_read(Path(path) if path is not None else _DATA / "ocs.json"),
                   "OCS", {"schema_version", "components"})
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValidationError("OCS schema_version must be 1 or 2")
    if not isinstance(root["components"], list) or not root["components"]:
        raise ValidationError("OCS components must be a non-empty array")
    result = {}
    for index, raw in enumerate(root["components"]):
        path_name = f"OCS components[{index}]"
        if not isinstance(raw, dict):
            raise ValidationError(f"{path_name} must be an object")
        required = {"cid", "version", "ports", "properties"}
        if not required <= set(raw) or set(raw) - required - {"required_properties"}:
            raise ValidationError(f"{path_name} has missing or unknown fields")
        entry = raw
        cid = _text(entry["cid"], f"{path_name}.cid")
        if cid in result:
            raise ValidationError(f"duplicate OCS cid {cid}")
        if not isinstance(entry["ports"], dict) or set(entry["ports"]) != {"in", "out"}:
            raise ValidationError(f"{cid} OCS ports must include in and out")
        ports = {}
        for name, raw_port in entry["ports"].items():
            if not isinstance(raw_port, dict):
                raise ValidationError(f"{cid}.{name} must be an object")
            fields = set(raw_port)
            if fields == {"direction", "protocols"}:
                port = raw_port
                mode_profiles = None
            elif schema_version == 2 and fields == {"direction", "protocols_by_mode"}:
                port = raw_port
                raw_profiles = port["protocols_by_mode"]
                if not isinstance(raw_profiles, dict) or not raw_profiles or any(
                    not isinstance(mode, str) or not mode.strip() for mode in raw_profiles
                ):
                    raise ValidationError(f"{cid}.{name} OCS mode profiles are invalid")
                mode_profiles = {}
                for mode, raw_refs in raw_profiles.items():
                    refs = _list(raw_refs, f"{cid}.{name}.{mode}.protocols")
                    if not set(refs) <= set(protocols):
                        raise ValidationError(f"{cid}.{name}.{mode} references unknown OPS {sorted(set(refs) - set(protocols))}")
                    mode_profiles[mode] = refs
            else:
                raise ValidationError(f"{cid}.{name} must declare protocols or v2 protocols_by_mode")
            direction = "input" if name == "in" else "output"
            if port["direction"] != direction:
                raise ValidationError(f"{cid}.{name} direction must be {direction}")
            refs = ()
            if mode_profiles is None:
                refs = _list(port["protocols"], f"{cid}.{name}.protocols")
                if not set(refs) <= set(protocols):
                    raise ValidationError(f"{cid}.{name} references unknown OPS {sorted(set(refs) - set(protocols))}")
            ports[name] = PortSpec(direction, refs, mode_profiles)
        properties = entry["properties"]
        if not isinstance(properties, dict) or not properties or any(
            not isinstance(name, str) or not name or kind not in _PROPERTY_TYPES
            for name, kind in properties.items()
        ):
            raise ValidationError(f"{cid} OCS properties have invalid names or types")
        input_modes = ports["in"].protocols_by_mode
        output_modes = ports["out"].protocols_by_mode
        if (input_modes is None) != (output_modes is None) or (
            input_modes is not None and (set(input_modes) != set(output_modes) or
                                         properties.get("mode") != "string")
        ):
            raise ValidationError(f"{cid} OCS input and output mode profiles must agree")
        required_properties = entry.get("required_properties", [])
        if (not isinstance(required_properties, list) or
                any(not isinstance(name, str) for name in required_properties) or
                len(required_properties) != len(set(required_properties)) or
                not set(required_properties) <= set(properties)):
            raise ValidationError(f"{cid} OCS required_properties are invalid")
        result[cid] = ComponentSpec(cid, _text(entry["version"], f"{cid}.version"),
                                    ports, dict(properties), tuple(required_properties))
    return result
