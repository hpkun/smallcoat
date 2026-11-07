"""按硬件和组件配置保存实测执行画像，单位不与仿真等级混用。"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PhysicalUsage:
    cpu_millicores: float = 0.0
    memory_bytes: float = 0.0
    bandwidth_bytes_per_s: float = 0.0
    dsp_units: float = 0.0
    fpga_units: float = 0.0


def _validate_usage(usage: PhysicalUsage) -> None:
    for name, value in asdict(usage).items():
        _number(value, name)


@dataclass(frozen=True)
class PhysicalHardwareProfile:
    hardware_id: str
    capacity: PhysicalUsage
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.hardware_id, str) or not self.hardware_id.strip() or
                not isinstance(self.features, tuple) or
                any(not isinstance(item, str) or not item for item in self.features) or
                len(set(self.features)) != len(self.features)):
            raise ValueError("hardware_id and features must be valid")
        _validate_usage(self.capacity)


@dataclass(frozen=True)
class PhysicalResourceState:
    hardware_id: str
    used: PhysicalUsage

    def __post_init__(self) -> None:
        if not isinstance(self.hardware_id, str) or not self.hardware_id.strip():
            raise ValueError("state hardware_id must be nonempty")
        _validate_usage(self.used)


@dataclass(frozen=True)
class ComponentMeasurement:
    cid: str
    implementation_id: str
    processor: str
    hardware_id: str
    required_features: tuple[str, ...]
    parameters: dict
    demand: PhysicalUsage
    runtime_p95_ms: float
    setup_p95_ms: float
    input_bytes: int
    output_bytes: int
    sample_count: int
    source: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in
               (self.cid, self.implementation_id, self.hardware_id, self.source)):
            raise ValueError("measurement identifiers and source must be nonempty")
        if self.processor not in ("cpu", "dsp", "fpga"):
            raise ValueError("measurement processor must be cpu, dsp or fpga")
        if (not isinstance(self.parameters, dict) or
                not isinstance(self.required_features, tuple) or
                any(not isinstance(value, str) or not value for value in self.required_features) or
                len(set(self.required_features)) != len(self.required_features)):
            raise ValueError("measurement parameters and required_features must be valid")
        _validate_usage(self.demand)
        primary = {"cpu": self.demand.cpu_millicores, "dsp": self.demand.dsp_units,
                   "fpga": self.demand.fpga_units}[self.processor]
        if primary <= 0 or (self.processor != "dsp" and self.demand.dsp_units) or (self.processor != "fpga" and self.demand.fpga_units):
            raise ValueError("measurement has inconsistent processor demand")
        _number(self.runtime_p95_ms, "runtime_p95_ms", positive=True)
        _number(self.setup_p95_ms, "setup_p95_ms")
        for name in ("input_bytes", "output_bytes", "sample_count"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "sample_count" else 0):
                raise ValueError(f"{name} must be a nonnegative integer")

    def applies_to(self, cid: str, parameters: dict,
                   hardware: PhysicalHardwareProfile) -> bool:
        return (self.cid == cid and self.parameters == parameters and
                self.hardware_id == hardware.hardware_id and
                set(self.required_features) <= set(hardware.features))


@dataclass(frozen=True)
class MeasurementCatalog:
    profiles: tuple[ComponentMeasurement, ...] = ()

    def candidates(self, cid: str, parameters: dict,
                   hardware: PhysicalHardwareProfile) -> tuple[ComponentMeasurement, ...]:
        return tuple(item for item in self.profiles if item.applies_to(cid, parameters, hardware))


def _number(value, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _usage(raw: dict) -> PhysicalUsage:
    keys = set(PhysicalUsage.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"demand must contain exactly {sorted(keys)}")
    return PhysicalUsage(**{key: _number(raw[key], key) for key in keys})


def load_measurements(path: str | Path) -> MeasurementCatalog:
    """只接收有来源、样本数和精确参数的标定记录。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "profiles"} or type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or not isinstance(raw["profiles"], list):
        raise ValueError("measurement catalog requires schema_version 1 and profiles array")
    required = set(ComponentMeasurement.__dataclass_fields__)
    profiles = []
    for index, row in enumerate(raw["profiles"]):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"profiles[{index}] has missing or unknown fields")
        for key in ("cid", "implementation_id", "hardware_id", "source"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"profiles[{index}].{key} must be nonempty")
        if type(row["processor"]) is not str or row["processor"] not in ("cpu", "dsp", "fpga"):
            raise ValueError(f"profiles[{index}].processor must be cpu, dsp or fpga")
        if (not isinstance(row["required_features"], list) or
                any(not isinstance(item, str) or not item for item in row["required_features"]) or
                len(set(row["required_features"])) != len(row["required_features"])):
            raise ValueError(f"profiles[{index}].required_features must be strings")
        if not isinstance(row["parameters"], dict):
            raise ValueError(f"profiles[{index}].parameters must be an object")
        demand = _usage(row["demand"])
        primary = {"cpu": demand.cpu_millicores, "dsp": demand.dsp_units,
                   "fpga": demand.fpga_units}[row["processor"]]
        if primary <= 0 or (row["processor"] != "dsp" and demand.dsp_units) or (row["processor"] != "fpga" and demand.fpga_units):
            raise ValueError(f"profiles[{index}] has inconsistent processor demand")
        times = {key: _number(row[key], key, positive=key == "runtime_p95_ms")
                 for key in ("runtime_p95_ms", "setup_p95_ms")}
        for key in ("input_bytes", "output_bytes", "sample_count"):
            if type(row[key]) is not int or row[key] < (1 if key == "sample_count" else 0):
                raise ValueError(f"profiles[{index}].{key} must be a nonnegative integer")
        profiles.append(ComponentMeasurement(
            cid=row["cid"], implementation_id=row["implementation_id"],
            processor=row["processor"], hardware_id=row["hardware_id"],
            required_features=tuple(row["required_features"]), parameters=row["parameters"],
            demand=demand, input_bytes=row["input_bytes"], output_bytes=row["output_bytes"],
            sample_count=row["sample_count"], source=row["source"], **times))
    return MeasurementCatalog(tuple(profiles))


def load_physical_hardware(path: str | Path) -> PhysicalHardwareProfile:
    """读取带明确单位的设备容量，不从 0–100 等级换算。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "hardware_id", "capacity", "features"} or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("physical hardware requires schema_version 1, hardware_id, capacity and features")
    if not isinstance(raw["hardware_id"], str) or not raw["hardware_id"].strip():
        raise ValueError("hardware_id must be nonempty")
    features = raw["features"]
    if (not isinstance(features, list) or any(not isinstance(item, str) or not item for item in features)
            or len(set(features)) != len(features)):
        raise ValueError("features must be unique nonempty strings")
    return PhysicalHardwareProfile(raw["hardware_id"], _usage(raw["capacity"]), tuple(features))
