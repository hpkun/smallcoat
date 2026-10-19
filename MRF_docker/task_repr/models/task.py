"""从原始任务描述提取组件组合与资源调度需要的信息。"""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping

from ..errors import ValidationError
from .resource import ExecutionProfile, ResourceLevels, Resources, SimulationProfile

@dataclass(frozen=True)
class Semantics:
    """S：任务类别与场景；基础优先级用于区别运行时优先级。"""

    family: str
    task_type: str
    scenario: str
    direction: str
    base_priority: int
    scenario_stage: str | None = None
    subtype: str | None = None
    trigger: str | None = None
    extension: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class QoS:
    """Q：任务服务质量约束；到达与截止时间使用同一场景时钟。"""

    arrival_ms: float
    deadline_ms: float
    max_latency_ms: float | None
    min_throughput_mbps: float | None
    min_reliability: float | None
    quality_constraints: tuple["QualityConstraint", ...]
    duration_ms: float | None = None
    period_ms: float | None = None
    max_response_ms: float | None = None
    security_level: int | None = None
    scheduling: "SchedulingPolicy | None" = None


@dataclass(frozen=True)
class SchedulingPolicy:
    """区分任务可被抢占与任务可抢占低优先级工作。"""

    preemptible: bool
    migratable: bool
    degradable: bool
    shareable: bool
    reconfiguration_ms: float
    can_preempt: bool = False


@dataclass(frozen=True)
class QualityConstraint:
    """按指标、比较符、数值和单位表示任务特有的质量门限。"""

    metric: str
    comparator: str
    value: float
    unit: str


@dataclass(frozen=True)
class FunctionNode:
    """G 中的功能需求节点；参数用于区分同类功能的不同配置。"""

    node_id: str
    capability: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    estimated_runtime_ms: float
    parameters: dict[str, str | int | float | bool]
    role: str | None = None
    shareable: bool = False
    preemptible: bool = False
    execution_profiles: dict[str, ExecutionProfile] = field(default_factory=dict)
    simulation_profile: SimulationProfile | None = None


@dataclass(frozen=True)
class Dependency:
    """功能节点之间的有向数据依赖，端口格式需一致。"""

    source: str
    source_port: str
    target: str
    target_port: str


@dataclass(frozen=True)
class FunctionGraph:
    """G：一次有限处理过程中的功能节点与依赖边。"""

    nodes: tuple[FunctionNode, ...]
    edges: tuple[Dependency, ...]


@dataclass(frozen=True)
class DynamicState:
    """X：观测时刻的任务状态；剩余时限由绝对截止时间计算。"""

    status: str
    observed_at_ms: float
    progress: float
    current_priority: int
    remaining_deadline_ms: float
    remaining_work: float | None = None
    work_unit: str | None = None
    allocated_resources: Resources | None = None


@dataclass(frozen=True)
class TaskRepresentation:
    """统一输出 T_rep={S,Q,R,G,X}，并携带任务与模式版本标识。"""

    task_id: str
    S: Semantics
    Q: QoS
    R: Resources
    G: FunctionGraph
    X: DynamicState
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _object(value: Any, path: str, required: set[str], optional: set[str] | None = None) -> Mapping[str, Any]:
    """严格检查对象字段，防止新增调度约束在转换时被静默丢弃。"""
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValidationError(f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _number(value: Any, path: str, *, minimum: float = 0, integer: bool = False) -> float | int:
    """拒绝布尔值、无穷大、NaN 和负资源量等无效数值。"""
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < minimum:
        kind = "integer" if integer else "number"
        raise ValidationError(f"{path} must be a finite {kind} >= {minimum}")
    return value


def _optional_number(value: Any, path: str, *, minimum: float = 0, integer: bool = False) -> float | int | None:
    return None if value is None else _number(value, path, minimum=minimum, integer=integer)


def _ports(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object of port names and data formats")
    return {_text(k, f"{path} key"): _text(v, f"{path}.{k}") for k, v in value.items()}


def _parameters(value: Any, path: str) -> dict[str, str | int | float | bool]:
    """仅接受可直接写入 JSON 的标量组件匹配参数。"""
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    result = {}
    for key, item in value.items():
        name = _text(key, f"{path} key")
        if type(item) not in (str, int, float, bool):
            raise ValidationError(f"{path}.{name} must be a scalar")
        if type(item) is str:
            item = _text(item, f"{path}.{name}")
        elif type(item) in (int, float) and not math.isfinite(item):
            raise ValidationError(f"{path}.{name} must be finite")
        result[name] = item
    return result


def _quality_constraints(value: Any) -> tuple[QualityConstraint, ...]:
    """保留误码率等任务特有指标的比较方向和物理单位。"""
    if not isinstance(value, list):
        raise ValidationError("qos.quality_constraints must be an array")
    constraints = []
    for index, item in enumerate(value):
        path = f"qos.quality_constraints[{index}]"
        entry = _object(item, path, {"metric", "comparator", "value", "unit"})
        comparator = _text(entry["comparator"], f"{path}.comparator")
        if comparator not in {"<=", ">=", "=="}:
            raise ValidationError(f"{path}.comparator must be <=, >=, or ==")
        metric = _text(entry["metric"], f"{path}.metric")
        unit = _text(entry["unit"], f"{path}.unit")
        value = entry["value"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValidationError(f"{path}.value must be a finite number")
        constraints.append(QualityConstraint(metric, comparator, value, unit))
    return tuple(constraints)


def _resources(value: Any, path: str, *, version: int) -> Resources:
    required = {"cpu_millicores", "memory_mib", "dsp_units", "fpga_units", "bandwidth_mbps"}
    rf_fields = {"rf_channels", "rf_bandwidth_mhz"}
    entry = _object(value, path, required | rf_fields if version == 2 else required,
                    set() if version == 2 else rf_fields)
    return Resources(
        cpu_millicores=_number(entry["cpu_millicores"], f"{path}.cpu_millicores", integer=True),
        memory_mib=_number(entry["memory_mib"], f"{path}.memory_mib", integer=True),
        dsp_units=_number(entry["dsp_units"], f"{path}.dsp_units", integer=True),
        fpga_units=_number(entry["fpga_units"], f"{path}.fpga_units", integer=True),
        bandwidth_mbps=_number(entry["bandwidth_mbps"], f"{path}.bandwidth_mbps"),
        rf_channels=_number(entry.get("rf_channels", 0), f"{path}.rf_channels", integer=True),
        rf_bandwidth_mhz=_number(entry.get("rf_bandwidth_mhz", 0), f"{path}.rf_bandwidth_mhz"),
    )


def _extension(value: Any, family: str) -> dict[str, str | int | float | bool]:
    """每类只保留能影响组件选择和调度的少量代表性参数。"""
    fields = {
        "CNI": {"data_rate_mbps": "positive", "modulation_coding": "text",
                "link_quality": "ratio", "payload_mib": "positive"},
        "Radar": {"pri_us": "positive", "pulse_bandwidth_mhz": "positive",
                  "fft_size": "positive_int", "mode": "text"},
        "EW": {"threat_level": "positive_int", "response_limit_ms": "positive",
               "intercept_bandwidth_mhz": "positive", "jamming_mode": "text"},
        "ISAC": {"communication_rate_mbps": "positive", "sensing_resolution_m": "positive",
                 "shared_component_ratio": "ratio"},
    }
    if family not in fields:
        raise ValidationError(f"semantics.family must be one of {sorted(fields)} for schema_version 2")
    entry = _object(value, "semantics.extension", set(fields[family]))
    result = {}
    for name, kind in fields[family].items():
        path = f"semantics.extension.{name}"
        if kind == "text":
            result[name] = _text(entry[name], path)
        else:
            number = _number(entry[name], path, minimum=1 if kind == "positive_int" else 0,
                             integer=kind == "positive_int")
            if kind == "ratio" and number > 1:
                raise ValidationError(f"{path} must be <= 1")
            if kind == "positive" and number == 0:
                raise ValidationError(f"{path} must be > 0")
            result[name] = number
    return result


def _scheduling(value: Any) -> SchedulingPolicy:
    entry = _object(value, "qos.scheduling", {"preemptible", "migratable", "degradable",
                                                "shareable", "reconfiguration_ms"},
                    {"can_preempt"})
    flags = {}
    for name in ("preemptible", "migratable", "degradable", "shareable", "can_preempt"):
        flag_value = entry.get(name, False)
        if type(flag_value) is not bool:
            raise ValidationError(f"qos.scheduling.{name} must be a boolean")
        flags[name] = flag_value
    return SchedulingPolicy(**flags, reconfiguration_ms=_number(
        entry["reconfiguration_ms"], "qos.scheduling.reconfiguration_ms"))


def _profiles(value: Any, path: str) -> dict[str, ExecutionProfile]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"{path} must be a non-empty object")
    profiles = {}
    for device, raw in value.items():
        if device not in {"CPU", "DSP", "FPGA"}:
            raise ValidationError(f"{path} supports only CPU, DSP, FPGA")
        item_path = f"{path}.{device}"
        entry = _object(raw, item_path, {"runtime_ms", "setup_ms", "resources"})
        runtime = _number(entry["runtime_ms"], f"{item_path}.runtime_ms")
        if runtime == 0:
            raise ValidationError(f"{item_path}.runtime_ms must be > 0")
        resources = _resources(entry["resources"], f"{item_path}.resources", version=2)
        if (device == "CPU" and resources.cpu_millicores == 0 or
                device == "DSP" and resources.dsp_units == 0 or
                device == "FPGA" and resources.fpga_units == 0):
            raise ValidationError(f"{item_path}.resources must require its mapped device")
        profiles[device] = ExecutionProfile(runtime, _number(entry["setup_ms"],
                                           f"{item_path}.setup_ms"), resources)
    return profiles


def _simulation_profile(value: Any, path: str) -> SimulationProfile:
    """解析第一版仿真标定：资源等级、单级/范围时延、共享和抢占等级。"""
    entry = _object(value, path,
                    {"resource_levels", "latency_level", "sharing_class", "preemption_class"})
    resource_entry = _object(entry["resource_levels"], f"{path}.resource_levels",
                             {"cpu", "mem", "dsp", "fpga", "bw"})
    levels = {
        name: _number(resource_entry[name], f"{path}.resource_levels.{name}",
                       minimum=0, integer=True)
        for name in ("cpu", "mem", "dsp", "fpga", "bw")
    }
    for name, level in levels.items():
        if level > 5:
            raise ValidationError(f"{path}.resource_levels.{name} must be <= 5")
    latency = _text(entry["latency_level"], f"{path}.latency_level")
    latency_parts = latency.split("-")
    latency_order = {f"L{level}": level for level in range(6)}
    if (len(latency_parts) not in {1, 2} or
            any(part not in latency_order for part in latency_parts) or
            len(latency_parts) == 2 and latency_order[latency_parts[0]] > latency_order[latency_parts[1]]):
        raise ValidationError(f"{path}.latency_level must be L0-L5 or an ascending range such as L0-L1")
    sharing = _text(entry["sharing_class"], f"{path}.sharing_class")
    if sharing not in {"S", "P", "X"}:
        raise ValidationError(f"{path}.sharing_class must be S, P, or X")
    preemption = _text(entry["preemption_class"], f"{path}.preemption_class")
    if preemption not in {"Y", "C", "N"}:
        raise ValidationError(f"{path}.preemption_class must be Y, C, or N")
    return SimulationProfile(ResourceLevels(**levels), latency, sharing, preemption)


def _graph(value: Any, *, version: int) -> FunctionGraph:
    """校验功能图的节点、端口连接与无环依赖关系。"""
    graph = _object(value, "graph", {"nodes", "edges"})
    if not isinstance(graph["nodes"], list) or not graph["nodes"]:
        raise ValidationError("graph.nodes must be a non-empty array")
    if not isinstance(graph["edges"], list):
        raise ValidationError("graph.edges must be an array")

    nodes = []
    for index, item in enumerate(graph["nodes"]):
        path = f"graph.nodes[{index}]"
        required = {"id", "capability", "inputs", "outputs", "estimated_runtime_ms"}
        v2_fields = {"role", "shareable", "preemptible", "execution_profiles"}
        optional_fields = {"parameters"} | ({"simulation_profile"} if version == 2 else set())
        node = _object(item, path, required | v2_fields if version == 2 else required,
                       optional_fields)
        if version == 2:
            for flag in ("shareable", "preemptible"):
                if type(node[flag]) is not bool:
                    raise ValidationError(f"{path}.{flag} must be a boolean")
        nodes.append(FunctionNode(
            node_id=_text(node["id"], f"{path}.id"),
            capability=_text(node["capability"], f"{path}.capability"),
            inputs=_ports(node["inputs"], f"{path}.inputs"),
            outputs=_ports(node["outputs"], f"{path}.outputs"),
            estimated_runtime_ms=_number(node["estimated_runtime_ms"], f"{path}.estimated_runtime_ms"),
            parameters=_parameters(node.get("parameters", {}), f"{path}.parameters"),
            role=_text(node["role"], f"{path}.role") if version == 2 else None,
            shareable=node["shareable"] if version == 2 else False,
            preemptible=node["preemptible"] if version == 2 else False,
            execution_profiles=_profiles(node["execution_profiles"], f"{path}.execution_profiles")
            if version == 2 else {},
            simulation_profile=_simulation_profile(node["simulation_profile"],
                                                   f"{path}.simulation_profile")
            if "simulation_profile" in node else None,
        ))
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValidationError("graph node ids must be unique")
    if version == 2 and len({node.role for node in nodes}) != len(nodes):
        raise ValidationError("graph node roles must be unique within a task")

    # 每个输入端口只接收一个上游输出；汇聚操作应显式建成独立功能节点。
    edges = []
    used_inputs: set[tuple[str, str]] = set()
    for index, item in enumerate(graph["edges"]):
        path = f"graph.edges[{index}]"
        edge = _object(item, path, {"source", "source_port", "target", "target_port"})
        dependency = Dependency(*(_text(edge[key], f"{path}.{key}") for key in (
            "source", "source_port", "target", "target_port"
        )))
        if dependency.source not in by_id or dependency.target not in by_id:
            raise ValidationError(f"{path} references an unknown node")
        source = by_id[dependency.source]
        target = by_id[dependency.target]
        if dependency.source_port not in source.outputs or dependency.target_port not in target.inputs:
            raise ValidationError(f"{path} references an unknown port")
        if source.outputs[dependency.source_port] != target.inputs[dependency.target_port]:
            raise ValidationError(f"{path} connects incompatible data formats")
        target_input = (dependency.target, dependency.target_port)
        if target_input in used_inputs:
            raise ValidationError(f"{path} connects multiple sources to one input port")
        used_inputs.add(target_input)
        edges.append(dependency)

    # 功能图描述一次有限处理过程。流式处理的内部反馈归入功能节点；
    # 图级依赖环会使后续组件排序和完成时间估计失去定义。
    indegree = {node_id: 0 for node_id in by_id}
    successors: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for edge in edges:
        indegree[edge.target] += 1
        successors[edge.source].append(edge.target)
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for next_id in successors[node_id]:
            indegree[next_id] -= 1
            if indegree[next_id] == 0:
                queue.append(next_id)
    if visited != len(nodes):
        raise ValidationError("graph contains a dependency cycle")
    return FunctionGraph(tuple(nodes), tuple(edges))


def represent_task(raw: Mapping[str, Any]) -> TaskRepresentation:
    """将一条原始任务投影为 S、Q、R、G、X 五部分。

    只省略显式标记的 raw_artifacts；遇到未知字段立即报错，以免丢失
    新增的调度约束。时间统一为场景时钟上的毫秒，资源量为任务需求估计。
    """
    task = _object(raw, "task", {"schema_version", "task_id", "semantics", "qos", "resources", "graph", "state"}, {"raw_artifacts"})
    if type(task["schema_version"]) is not int or task["schema_version"] not in {1, 2}:
        raise ValidationError("schema_version must be 1 or 2")
    version = task["schema_version"]
    task_id = _text(task["task_id"], "task_id")

    # 静态任务属性进入 S；执行中可能变化的优先级另存于 X。
    semantics_fields = {"family", "task_type", "scenario", "direction", "base_priority"}
    v2_semantics = {"scenario_stage", "subtype", "trigger", "extension"}
    s = _object(task["semantics"], "semantics",
                semantics_fields | v2_semantics if version == 2 else semantics_fields)
    family = _text(s["family"], "semantics.family")
    semantics = Semantics(
        family=family,
        task_type=_text(s["task_type"], "semantics.task_type"),
        scenario=_text(s["scenario"], "semantics.scenario"),
        direction=_text(s["direction"], "semantics.direction"),
        base_priority=_number(s["base_priority"], "semantics.base_priority", integer=True),
        scenario_stage=_text(s["scenario_stage"], "semantics.scenario_stage") if version == 2 else None,
        subtype=_text(s["subtype"], "semantics.subtype") if version == 2 else None,
        trigger=_text(s["trigger"], "semantics.trigger") if version == 2 else None,
        extension=_extension(s["extension"], family) if version == 2 else {},
    )

    v2_qos = {"duration_ms", "max_response_ms", "security_level", "scheduling"}
    q = _object(task["qos"], "qos", {"arrival_ms", "deadline_ms"} | (v2_qos if version == 2 else set()),
                {"max_latency_ms", "min_throughput_mbps", "min_reliability", "quality_constraints",
                 "period_ms"} if version == 2 else
                {"max_latency_ms", "min_throughput_mbps", "min_reliability", "quality_constraints"})
    # V2 使用小数毫秒表达亚毫秒场景；V1 保留既有整数毫秒校验。
    legacy_integer_time = version == 1
    arrival = _number(q["arrival_ms"], "qos.arrival_ms", integer=legacy_integer_time)
    deadline = _number(q["deadline_ms"], "qos.deadline_ms", integer=legacy_integer_time)
    if deadline <= arrival:
        raise ValidationError("qos.deadline_ms must be after arrival_ms")
    reliability = _optional_number(q.get("min_reliability"), "qos.min_reliability")
    if reliability is not None and reliability > 1:
        raise ValidationError("qos.min_reliability must be <= 1")
    duration = _number(q["duration_ms"], "qos.duration_ms") if version == 2 else None
    period = _optional_number(q.get("period_ms"), "qos.period_ms")
    # 响应时限允许小数毫秒，以表达亚毫秒级告警/干扰链约束。
    response = _number(q["max_response_ms"], "qos.max_response_ms") if version == 2 else None
    if version == 2 and (duration == 0 or response == 0 or period == 0):
        raise ValidationError("qos duration, period and response must be > 0 when specified")
    qos = QoS(
        arrival_ms=arrival,
        deadline_ms=deadline,
        max_latency_ms=_optional_number(q.get("max_latency_ms"), "qos.max_latency_ms",
                                        integer=legacy_integer_time),
        min_throughput_mbps=_optional_number(q.get("min_throughput_mbps"), "qos.min_throughput_mbps"),
        min_reliability=reliability,
        quality_constraints=_quality_constraints(q.get("quality_constraints", [])),
        duration_ms=duration,
        period_ms=period,
        max_response_ms=response,
        security_level=_number(q["security_level"], "qos.security_level", integer=True)
        if version == 2 else None,
        scheduling=_scheduling(q["scheduling"]) if version == 2 else None,
    )

    resources = _resources(task["resources"], "resources", version=version)

    # 不在这一阶段合并或删减功能节点，先完整保留组件匹配所需的依赖图。
    graph = _graph(task["graph"], version=version)
    x_fields = {"remaining_work", "work_unit", "allocated_resources"}
    x = _object(task["state"], "state", {"status", "observed_at_ms", "progress", "current_priority"} |
                (x_fields if version == 2 else set()))
    status = _text(x["status"], "state.status")
    if status not in {"pending", "running", "completed", "failed"}:
        raise ValidationError("state.status must be pending, running, completed, or failed")
    observed = _number(x["observed_at_ms"], "state.observed_at_ms", integer=legacy_integer_time)
    if observed < arrival:
        raise ValidationError("state.observed_at_ms must not precede arrival_ms")
    progress = _number(x["progress"], "state.progress")
    if progress > 1:
        raise ValidationError("state.progress must be <= 1")
    remaining_work = _number(x["remaining_work"], "state.remaining_work") if version == 2 else None
    if version == 2 and status == "completed" and (progress != 1 or remaining_work != 0):
        raise ValidationError("completed state requires progress=1 and remaining_work=0")
    # 剩余时限从绝对截止时间派生，避免输入同时维护两个可能冲突的值。
    state = DynamicState(
        status=status,
        observed_at_ms=observed,
        progress=progress,
        current_priority=_number(x["current_priority"], "state.current_priority", integer=True),
        remaining_deadline_ms=max(0, deadline - observed),
        remaining_work=remaining_work,
        work_unit=_text(x["work_unit"], "state.work_unit") if version == 2 else None,
        allocated_resources=_resources(x["allocated_resources"], "state.allocated_resources", version=2)
        if version == 2 else None,
    )

    return TaskRepresentation(task_id, semantics, qos, resources, graph, state, version)
