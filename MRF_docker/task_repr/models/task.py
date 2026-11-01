"""Task representation data structures."""

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from resources.models import ExecutionProfile, Resources, SimulationProfile


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


