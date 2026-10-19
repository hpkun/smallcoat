"""静态组件定义与任务内组件实例的数据结构。"""

from dataclasses import dataclass, field

from .resource import ResourceLevels, ResourceProfile


_LATENCY_LEVELS = {f"L{level}": level for level in range(6)}


@dataclass(frozen=True)
class ComponentSimulationProfile:
    """组件的第一版仿真标定；等级不是设备实测占用。"""

    resource_levels: ResourceLevels
    latency_level: str
    sharing_class: str
    preemption_class: str

    @property
    def latency_bounds(self) -> tuple[int, int]:
        """返回时延等级的上下界；单级时延的上下界相同。"""
        parts = self.latency_level.split("-")
        return _LATENCY_LEVELS[parts[0]], _LATENCY_LEVELS[parts[-1]]


@dataclass(frozen=True)
class ComponentDefinition:
    """组件库中的可复用静态定义，与任务实例分开。"""

    cid: str
    name: str
    short_name: str
    function: str
    input_formats: tuple[str, ...]
    output_formats: tuple[str, ...]
    configurable_parameters: tuple[str, ...]
    families: tuple[str, ...]
    typical_predecessors: tuple[str, ...]
    typical_successors: tuple[str, ...]
    primary_resources: tuple[str, ...]
    estimated_runtime_ms: float | None
    runtime_status: str
    simulation_profile: ComponentSimulationProfile


@dataclass(frozen=True)
class ComponentInstance:
    """某个任务中一次具体配置的组件；CID 引用静态定义。"""

    instance_id: str
    cid: str
    task_id: str
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)
    resource_profile: ResourceProfile | None = None
