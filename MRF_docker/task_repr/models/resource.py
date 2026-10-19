"""任务资源、仿真等级及动态资源画像的数据结构。"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Resources:
    """旧任务输入中的绝对资源估计，不是组件等级标定。"""

    cpu_millicores: int
    memory_mib: int
    dsp_units: int
    fpga_units: int
    bandwidth_mbps: float
    rf_channels: int = 0
    rf_bandwidth_mhz: float = 0


@dataclass(frozen=True)
class ExecutionProfile:
    """功能映射到某类处理资源时的时间与资源代价估计。"""

    runtime_ms: float
    setup_ms: float
    resources: Resources


@dataclass(frozen=True)
class ResourceLevels:
    """仿真基线中的 0-5 级异构资源压力/适配属性。"""

    cpu: int
    mem: int
    dsp: int
    fpga: int
    bw: int

    def to_normalized_units(self) -> dict[str, int]:
        levels = (0, 5, 10, 20, 35, 50)
        return {name: levels[value] for name, value in asdict(self).items()}


@dataclass(frozen=True)
class SimulationProfile:
    """第一版组件仿真标定属性；不是设备实测占用。"""

    resource_levels: ResourceLevels
    latency_level: str
    sharing_class: str
    preemption_class: str


@dataclass(frozen=True)
class ResourceProfile:
    """五维值是归一化压力；动态量单独存放，不能当作资源占用。"""

    cpu: float
    memory: float
    dsp: float
    fpga: float
    bandwidth: float
    latency_level: str | None = None
    runtime_ms: float | None = None
    workload: float | None = None
    memory_bytes: int | None = None
    bandwidth_bytes_per_s: float | None = None
    observation_ms: float | None = None
