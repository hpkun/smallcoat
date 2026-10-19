"""第一版仿真中的五维节点容量。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """归一化容量；默认每类资源容量为 100。"""

    cpu_capacity: float = 100.0
    memory_capacity: float = 100.0
    dsp_capacity: float = 100.0
    fpga_capacity: float = 100.0
    bandwidth_capacity: float = 100.0
