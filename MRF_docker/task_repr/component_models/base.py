"""动态组件数学模型的统一接口；具体模型将在后续加入。"""

from abc import ABC, abstractmethod
from typing import Mapping

from ..models.component import ComponentDefinition
from ..models.resource import ResourceProfile
from ..resource.hardware import HardwareProfile


class ComponentModel(ABC):
    """根据运行参数和硬件估计单个组件实例的资源画像。"""

    @abstractmethod
    def estimate(
        self,
        definition: ComponentDefinition,
        parameters: Mapping[str, str | int | float | bool],
        hardware: HardwareProfile | None = None,
    ) -> ResourceProfile:
        raise NotImplementedError
