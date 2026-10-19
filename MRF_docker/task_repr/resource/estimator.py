"""把静态组件定义转换为统一资源画像，后续可接入数学模型。"""

from typing import Mapping

from ..component_models.registry import get_component_model
from ..models.component import ComponentDefinition
from ..models.resource import ResourceProfile
from .hardware import HardwareProfile


class ResourceEstimator:
    """优先使用动态模型；静态回退保留相对资源压力，不代表同时占用。"""

    def estimate(
        self,
        definition: ComponentDefinition,
        parameters: Mapping[str, str | int | float | bool],
        hardware: HardwareProfile | None = None,
    ) -> ResourceProfile:
        model = get_component_model(definition.cid)
        if model is not None:
            return model.estimate(definition, parameters, hardware)

        units = definition.simulation_profile.resource_levels.to_normalized_units()
        return ResourceProfile(
            cpu=units["cpu"],
            memory=units["mem"],
            dsp=units["dsp"],
            fpga=units["fpga"],
            bandwidth=units["bw"],
            latency_level=definition.simulation_profile.latency_level,
            runtime_ms=definition.estimated_runtime_ms,
        )
