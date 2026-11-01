"""Common interface for parameterized resource estimation models."""

from abc import ABC, abstractmethod
from typing import Mapping

from components.models.component import ComponentDefinition
from resources.models import ResourceProfile
from resources.hardware import HardwareProfile


class ResourceModel(ABC):
    """Estimate one component instance's resource profile."""

    @abstractmethod
    def estimate(
        self,
        definition: ComponentDefinition,
        parameters: Mapping[str, str | int | float | bool],
        hardware: HardwareProfile | None = None,
    ) -> ResourceProfile:
        raise NotImplementedError
