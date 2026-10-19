"""动态组件模型接口及注册表。"""

from .base import ComponentModel
from .registry import COMPONENT_MODEL_REGISTRY, get_component_model

__all__ = ["ComponentModel", "COMPONENT_MODEL_REGISTRY", "get_component_model"]
