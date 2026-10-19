"""CID 到具体数学模型的映射入口。"""

from .base import ComponentModel
from .common.fft import FFTModel


# 当前仅 C08 已有参数化模型，其余组件继续使用静态画像。
COMPONENT_MODEL_REGISTRY: dict[str, type[ComponentModel]] = {"C08": FFTModel}


def get_component_model(cid: str) -> ComponentModel | None:
    """返回已注册模型实例；未建模组件返回 None，由估计器回退。"""
    model_type = COMPONENT_MODEL_REGISTRY.get(cid)
    return model_type() if model_type is not None else None
