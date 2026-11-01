"""运行组件之间的数据包；时间和频率单位在这里统一。"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DataPacket:
    """一维数据流或二维帧批次，最后一轴为帧内采样点。

    sample_rate: 样点/秒；center_frequency: Hz；timestamp: 首帧首点的秒时间戳。
    频域包沿用对应时域帧的时间戳和采样率。数组由包复制持有、只读，
    metadata 为可 JSON 序列化的描述信息，不承载原始数据数组。
    """

    samples: np.ndarray
    sample_rate: float
    center_frequency: float
    timestamp: float
    domain: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sample_rate", "center_frequency", "timestamp"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, float, np.integer, np.floating)) or
                    not math.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0 samples/s")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be a non-empty string")
        data = np.array(self.samples, copy=True)
        if data.ndim not in (1, 2) or data.dtype.kind not in "biufc":
            raise ValueError("samples must be a numeric 1D stream or 2D batch")
        if data.ndim == 2 and 0 in data.shape:
            raise ValueError("a 2D batch must have non-empty rows")
        data.setflags(write=False)
        object.__setattr__(self, "samples", data)
        if not isinstance(self.metadata, Mapping) or any(
            not isinstance(key, str) for key in self.metadata
        ):
            raise ValueError("metadata must be a mapping with string keys")
        metadata = deepcopy(dict(self.metadata))
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON-compatible values") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def item_count(self) -> int:
        return int(self.samples.size)

    @property
    def nbytes(self) -> int:
        """仅 ndarray 有效载荷字节，不含 Python 对象与元数据。"""
        return self.samples.nbytes
