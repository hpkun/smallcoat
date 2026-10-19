"""公共运行测量快照，不等同于 CPU/FPGA 占用或资源估算。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeMetrics:
    """普通字段描述最近一次公开处理调用；total_* 为成功调用累计。

    call_count/error_count 包含失败调用；total_processing_time_ns 含失败耗时。
    throughput_items_per_s 以输出逻辑单元计，字节吞吐以输入+输出有效载荷计。
    同步实现 queue_depth 恒为 0；缓存仅计组件显式持有的输入缓存。
    """

    operation: str = ""
    item_unit: str = "samples"
    last_call_succeeded: bool = False
    processing_time_ns: int = 0
    input_samples: int = 0
    output_samples: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    buffer_current_bytes: int = 0
    buffer_peak_bytes: int = 0
    queue_depth: int = 0
    call_count: int = 0
    error_count: int = 0
    total_processing_time_ns: int = 0
    total_success_time_ns: int = 0
    total_input_samples: int = 0
    total_output_samples: int = 0
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    total_frames: int = 0
    component_specific_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return self.processing_time_ns / 1_000_000

    @property
    def throughput_items_per_s(self) -> float:
        return (self.output_samples * 1_000_000_000 / self.processing_time_ns
                if self.processing_time_ns and self.last_call_succeeded else 0.0)

    @property
    def throughput_bytes_per_s(self) -> float:
        count = self.input_bytes + self.output_bytes
        return (count * 1_000_000_000 / self.processing_time_ns
                if self.processing_time_ns and self.last_call_succeeded else 0.0)

    @property
    def average_latency_ms(self) -> float:
        return self.total_processing_time_ns / self.call_count / 1_000_000 if self.call_count else 0.0
