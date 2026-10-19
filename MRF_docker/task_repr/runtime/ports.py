"""运行端口契约与连接检查；不同于任务 DAG 内的描述性端口名称。"""

from dataclasses import dataclass

from .packet import DataPacket


@dataclass(frozen=True)
class RuntimePort:
    name: str
    direction: str
    domains: tuple[str, ...]
    dtype_kinds: tuple[str, ...] = ("c",)
    dimensions: tuple[int, ...] = (1, 2)
    item_unit: str = "samples"

    def __post_init__(self) -> None:
        if (not self.name or self.direction not in ("input", "output") or
                not self.domains or not self.dtype_kinds or not self.dimensions or
                not self.item_unit):
            raise ValueError("invalid runtime port contract")

    def validate(self, packet: DataPacket) -> None:
        if not isinstance(packet, DataPacket):
            raise TypeError(f"{self.name} requires a DataPacket")
        if packet.domain not in self.domains:
            raise ValueError(f"{self.name} requires domain in {self.domains}")
        if packet.samples.dtype.kind not in self.dtype_kinds:
            raise ValueError(f"{self.name} requires dtype kind in {self.dtype_kinds}")
        if packet.samples.ndim not in self.dimensions:
            raise ValueError(f"{self.name} requires dimensions in {self.dimensions}")

    def can_connect_to(self, target: "RuntimePort") -> bool:
        """输出的所有可能格式均应被输入接受；帧长等仍在运行时校验。"""
        return (
            self.direction == "output" and target.direction == "input" and
            self.item_unit == target.item_unit and
            set(self.domains) <= set(target.domains) and
            set(self.dtype_kinds) <= set(target.dtype_kinds) and
            set(self.dimensions) <= set(target.dimensions)
        )
