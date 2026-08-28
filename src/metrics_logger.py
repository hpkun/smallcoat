from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class MetricsLogger:
    """训练/评估指标记录器。"""

    records: list[dict] = field(default_factory=list)

    def log(self, **kwargs) -> None:
        """记录一条指标。"""
        self.records.append(dict(kwargs))

    def to_json(self, output_path: str | Path) -> None:
        """导出为 JSON 文件。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
