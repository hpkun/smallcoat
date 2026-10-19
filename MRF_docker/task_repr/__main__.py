"""读取 JSON 任务并打印校验后的轻量化表征。"""

import argparse
import json
from pathlib import Path

from .models.task import represent_task


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并表征一条波形任务")
    parser.add_argument("task_json", type=Path)
    args = parser.parse_args()
    with args.task_json.open(encoding="utf-8") as source:
        raw = json.load(source)
    print(represent_task(raw).to_json())


if __name__ == "__main__":
    main()
