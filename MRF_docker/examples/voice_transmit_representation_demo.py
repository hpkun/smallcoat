"""通过统一外部任务入口展示语音发送请求的抽象表征。"""

import argparse
import json
from pathlib import Path

from task_repr import represent_external_task


DEFAULT_REQUEST = Path(__file__).parent / "BusinessRequests" / "voice_transmit_request.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="语音发送请求的任务表征演示")
    parser.add_argument("request_json", nargs="?", type=Path, default=DEFAULT_REQUEST)
    args = parser.parse_args()
    with args.request_json.open(encoding="utf-8") as source:
        request = json.load(source)
    print(json.dumps(represent_external_task(request), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
