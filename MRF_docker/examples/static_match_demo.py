"""Run: python -m examples.static_match_demo [task.json]"""

import json
from pathlib import Path
import sys

from task_repr import match_task


def main() -> None:
    path = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            Path(__file__).parent / "TaskDemos" / "signal_analysis_task.json")
    print(json.dumps(match_task(path).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
