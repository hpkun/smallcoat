"""任务 DAG 公共骨架与变体。"""

from .task_template import TaskTemplate, build_template, restore_task

__all__ = ["TaskTemplate", "build_template", "restore_task"]
