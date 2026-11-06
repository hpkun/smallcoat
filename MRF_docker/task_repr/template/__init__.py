"""任务 DAG 公共骨架与变体。"""

from .task_template import TaskTemplate, build_template, restore_task
from .business_task import BusinessTaskTemplate
from .external import represent_external_task, represent_external_tasks
from .refinement import (RefinementResult, SpecificationTask,
                         refine_business_task)

__all__ = ["BusinessTaskTemplate", "TaskTemplate", "build_template",
           "represent_external_task", "represent_external_tasks",
           "restore_task", "RefinementResult", "SpecificationTask",
           "refine_business_task"]
