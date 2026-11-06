"""外部业务请求的抽象表征与跨任务公共功能模板。"""

from collections.abc import Mapping

from ..errors import ValidationError
from ..parsing.external_task import parse_external_task
from .task_template import build_template, restore_task


def represent_external_task(raw: Mapping) -> dict:
    """解析单条请求并返回可恢复的抽象功能模板。"""
    task = parse_external_task(raw)
    template = build_template([task])
    return {"template_id": template.template_id,
            "task_template": template.to_dict(),
            "task_representation": restore_task(template, task["task_id"]),
            "status": "abstract_only_requires_refinement"}


def represent_external_tasks(raw: Mapping | list) -> dict:
    """批量解析请求，按任务族提取跨类型公共功能图。"""
    source_collection = None
    if isinstance(raw, Mapping) and "tasks" in raw:
        if set(raw) - {"tasks", "collection_version", "description"}:
            raise ValidationError("external task batch has unknown fields")
        if ("collection_version" in raw and
                raw["collection_version"] != "external_rf_tasks_collection.v1"):
            raise ValidationError("unsupported external task collection_version")
        if "description" in raw and not isinstance(raw["description"], str):
            raise ValidationError("external task collection description must be a string")
        source_collection = {key: raw[key] for key in ("collection_version", "description")
                             if key in raw}
        raw = raw["tasks"]
    if not isinstance(raw, list) or not raw:
        raise ValidationError("external task batch must be a nonempty array")
    tasks = [parse_external_task(item) for item in raw]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValidationError("external task ids must be unique")
    groups = {}
    for task in tasks:
        groups.setdefault(task["S"]["family"], []).append(task)
    templates = {family: build_template(group).to_dict()
                 for family, group in groups.items()}
    result = {"task_representations": tasks, "task_templates": templates,
              "status": "abstract_only_requires_refinement"}
    if source_collection:
        result["source_collection"] = source_collection
    return result
