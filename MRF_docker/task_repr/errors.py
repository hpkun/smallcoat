"""任务、组件库和资源建模共用的校验异常。"""


class ValidationError(ValueError):
    """输入缺失关键字段、含未知字段或依赖关系无效。"""
