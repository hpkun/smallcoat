"""任务到运行组件的静态匹配入口。"""

from .matcher import ComponentBinding, MatchIssue, StaticMatchResult, match_task
from .rules import MatchRuleError, adapt_parameters

__all__ = ["ComponentBinding", "MatchIssue", "MatchRuleError", "StaticMatchResult",
           "adapt_parameters", "match_task"]
