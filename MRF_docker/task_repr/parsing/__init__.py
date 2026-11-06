"""Strict task input conversion."""

from .parser import represent_task
from .external_task import parse_external_task

__all__ = ["parse_external_task", "represent_task"]
