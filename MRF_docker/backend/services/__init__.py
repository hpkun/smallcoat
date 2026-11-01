"""Application operations used by HTTP and other future adapters."""

from .resources import ResourceService
from .tasks import TaskService

__all__ = ["ResourceService", "TaskService"]
