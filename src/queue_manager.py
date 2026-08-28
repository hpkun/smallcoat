from __future__ import annotations

from dataclasses import dataclass
import itertools


@dataclass
class QueueEntry:
    """计算队列中的一个任务条目。"""

    task_id: str
    arrival_time_s: float
    service_time_s: float
    priority_eta: float
    order_index: int
    start_time_s: float = 0.0
    finish_time_s: float = 0.0

    @property
    def queue_delay_s(self) -> float:
        """排队等待时延。"""
        return max(0.0, self.start_time_s - self.arrival_time_s)


@dataclass(frozen=True)
class QueueSnapshot:
    """当前时刻队列拥塞快照。"""

    executing_queue_length: int
    buffer_queue_length: int
    expected_total_wait_s: float


class TaskQueueManager:
    """
    非抢占式优先级队列管理器。

    设计思路：
    - 任务到达后进入缓冲队列
    - 队列按优先级 eta 从高到低排序
    - 若优先级相同，则按到达时间和提交顺序排序
    - 已经开始执行的任务不再被重排，未开始任务允许动态调整顺序
    """

    def __init__(self) -> None:
        self.pending_entries: list[QueueEntry] = []
        self.order_counter = itertools.count()
        self.busy_until_s = 0.0

    def reset(self) -> None:
        self.pending_entries.clear()
        self.order_counter = itertools.count()
        self.busy_until_s = 0.0

    def _sort_entries(self, entries: list[QueueEntry]) -> list[QueueEntry]:
        """按 eta 优先级、到达时间和插入顺序排序。"""
        return sorted(
            entries,
            key=lambda entry: (-entry.priority_eta, entry.arrival_time_s, entry.order_index),
        )

    def _reschedule_entries(
        self,
        entries: list[QueueEntry],
        anchor_time_s: float,
    ) -> list[QueueEntry]:
        """从给定时刻开始重新计算所有未开始任务的开始/结束时间。"""

        cursor_s = max(anchor_time_s, self.busy_until_s)
        scheduled: list[QueueEntry] = []
        for entry in self._sort_entries(entries):
            start_time_s = max(cursor_s, entry.arrival_time_s)
            finish_time_s = start_time_s + entry.service_time_s
            scheduled.append(
                QueueEntry(
                    task_id=entry.task_id,
                    arrival_time_s=entry.arrival_time_s,
                    service_time_s=entry.service_time_s,
                    priority_eta=entry.priority_eta,
                    order_index=entry.order_index,
                    start_time_s=start_time_s,
                    finish_time_s=finish_time_s,
                )
            )
            cursor_s = finish_time_s
        return scheduled

    def advance_to(self, current_time_s: float) -> None:
        """
        推进物理时间，清理已完成任务，并冻结正在执行的任务。

        已经开始但尚未结束的任务不允许被重新排序，
        因此将其占用时段固化到 busy_until_s。
        """

        remaining_entries: list[QueueEntry] = []
        new_busy_until_s = (
            self.busy_until_s if self.busy_until_s > current_time_s else 0.0
        )

        for entry in self.pending_entries:
            if entry.finish_time_s <= current_time_s:
                continue
            if entry.start_time_s < current_time_s < entry.finish_time_s:
                new_busy_until_s = max(new_busy_until_s, entry.finish_time_s)
                continue
            if entry.start_time_s >= current_time_s:
                remaining_entries.append(entry)

        self.busy_until_s = new_busy_until_s
        self.pending_entries = self._reschedule_entries(remaining_entries, current_time_s)

    def estimate(
        self,
        *,
        task_id: str,
        arrival_time_s: float,
        service_time_s: float,
        priority_eta: float,
        current_time_s: float,
    ) -> QueueEntry:
        """估计一个新任务加入队列后的排队结果，但不真正写入队列。"""

        self.advance_to(current_time_s)
        candidate = QueueEntry(
            task_id=task_id,
            arrival_time_s=arrival_time_s,
            service_time_s=service_time_s,
            priority_eta=priority_eta,
            order_index=next(self.order_counter),
        )
        scheduled = self._reschedule_entries(
            self.pending_entries + [candidate],
            current_time_s,
        )
        for entry in scheduled:
            if entry.task_id == task_id and entry.order_index == candidate.order_index:
                return entry
        raise RuntimeError("Failed to estimate queue entry.")

    def commit(
        self,
        *,
        task_id: str,
        arrival_time_s: float,
        service_time_s: float,
        priority_eta: float,
        current_time_s: float,
    ) -> QueueEntry:
        """将任务真正加入队列，并返回最终调度结果。"""

        self.advance_to(current_time_s)
        candidate = QueueEntry(
            task_id=task_id,
            arrival_time_s=arrival_time_s,
            service_time_s=service_time_s,
            priority_eta=priority_eta,
            order_index=next(self.order_counter),
        )
        self.pending_entries = self._reschedule_entries(
            self.pending_entries + [candidate],
            current_time_s,
        )
        for entry in self.pending_entries:
            if entry.task_id == task_id and entry.order_index == candidate.order_index:
                return entry
        raise RuntimeError("Failed to commit queue entry.")

    def cancel(self, task_id: str, current_time_s: float) -> bool:
        """取消排队中或正在执行的任务，并从取消时刻释放计算资源。"""

        cancellable = [
            entry
            for entry in self.pending_entries
            if entry.task_id == task_id and entry.finish_time_s > current_time_s
        ]
        if not cancellable:
            return False

        # 清除已完成项和被取消项；若另有正在执行的任务，则保持其占用区间。
        active_entries: list[QueueEntry] = []
        future_entries: list[QueueEntry] = []
        for entry in self.pending_entries:
            if entry.finish_time_s <= current_time_s or entry in cancellable:
                continue
            if entry.start_time_s <= current_time_s < entry.finish_time_s:
                active_entries.append(entry)
            else:
                future_entries.append(entry)

        self.busy_until_s = max(
            (entry.finish_time_s for entry in active_entries),
            default=0.0,
        )
        self.pending_entries = self._reschedule_entries(future_entries, current_time_s)
        return True

    def snapshot(self, current_time_s: float) -> QueueSnapshot:
        """返回当前时刻的执行队列/缓冲队列/总等待时长快照。"""

        self.advance_to(current_time_s)
        executing_queue_length = 0
        buffer_queue_length = 0
        expected_total_wait_s = 0.0

        for entry in self.pending_entries:
            if entry.start_time_s <= current_time_s < entry.finish_time_s:
                executing_queue_length += 1
            elif entry.start_time_s >= current_time_s:
                buffer_queue_length += 1

        if self.pending_entries:
            expected_total_wait_s = max(
                0.0,
                max(entry.finish_time_s for entry in self.pending_entries) - current_time_s,
            )

        return QueueSnapshot(
            executing_queue_length=executing_queue_length,
            buffer_queue_length=buffer_queue_length,
            expected_total_wait_s=expected_total_wait_s,
        )

    def workload_s(self, current_time_s: float) -> float:
        """Return remaining compute service in the executing and buffered queue."""

        self.advance_to(current_time_s)
        running_workload_s = max(0.0, self.busy_until_s - current_time_s)
        buffered_workload_s = sum(entry.service_time_s for entry in self.pending_entries)
        return float(running_workload_s + buffered_workload_s)
