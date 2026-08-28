from __future__ import annotations

from src.queue_manager import TaskQueueManager


def test_cancel_running_task_releases_queue_at_cancellation_time() -> None:
    queue = TaskQueueManager()
    queue.commit(
        task_id="running",
        arrival_time_s=0.0,
        service_time_s=10.0,
        priority_eta=1.0,
        current_time_s=0.0,
    )
    queue.commit(
        task_id="waiting",
        arrival_time_s=0.0,
        service_time_s=2.0,
        priority_eta=0.0,
        current_time_s=0.0,
    )

    assert queue.cancel("running", current_time_s=3.0)
    waiting = next(entry for entry in queue.pending_entries if entry.task_id == "waiting")
    assert waiting.start_time_s == 3.0
    assert waiting.finish_time_s == 5.0


def test_repeated_queue_updates_preserve_running_workload() -> None:
    queue = TaskQueueManager()
    queue.commit(
        task_id="running",
        arrival_time_s=0.0,
        service_time_s=1.0,
        priority_eta=1.0,
        current_time_s=0.0,
    )

    assert queue.workload_s(0.25) == 0.75
    assert queue.workload_s(0.25) == 0.75
