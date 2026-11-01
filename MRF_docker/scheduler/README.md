# Scheduling simulation

`TaskScheduler` provides a single-process control-plane prototype. It accepts
a complete match, checks resource capacity, assigns simulated instance IDs,
and tracks `ACTIVE`, `WARM`, and `RELEASED` instances. Completing a task retains
its instances as warm; a later task can reuse matching CID/parameter instances.
When memory is tight, unrelated warm instances can be released before a new
placement. Preview has no side effects.

This module does not deploy Docker containers, perform concurrent execution,
or restore checkpoints. The existing `task_executor` runs the checked DAG in
process. `TaskService.execute_with_schedule()` connects the simulated
reservation to that executor for one task. A container adapter and measured
hardware profiles are separate development work before this scheduler can
control real deployments.
