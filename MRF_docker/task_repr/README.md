# Task representation

`task_repr` owns the task schema and conversion from task JSON to
`TaskRepresentation`. Its output is `T_rep = {S, Q, R, G, X}`: semantics,
quality constraints, resource hints, a function DAG, and dynamic state.

| Directory | Responsibility |
| --- | --- |
| `models/` | Task data structures, including DAG nodes and edges |
| `parsing/` | Strict v1/v2 JSON validation and `represent_task()` |
| `template/` | Common DAG skeleton and task variant restoration |

The task graph describes required functions. It does not bind component CIDs.
The `matching` package performs that step after reading the component catalog
and OPS/OCS contracts from `components`.

```python
from task_repr import represent_task

task = represent_task(raw_task_json)
print(task.G.nodes, task.Q.deadline_ms)
```

The former combined task/component/resource documentation is retained in
[prototype notes](../docs/prototype_notes.md) as a historical design reference.
