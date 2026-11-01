# Resource planning

`ResourceEstimator` uses models in `estimators/` to map a component definition and parameters to a resource
profile. `ResourcePlanner` takes a complete static match and an available
`HardwareProfile`, then returns per-node profiles and a capacity/time estimate.

The current executor runs nodes serially on CPU. Planning reserves the maximum
CPU and bandwidth pressure of one node, plus memory for all component instances.
These values use the existing normalized 0-100 simulation scale. DSP and FPGA
levels remain advisory and are not booked as physical allocations. Scenario
`estimated_runtime_ms` values are estimates, not measured QoS guarantees.

```python
from matching import match_task
from resources.planner import ResourcePlanner

match = match_task(task)
plan = ResourcePlanner().plan(task, match)
```
