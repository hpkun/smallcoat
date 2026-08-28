# Task Flow in This System

This diagram reflects the current code path in `SAGINEnvironment` and `CMADDPGEnv`.

```mermaid
flowchart TD
    A[Ground Device\nGenerate TaskInstance] --> B[Find nearest UAV\nIngress UAV]
    B --> C{Clustered?}
    C -- Yes --> D[Find decision UAV\nCluster Head CH]
    C -- No --> E[Ingress UAV is also\nDecision UAV]
    D --> F[Build candidate targets]
    E --> F

    F --> F1[Candidate 1: Ingress UAV]
    F --> F2[Candidate 2: Reachable BS]
    F --> F3[Candidate 3: LEO]

    F1 --> G[Estimate plan]
    F2 --> G
    F3 --> G

    G --> G1[Communication delay\nGround -> UAV\nUAV -> BS or LEO if needed]
    G --> G2[Compute delay\ncycles / (parallel_efficiency * node_capacity)]
    G --> G3[Queue delay\nPriority queue ordered by eta]
    G --> G4[Service time constrained by node capacity\ntasks may span multiple slots]
    G --> G5[Deadline check]
    G --> G6[Profit evaluation]

    G6 --> H{How is target selected?}
    H -- RL training/inference --> I[CH agent outputs\ntarget_node_id + priority_eta]
    H -- Heuristic/base env --> J[LEO-coordinated best-plan selection]

    I --> K[Resolve final target plan]
    J --> K

    K --> L{Queued workload within\nfinite buffer limit?}
    L -- Yes --> N[Commit task to target node queue]
    L -- No --> M[Capacity drop]

    N --> O[Start compute at\nscheduled start_time]
    O --> P[Finish at finish_time]
    P --> Q[Create ExecutionRecord]
    M --> Q
    Q --> R[Aggregate slot reward\nand metrics]
```

## Roles of Each Node Type

- `Ground Device`
  - Only generates tasks.
  - Does not execute computation in the current implementation.

- `UAV`
  - First access point for each task: the task is uploaded to the nearest UAV.
  - May act as the `decision UAV` / cluster head (`CH`) for RL decisions.
  - May also be the final compute node if the task is executed on the ingress UAV itself.

- `Base Station (BS)`
  - Pure compute target in the current implementation.
  - Receives tasks from UAV over the backhaul link.
  - Competes with UAV and LEO as a candidate execution node.

- `LEO`
  - Always available as a candidate compute target.
  - In the base environment logic, also acts as the global coordinator that evaluates candidate plans and profit.
  - In the RL wrapper, the CH agent chooses the target, but the environment still uses LEO-side profit evaluation logic.

## What “Resource” Means Here

- Compute resource
  - Each compute node has `compute_capacity_cycles_per_s`.
  - This is the node's raw processing capability.

- Service capacity
  - `compute_capacity_cycles_per_s` is used as the queue's processing rate.
  - A task's service time is its compute load divided by this rate and may span multiple slots.
  - Tasks are not rejected merely because their full compute load exceeds one slot of service.
  - Admission is rejected only when remaining queued service plus the new task exceeds
    its layer limit in `SimulationConfig.queue_capacity`: UAV 0.40 seconds, BS 0.20
    seconds, and LEO 0.10 seconds by default.

- Task compute load
  - A task consumes `total_compute_cycles`.
  - In code this is represented as `input_size_bits * cycles_per_bit`, which simplifies to `total_compute_cycles`.

- Queue resource
  - Tasks do not execute immediately.
  - Each node has a non-preemptive priority queue sorted by `priority_eta`, then arrival time.

- Communication resource
  - Upload and backhaul links contribute transmission and propagation delays.
  - These delays affect deadline feasibility and final reward.

## Code Landmarks

- Task generation: `TaskGenerator.generate_tasks()`
- First access UAV: `TaskInstance.ingress_uav_id`
- Decision UAV / CH: `SAGINEnvironment.get_decision_uav()`
- Candidate plan estimation: `SAGINEnvironment.build_candidate_plan()`
- Candidate targets: `SAGINEnvironment.iter_compute_targets()`
- RL action resolution: `CMADDPGEnv.step()`
- Finite-buffer admission: `SAGINEnvironment.queue_has_capacity()`
- Queue scheduling: `TaskQueueManager.estimate()` and `TaskQueueManager.commit()`
- Execution record output: `SAGINEnvironment.commit_plan()`
