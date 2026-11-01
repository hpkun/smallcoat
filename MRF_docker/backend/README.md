# Backend framework

The HTTP layer lives in `api`, use cases in `services`, and JSON file access in
`repositories`. It calls the task, component matching, resource planning, and
scheduling packages. The resource endpoint exposes an independent virtual
inventory; the simulation scheduler currently uses normalized default capacity.

Install dependencies and start the API from the project root with the same
Python interpreter:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.api.app:app --reload
```

The API is at `http://127.0.0.1:8000`; interactive documentation is at `/docs`.
Set `MRF_TASK_DIR` and `MRF_RESOURCE_DIR` to directories of JSON files to load
different experiments. File names become API names, for example
`signal_analysis_task.json` is available as `signal_analysis_task`.

| Method | Path | Operation |
| --- | --- | --- |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/tasks` | List task file names |
| GET | `/api/v1/tasks/{name}` | Read a task JSON document |
| POST | `/api/v1/tasks/validate` | Validate a submitted task JSON |
| POST | `/api/v1/tasks/{name}/match` | Match a saved task |
| POST | `/api/v1/matches` | Match a submitted task JSON |
| POST | `/api/v1/plans` | Preview normalized resource capacity and time estimate |
| POST | `/api/v1/simulations/schedule` | Reserve capacity and activate simulated instances |
| POST | `/api/v1/simulations/tasks/{task_id}/complete` | Move a task's instances to Warm |
| GET | `/api/v1/simulations/state` | Inspect active tasks and instance states |
| GET | `/api/v1/resources` | List resource file names |
| GET | `/api/v1/resources/{name}` | Read a resource JSON document |

`TaskService.execute(document, sources)` runs the checked DAG in process.
`TaskService.execute_with_schedule(document, sources)` also reserves simulated
capacity and moves instances to Warm on success or Released on failure. An HTTP
execution endpoint still needs a packet upload and result format. Resource JSON
is currently read as inventory and is not automatically mapped to normalized
simulation capacity. Simulation scheduling does not deploy Docker.
