"""FastAPI entry point. Services can be replaced in tests or later deployments."""

import os
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from task_repr import ValidationError
from scheduler import SchedulingError

from ..repositories import DocumentNotFound, InvalidDocument, JsonDocumentRepository
from ..services import ResourceService, TaskService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(task_service: TaskService | None = None,
               resource_service: ResourceService | None = None) -> FastAPI:
    task_service = task_service or TaskService(JsonDocumentRepository(
        os.environ.get("MRF_TASK_DIR", PROJECT_ROOT / "examples" / "TaskDemos")))
    resource_service = resource_service or ResourceService(JsonDocumentRepository(
        os.environ.get("MRF_RESOURCE_DIR", PROJECT_ROOT / "examples" / "ResourceDemos")))
    app = FastAPI(title="MRF Experiment API", version="0.1.0")

    @app.exception_handler(DocumentNotFound)
    async def document_not_found(_request, exc: DocumentNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidDocument)
    @app.exception_handler(ValidationError)
    async def invalid_document(_request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(SchedulingError)
    async def scheduling_error(_request, exc: SchedulingError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/tasks")
    def list_tasks():
        return {"items": task_service.list_names()}

    @app.post("/api/v1/tasks/validate")
    def validate_task(document: dict):
        return task_service.validate(document)

    @app.post("/api/v1/task-representations/external")
    def represent_external(request: dict | list = Body(...)):
        return task_service.represent_external(request)

    @app.get("/api/v1/tasks/{name}")
    def get_task(name: str):
        return task_service.get(name)

    @app.post("/api/v1/tasks/{name}/match")
    def match_saved_task(name: str):
        return task_service.match(task_service.get(name))

    @app.post("/api/v1/matches")
    def match_submitted_task(document: dict):
        return task_service.match(document)

    @app.post("/api/v1/plans")
    def plan_submitted_task(document: dict):
        return task_service.plan(document)

    @app.post("/api/v1/simulations/schedule")
    def simulate_schedule(document: dict):
        return task_service.simulate_schedule(document)

    @app.post("/api/v1/simulations/tasks/{task_id}/complete")
    def complete_simulation(task_id: str):
        return task_service.complete_simulation(task_id)

    @app.get("/api/v1/simulations/state")
    def simulation_state():
        return task_service.simulation_state()

    @app.get("/api/v1/resources")
    def list_resources():
        return {"items": resource_service.list_names()}

    @app.get("/api/v1/resources/{name}")
    def get_resource(name: str):
        return resource_service.get(name)

    return app


app = create_app()
