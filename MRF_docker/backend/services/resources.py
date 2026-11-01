"""Access virtual resource inventories independently from task documents."""

from ..repositories import JsonDocumentRepository


class ResourceService:
    def __init__(self, repository: JsonDocumentRepository):
        self.repository = repository

    def list_names(self) -> list[str]:
        return self.repository.list_names()

    def get(self, name: str) -> dict:
        return self.repository.get(name)
