"""Read named JSON documents from a configured directory."""

import json
from pathlib import Path
import re


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class DocumentNotFound(FileNotFoundError):
    pass


class InvalidDocument(ValueError):
    pass


class JsonDocumentRepository:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def list_names(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json")
                      if path.is_file() and _NAME.fullmatch(path.stem))

    def get(self, name: str) -> dict:
        if not _NAME.fullmatch(name):
            raise InvalidDocument("document name must contain only letters, digits, _ or -")
        path = self.directory / f"{name}.json"
        try:
            with path.open(encoding="utf-8") as source:
                document = json.load(source)
        except FileNotFoundError as exc:
            raise DocumentNotFound(f"document {name!r} was not found") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidDocument(f"cannot read document {name!r}: {exc}") from exc
        if not isinstance(document, dict):
            raise InvalidDocument(f"document {name!r} must be a JSON object")
        return document
