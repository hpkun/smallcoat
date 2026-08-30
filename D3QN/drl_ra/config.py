from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/paper.yaml") -> dict[str, Any]:
    """Load a YAML configuration without mutating shared defaults."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return deepcopy(data)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides such as ``training.episodes=10``."""
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}; expected key=value")
        dotted_key, raw_value = item.split("=", 1)
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"unknown configuration path: {dotted_key}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown configuration key: {dotted_key}")
        target[parts[-1]] = yaml.safe_load(raw_value)
    return result

