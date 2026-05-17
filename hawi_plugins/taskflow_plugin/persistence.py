from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hawi.utils.config_loader import load_config_file, save_config_file

from .models import TaskflowDefinition


DEFAULT_TASKFLOW_DIR = Path.home() / ".hawi" / "taskflows"


def _ensure_dir() -> Path:
    path = Path(os.environ.get("HAWI_TASKFLOW_DIR", str(DEFAULT_TASKFLOW_DIR)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _taskflow_path(name: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return _ensure_dir() / f"{safe_name}.yaml"


def save_taskflow(taskflow: TaskflowDefinition) -> str:
    path = _taskflow_path(taskflow.title)
    save_config_file(path, taskflow.to_dict())
    return str(path)


def load_taskflow(name: str) -> TaskflowDefinition:
    path = _taskflow_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Taskflow {name!r} not found at {path}")
    data = load_config_file(path)
    return TaskflowDefinition.from_dict(data)


def list_taskflows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(_ensure_dir().glob("*.yaml")):
        name = path.stem
        try:
            taskflow = load_taskflow(name)
        except Exception as exc:
            result.append({"name": name, "path": str(path), "error": str(exc)})
            continue
        result.append(
            {
                "name": taskflow.title,
                "id": taskflow.id,
                "mode": taskflow.mode,
                "execution_policy": taskflow.execution_policy,
                "description": taskflow.description,
                "path": str(path),
                "step_count": len(taskflow.steps),
                "edge_count": len(taskflow.edges),
            }
        )
    return result
