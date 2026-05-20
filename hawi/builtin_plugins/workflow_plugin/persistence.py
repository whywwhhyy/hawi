"""Workflow persistence — save/load/list workflows as YAML files."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hawi.utils.config_loader import load_config_file, save_config_file
from hawi.builtin_plugins.workflow_plugin.models import Workflow

logger = logging.getLogger(__name__)

# Default directory: ~/.hawi/workflows/
DEFAULT_WORKFLOW_DIR = Path.home() / ".hawi" / "workflows"


def _ensure_dir() -> Path:
    """Return the workflow storage directory, creating it if needed."""
    path = Path(os.environ.get("HAWI_WORKFLOW_DIR", str(DEFAULT_WORKFLOW_DIR)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workflow_path(name: str) -> Path:
    """Return the file path for a saved workflow."""
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return _ensure_dir() / f"{safe_name}.yaml"


def save_workflow(workflow: Workflow) -> str:
    """Persist a workflow definition to disk.

    Args:
        workflow: The workflow to save.

    Returns:
        The absolute path to the saved file.

    Raises:
        OSError: If the file cannot be written.
    """
    data = workflow.to_dict()
    path = _workflow_path(workflow.name)

    save_config_file(path, data)

    logger.info("Workflow '%s' saved to %s", workflow.name, path)
    return str(path)


def load_workflow(name: str) -> Workflow:
    """Load a workflow definition from disk.

    Args:
        name: The workflow name (matches the filename stem).

    Returns:
        The deserialized Workflow.

    Raises:
        FileNotFoundError: If no workflow with this name exists.
    """
    path = _workflow_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Workflow '{name}' not found at {path}")

    data = load_config_file(path)

    return Workflow.from_dict(data)


def list_workflows() -> list[dict[str, Any]]:
    """List all saved workflows with summary metadata.

    Returns:
        A list of dicts with keys: ``name``, ``path``, ``node_count``,
        ``edge_count``.
    """
    dir_path = _ensure_dir()
    result: list[dict[str, Any]] = []
    for fpath in sorted(dir_path.glob("*.yaml")):
        name = fpath.stem
        try:
            wf = load_workflow(name)
            result.append({
                "name": wf.name,
                "id": wf.id,
                "description": wf.description,
                "path": str(fpath),
                "node_count": len(wf.nodes),
                "edge_count": len(wf.edges),
            })
        except Exception as exc:
            logger.warning("Skipping unreadable workflow file %s: %s", fpath, exc)
            result.append({
                "name": name,
                "path": str(fpath),
                "error": str(exc),
            })
    return result


def delete_workflow(name: str) -> bool:
    """Delete a saved workflow.

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    path = _workflow_path(name)
    if not path.exists():
        return False
    path.unlink()
    logger.info("Workflow '%s' deleted (%s)", name, path)
    return True
