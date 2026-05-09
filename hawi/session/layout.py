"""Filesystem layout for session storage.

Each session lives in its own directory under ``root``::

    <root>/<session_id>/
        session.json           # manifest (written last on every checkpoint)
        context.json           # AgentContext snapshot
        queues.json            # scheduler queues + steer + audit
        runtime.json           # in-flight run state + last unsent results
        plugins/<name>.json    # one per plugin returning non-None state

All component files share the same atomic-write protocol: write to a
``*.json.tmp`` sibling then ``os.replace`` it onto the final name.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("~/.hawi/sessions").expanduser()

# Component file names (relative to a session directory).
MANIFEST_FILENAME = "session.json"
CONTEXT_FILENAME = "context.json"
QUEUES_FILENAME = "queues.json"
RUNTIME_FILENAME = "runtime.json"
PLUGINS_DIRNAME = "plugins"

# Top-level versions for migration handling.
MANIFEST_VERSION = 1
QUEUES_VERSION = 1
RUNTIME_VERSION = 1
PLUGIN_FILE_VERSION = 1
# Context already uses string "1.0" — keep that for backward compat with
# AgentContext.save/load.
CONTEXT_VERSION = "1.0"

# Component identifiers used in routing tables and WriteJob payloads.
COMPONENT_CONTEXT = "context"
COMPONENT_QUEUES = "queues"
COMPONENT_RUNTIME = "runtime"
COMPONENT_PLUGINS = "plugins"
COMPONENT_MANIFEST = "manifest"


def session_dir(root: Path, session_id: str) -> Path:
    """Return the on-disk path for one session."""
    return root / session_id


def manifest_path(session_dir_: Path) -> Path:
    return session_dir_ / MANIFEST_FILENAME


def context_path(session_dir_: Path) -> Path:
    return session_dir_ / CONTEXT_FILENAME


def queues_path(session_dir_: Path) -> Path:
    return session_dir_ / QUEUES_FILENAME


def runtime_path(session_dir_: Path) -> Path:
    return session_dir_ / RUNTIME_FILENAME


def plugins_dir(session_dir_: Path) -> Path:
    return session_dir_ / PLUGINS_DIRNAME


def plugin_state_path(session_dir_: Path, plugin_name: str) -> Path:
    return plugins_dir(session_dir_) / f"{plugin_name}.json"


def ensure_session_layout(session_dir_: Path) -> None:
    """Create the session directory and ``plugins/`` subdir if missing."""
    session_dir_.mkdir(parents=True, exist_ok=True)
    plugins_dir(session_dir_).mkdir(exist_ok=True)


def atomic_write_text(path: Path, content: str, *, fsync: bool) -> None:
    """Write ``content`` to ``path`` atomically via tempfile + os.replace.

    With ``fsync=True``, fsync the temp file's contents AND the parent
    directory inode before returning — required for guaranteed durability on
    POSIX (without it, the rename may not survive a power loss).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
        if fsync:
            # fsync parent dir to persist the rename.
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def remove_session_dir(session_dir_: Path) -> None:
    """Best-effort recursive delete of a session directory."""
    if session_dir_.exists():
        shutil.rmtree(session_dir_, ignore_errors=True)
