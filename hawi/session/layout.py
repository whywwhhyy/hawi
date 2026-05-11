"""Filesystem layout for session storage.

Each session lives in its own directory under ``root``::

    <root>/<session_id>/
        session.json           # manifest (written last on every checkpoint)
        context.json           # AgentContext snapshot
        message_history.jsonl  # append-only user-visible messages
        queues.json            # scheduler queues + steer + audit
        runtime.json           # in-flight run state + last unsent results
        plugins/<name>.json    # one per plugin returning non-None state
        exports/<id>/...       # session-internal markdown export bundles
        subagents/<id>/...     # child agent histories and export bundles

Snapshot component files share the same atomic-write protocol: write to a
``*.json.tmp`` sibling then ``os.replace`` it onto the final name. Message
history is incremental and append-only.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("~/.hawi/sessions").expanduser()

# Component file names (relative to a session directory).
MANIFEST_FILENAME = "session.json"
SESSION_LOCK_FILENAME = ".session.lock"
CONTEXT_FILENAME = "context.json"
MESSAGE_HISTORY_FILENAME = "message_history.jsonl"
QUEUES_FILENAME = "queues.json"
RUNTIME_FILENAME = "runtime.json"
PLUGINS_DIRNAME = "plugins"
EXPORTS_DIRNAME = "exports"
SUBAGENTS_DIRNAME = "subagents"

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
COMPONENT_MESSAGE_HISTORY = "message_history"
COMPONENT_QUEUES = "queues"
COMPONENT_RUNTIME = "runtime"
COMPONENT_PLUGINS = "plugins"
COMPONENT_MANIFEST = "manifest"


def session_dir(root: Path, session_id: str) -> Path:
    """Return the on-disk path for one session."""
    return root / session_id


def manifest_path(session_dir_: Path) -> Path:
    return session_dir_ / MANIFEST_FILENAME


def session_lock_path(session_dir_: Path) -> Path:
    return session_dir_ / SESSION_LOCK_FILENAME


def context_path(session_dir_: Path) -> Path:
    return session_dir_ / CONTEXT_FILENAME


def message_history_path(session_dir_: Path) -> Path:
    return session_dir_ / MESSAGE_HISTORY_FILENAME


def queues_path(session_dir_: Path) -> Path:
    return session_dir_ / QUEUES_FILENAME


def runtime_path(session_dir_: Path) -> Path:
    return session_dir_ / RUNTIME_FILENAME


def plugins_dir(session_dir_: Path) -> Path:
    return session_dir_ / PLUGINS_DIRNAME


def exports_dir(session_dir_: Path) -> Path:
    return session_dir_ / EXPORTS_DIRNAME


def export_dir(session_dir_: Path, export_id: str) -> Path:
    return exports_dir(session_dir_) / export_id


def subagents_dir(session_dir_: Path) -> Path:
    return session_dir_ / SUBAGENTS_DIRNAME


def subagent_dir(session_dir_: Path, subagent_id: str) -> Path:
    return subagents_dir(session_dir_) / subagent_id


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


def append_jsonl(path: Path, entries: list[dict[str, Any]], *, fsync: bool) -> None:
    """Append JSON records to ``path`` as newline-delimited JSON."""
    if not entries:
        return
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        if fsync:
            f.flush()
            os.fsync(f.fileno())

    if fsync:
        dir_fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records, skipping blank lines."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            out.append(value)
    return out


def remove_session_dir(session_dir_: Path) -> None:
    """Best-effort recursive delete of a session directory."""
    if session_dir_.exists():
        shutil.rmtree(session_dir_, ignore_errors=True)
