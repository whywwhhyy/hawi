"""Cross-process advisory locks for persisted Hawi sessions."""

from __future__ import annotations

import errno
import json
import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt
else:  # pragma: no cover - import branch depends on platform
    import fcntl


_PROCESS_LOCKS: dict[Path, str] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class SessionLockInfo:
    """Best-effort status for a session lock file."""

    locked: bool
    owner: dict[str, Any] | None = None
    owned_by_self: bool = False


class SessionLockedError(RuntimeError):
    """Raised when a session is already held by another Hawi engine."""

    def __init__(
        self,
        session_id: str,
        *,
        owner: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.owner = owner
        super().__init__(f"Session {session_id} is locked by another Hawi engine")

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "lock_owner": self.owner}


class SessionLockUnavailable(RuntimeError):
    """Internal signal used when the raw file lock cannot be acquired."""


class SessionFileLock:
    """Exclusive lock handle for one session lock file.

    The file descriptor stays open for the lifetime of the lock. POSIX locks do
    not protect against a second lock attempt in the same process, so a small
    process-local registry mirrors the OS lock for tests and embedded runtimes.
    """

    def __init__(
        self,
        path: Path,
        *,
        owner_token: str,
        metadata: dict[str, Any],
    ) -> None:
        self.path = path
        self._owner_token = owner_token
        self._metadata = dict(metadata)
        self._file: TextIO | None = None
        self._held = False

    def acquire(self) -> "SessionFileLock":
        if self._held:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        resolved = self.path.resolve()
        with _PROCESS_LOCKS_GUARD:
            owner = _PROCESS_LOCKS.get(resolved)
            if owner is not None:
                raise SessionLockUnavailable(str(self.path))

        file = self.path.open("a+", encoding="utf-8")
        registered = False
        try:
            _ensure_lockable_byte(file)
            _lock_file(file)
            with _PROCESS_LOCKS_GUARD:
                owner = _PROCESS_LOCKS.get(resolved)
                if owner is not None:
                    _unlock_file(file)
                    raise SessionLockUnavailable(str(self.path))
                _PROCESS_LOCKS[resolved] = self._owner_token
                registered = True
            self._file = file
            self._held = True
            self._write_metadata()
        except Exception:
            if registered:
                with _PROCESS_LOCKS_GUARD:
                    if _PROCESS_LOCKS.get(resolved) == self._owner_token:
                        _PROCESS_LOCKS.pop(resolved, None)
            try:
                _unlock_file(file)
            except OSError:
                pass
            file.close()
            self._file = None
            self._held = False
            raise
        return self

    def release(self) -> None:
        file = self._file
        if file is None:
            return
        try:
            _unlock_file(file)
        finally:
            file.close()
            self._file = None
            self._held = False
            resolved = self.path.resolve()
            with _PROCESS_LOCKS_GUARD:
                if _PROCESS_LOCKS.get(resolved) == self._owner_token:
                    _PROCESS_LOCKS.pop(resolved, None)

    def _write_metadata(self) -> None:
        if self._file is None:
            return
        self._file.seek(0)
        self._file.truncate()
        self._file.write(json.dumps(self._metadata, ensure_ascii=False, indent=2))
        self._file.write("\n")
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass


def make_lock_metadata(manager_id: str) -> dict[str, Any]:
    """Return human-readable owner metadata for a lock file."""

    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "manager_id": manager_id,
        "acquired_at": datetime.now().isoformat(),
    }


def read_lock_owner(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[:200]}
    return data if isinstance(data, dict) else None


def probe_session_lock(path: Path, *, owner_token: str) -> SessionLockInfo:
    """Return whether ``path`` is locked by another manager/process."""

    resolved = path.resolve()
    with _PROCESS_LOCKS_GUARD:
        owner = _PROCESS_LOCKS.get(resolved)
        if owner is not None:
            return SessionLockInfo(
                locked=owner != owner_token,
                owner=read_lock_owner(path),
                owned_by_self=owner == owner_token,
            )

    if not path.exists():
        return SessionLockInfo(locked=False)

    try:
        file = path.open("a+", encoding="utf-8")
    except OSError:
        return SessionLockInfo(locked=True, owner=read_lock_owner(path))
    try:
        _ensure_lockable_byte(file)
        _lock_file(file)
    except (BlockingIOError, OSError) as exc:
        file.close()
        if _is_lock_conflict(exc):
            return SessionLockInfo(locked=True, owner=read_lock_owner(path))
        return SessionLockInfo(locked=True, owner=read_lock_owner(path))
    try:
        return SessionLockInfo(locked=False)
    finally:
        _unlock_file(file)
        file.close()


def _ensure_lockable_byte(file: TextIO) -> None:
    if os.name != "nt":
        return
    file.seek(0, os.SEEK_END)
    if file.tell() == 0:
        file.write(" ")
        file.flush()
    file.seek(0)


def _lock_file(file: TextIO) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if _is_lock_conflict(exc):
                raise BlockingIOError(*exc.args) from exc
            raise
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file: TextIO) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _is_lock_conflict(exc: BaseException) -> bool:
    code = getattr(exc, "errno", None)
    return isinstance(exc, BlockingIOError) or code in {errno.EACCES, errno.EAGAIN}
