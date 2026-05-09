"""Daemon-thread writer that performs all session disk I/O off the agent thread.

Design constraints:
- Writers MUST NOT touch live agent state. Snapshots are dicts captured on the
  caller's thread and handed over.
- Failures must not crash the writer thread; they emit ``session.write_failed``
  events and the loop continues.
- A bounded queue prevents pathological cases from blowing memory; on overflow,
  droppable state snapshots discard the oldest same-component pending job.
- Message history entries are append-only records, not full snapshots.
- ``fsync=True`` (set on the final-flush sentinel and on explicit
  ``save_session``) flushes file contents AND the parent directory inode.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hawi.events import EventBus, SessionWriteFailedEvent

from . import layout

logger = logging.getLogger(__name__)


_FINAL_FLUSH_SENTINEL = object()


@dataclass
class WriteJob:
    """One unit of work for the SessionWriter.

    Each entry in ``snapshots`` maps a component name (one of
    ``layout.COMPONENT_*``) to its already-captured pure-data snapshot dict
    (or, for plugins, a nested mapping of plugin name → state dict).
    ``message_history_entries`` are appended to the session history JSONL file.

    When ``fsync`` is True, the writer fsyncs each file and the parent dir.
    """

    session_dir: Path
    snapshots: dict[str, Any] = field(default_factory=dict)
    message_history_entries: list[dict[str, Any]] = field(default_factory=list)
    manifest_patch: dict[str, Any] = field(default_factory=dict)
    fsync: bool = False
    component_set_key: str = ""
    drop_on_overflow: bool = True

    def __post_init__(self) -> None:
        if not self.component_set_key:
            components = set(self.snapshots.keys())
            if self.message_history_entries:
                components.add(layout.COMPONENT_MESSAGE_HISTORY)
            self.component_set_key = ",".join(sorted(components))


class SessionWriter:
    """Daemon thread that consumes :class:`WriteJob` items.

    Use :py:meth:`start` to spin up the thread and :py:meth:`shutdown` to drain
    and stop it. On a full queue, droppable jobs discard the oldest job sharing
    the same component set. Non-droppable jobs (message history increments)
    wait briefly for capacity before logging a loss.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        max_queue_size: int = 256,
        thread_name: str = "hawi-session-writer",
    ) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._event_bus = event_bus
        self._thread_name = thread_name
        self._stop_requested = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, job: WriteJob) -> None:
        """Enqueue a job. On overflow drop the oldest same-component-set job."""
        with self._inflight_lock:
            self._inflight_count += 1
            self._idle_event.clear()
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            dropped = self._drop_oldest_with_key(job.component_set_key)
            if not dropped and not job.drop_on_overflow:
                try:
                    self._queue.put(job, timeout=1.0)
                    return
                except queue.Full:
                    logger.error(
                        "session writer queue full; losing non-droppable job %s",
                        job.component_set_key,
                    )
                    with self._inflight_lock:
                        self._inflight_count -= 1
                        if self._inflight_count == 0:
                            self._idle_event.set()
                    return
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                # Unreachable in practice — _drop_oldest just freed a slot.
                logger.warning(
                    "session writer queue still full after drop; losing job %s",
                    job.component_set_key,
                )
                with self._inflight_lock:
                    self._inflight_count -= 1
                    if self._inflight_count == 0:
                        self._idle_event.set()

    def shutdown(self, *, timeout: float = 10.0) -> bool:
        """Request the thread stops once the queue drains.

        Returns True if the thread joined within the timeout, False otherwise.
        """
        self._queue.put(_FINAL_FLUSH_SENTINEL)
        self._stop_requested.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def wait_idle(self, *, timeout: float = 10.0) -> bool:
        """Block until all submitted jobs have been processed."""
        return self._idle_event.wait(timeout)

    def _drop_oldest_with_key(self, key: str) -> bool:
        # We can't remove a specific item from queue.Queue without draining it.
        # Drain into a list, drop the oldest matching job, refill.
        drained: list[WriteJob | object] = []
        try:
            while True:
                drained.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        dropped = False
        kept: list[WriteJob | object] = []
        for item in drained:
            if (
                not dropped
                and isinstance(item, WriteJob)
                and item.component_set_key == key
                and item.drop_on_overflow
            ):
                dropped = True
                logger.warning(
                    "session writer dropped oldest pending job for components=%s",
                    key,
                )
                with self._inflight_lock:
                    self._inflight_count -= 1
                    if self._inflight_count == 0:
                        self._idle_event.set()
                continue
            kept.append(item)

        if not dropped and kept:
            # No same-key match found; drop the oldest WriteJob anyway.
            for i, item in enumerate(kept):
                if isinstance(item, WriteJob) and item.drop_on_overflow:
                    dropped = True
                    logger.warning(
                        "session writer queue full and no match for %s; "
                        "dropping oldest job (components=%s)",
                        key,
                        item.component_set_key,
                    )
                    kept.pop(i)
                    with self._inflight_lock:
                        self._inflight_count -= 1
                        if self._inflight_count == 0:
                            self._idle_event.set()
                    break

        for item in kept:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                logger.error(
                    "session writer requeue failed unexpectedly; losing job"
                )
                if isinstance(item, WriteJob):
                    with self._inflight_lock:
                        self._inflight_count -= 1
                        if self._inflight_count == 0:
                            self._idle_event.set()
        return dropped

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _FINAL_FLUSH_SENTINEL:
                return
            assert isinstance(item, WriteJob)
            try:
                self._write_job(item)
            except Exception as exc:
                logger.exception("session writer failed on job %s", item.component_set_key)
                self._emit_failure(item, "<job>", exc)
            finally:
                with self._inflight_lock:
                    self._inflight_count -= 1
                    if self._inflight_count == 0:
                        self._idle_event.set()

    def _write_job(self, job: WriteJob) -> None:
        layout.ensure_session_layout(job.session_dir)
        wrote_components: list[str] = []

        for name, payload in job.snapshots.items():
            if name == layout.COMPONENT_MANIFEST:
                # Manifest is always written last (below) so include any
                # component patches there.
                continue
            try:
                self._write_component(job.session_dir, name, payload, fsync=job.fsync)
                wrote_components.append(name)
            except Exception as exc:
                logger.exception(
                    "session writer failed for component %s in %s",
                    name,
                    job.session_dir,
                )
                self._emit_failure(job, name, exc)

        if job.message_history_entries:
            try:
                self._write_message_history(job.session_dir, job, fsync=job.fsync)
                wrote_components.append(layout.COMPONENT_MESSAGE_HISTORY)
            except Exception as exc:
                logger.exception(
                    "session writer failed appending message history in %s",
                    job.session_dir,
                )
                self._emit_failure(job, layout.COMPONENT_MESSAGE_HISTORY, exc)

        # Manifest write — merge in patch with current updated_at + last
        # checkpoint trigger. The manifest is always written last so that a
        # torn write leaves the previous manifest pointing at the previous
        # component set.
        try:
            self._write_manifest(job, wrote_components)
        except Exception as exc:
            logger.exception("session writer failed writing manifest in %s", job.session_dir)
            self._emit_failure(job, layout.COMPONENT_MANIFEST, exc)

    def _write_component(
        self,
        session_dir_: Path,
        component: str,
        payload: Any,
        *,
        fsync: bool,
    ) -> None:
        if component == layout.COMPONENT_CONTEXT:
            path = layout.context_path(session_dir_)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            layout.atomic_write_text(path, text, fsync=fsync)
        elif component == layout.COMPONENT_QUEUES:
            path = layout.queues_path(session_dir_)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            layout.atomic_write_text(path, text, fsync=fsync)
        elif component == layout.COMPONENT_RUNTIME:
            path = layout.runtime_path(session_dir_)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            layout.atomic_write_text(path, text, fsync=fsync)
        elif component == layout.COMPONENT_PLUGINS:
            assert isinstance(payload, dict)
            for plugin_name, state in payload.items():
                file_path = layout.plugin_state_path(session_dir_, plugin_name)
                wrapped = {
                    "version": layout.PLUGIN_FILE_VERSION,
                    "plugin_name": plugin_name,
                    "state": state.get("state", state),
                }
                if "plugin_class" in state:
                    wrapped["plugin_class"] = state["plugin_class"]
                text = json.dumps(wrapped, ensure_ascii=False, indent=2)
                layout.atomic_write_text(file_path, text, fsync=fsync)
        else:
            raise ValueError(f"Unknown session component: {component!r}")

    def _write_message_history(
        self,
        session_dir_: Path,
        job: WriteJob,
        *,
        fsync: bool,
    ) -> None:
        layout.append_jsonl(
            layout.message_history_path(session_dir_),
            job.message_history_entries,
            fsync=fsync,
        )

    def _write_manifest(self, job: WriteJob, wrote_components: list[str]) -> None:
        manifest_path = layout.manifest_path(job.session_dir)
        existing: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "could not parse existing manifest at %s; rewriting", manifest_path
                )
                existing = {}

        merged = dict(existing)
        merged.setdefault("version", layout.MANIFEST_VERSION)
        merged.update(job.manifest_patch)
        merged["updated_at"] = datetime.now().isoformat()

        components_present = set(merged.get("components_present", []))
        components_present.update(wrote_components)
        merged["components_present"] = sorted(components_present)

        text = json.dumps(merged, ensure_ascii=False, indent=2)
        layout.atomic_write_text(manifest_path, text, fsync=job.fsync)

    def _emit_failure(self, job: WriteJob, component: str, exc: BaseException) -> None:
        if self._event_bus is None:
            return
        try:
            ev = SessionWriteFailedEvent.create(
                session_id=job.session_dir.name,
                component=component,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._event_bus.publish(ev)
        except Exception:
            logger.exception("failed to emit session.write_failed event")
