"""SessionManager — central coordinator for session persistence.

Subscribes to boundary events on the live :class:`EventBus`, captures
synchronous snapshots of every relevant component, and hands write jobs to a
:class:`SessionWriter` daemon thread. Also installs a final-flush callback
into :class:`ExitHandler` so a SIGINT / SIGTERM / uncaught-exception path
still produces a complete on-disk state.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from hawi.events import (
    Event,
    EventBus,
    SessionLoadedEvent,
    SessionSwitchedEvent,
)
from hawi.utils.lifecycle import ExitHandler

from . import layout
from .markdown_export import (
    MarkdownExport,
    export_message_history_to_markdown,
    write_markdown_export_bundle,
)
from .message_history import message_history_entry_from_event, should_persist_message
from .lock import (
    SessionFileLock,
    SessionLockInfo,
    SessionLockedError,
    SessionLockUnavailable,
    make_lock_metadata,
    probe_session_lock,
    read_lock_owner,
)
from .writer import SessionWriter, WriteJob

if TYPE_CHECKING:
    from hawi.agent.agent import HawiAgent
    from hawi.agent.runner.runner import AgentRunner

logger = logging.getLogger(__name__)
SESSION_ID_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


# Boundary events whose firing triggers a checkpoint, mapped to the components
# that get re-snapshotted. Components are written through the SessionWriter,
# which handles atomic rename + (optionally) fsync.
EVENT_ROUTING: dict[str, tuple[str, ...]] = {
    "agent.run_start": (layout.COMPONENT_RUNTIME, layout.COMPONENT_QUEUES),
    "agent.system_prompt": (),
    "agent.context_injected": (),
    "agent.tool_runtime_context_injected": (),
    "agent.tool_call": (layout.COMPONENT_RUNTIME,),
    "agent.tool_result": (
        layout.COMPONENT_CONTEXT,
        layout.COMPONENT_RUNTIME,
        layout.COMPONENT_PLUGINS,
    ),
    "agent.run_stop": (
        layout.COMPONENT_CONTEXT,
        layout.COMPONENT_RUNTIME,
        layout.COMPONENT_PLUGINS,
    ),
    "agent.message_added": (layout.COMPONENT_CONTEXT,),
    "agent.error": (layout.COMPONENT_RUNTIME,),
    "agent.interrupt": (layout.COMPONENT_RUNTIME,),
    "agent.compact_start": (layout.COMPONENT_RUNTIME,),
    "agent.compact_stop": (layout.COMPONENT_CONTEXT,),
    "model.retry": (layout.COMPONENT_RUNTIME,),
    "model.error": (layout.COMPONENT_RUNTIME,),
    "runner.enqueue": (layout.COMPONENT_QUEUES,),
    "runner.dequeue": (layout.COMPONENT_QUEUES,),
    "runner.interrupt": (layout.COMPONENT_RUNTIME,),
    "plugin.event": (),
    "plugin.message": (),
    "plugin.status": (),
    "plugin.tool_progress": (),
    "plugin.artifact.upsert": (),
    "plugin.artifact.delta": (),
    "plugin.artifact.remove": (),
    "plugin.artifact.clear": (),
    "subagent.created": (),
    "subagent.event": (),
    "subagent.closed": (),
}


@dataclass
class SessionMeta:
    """Lightweight metadata for session listing UI."""

    session_id: str
    name: str
    created_at: str
    updated_at: str
    last_checkpoint_event: str | None
    components_present: list[str]
    locked: bool = False
    lock_owner: dict[str, Any] | None = None
    gui_launch_profile: dict[str, Any] | None = None
    last_cwd: str | None = None


@dataclass
class SessionContextBranchResult:
    """Metadata returned by session fork/rewind-at-message operations."""

    session_id: str
    source_session_id: str | None = None
    message_index: int | None = None
    context_message_id: str | None = None
    target_role: str | None = None
    boundary_index: int | None = None
    popped_user_message: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"session_id": self.session_id}
        if self.source_session_id is not None:
            payload["source_session_id"] = self.source_session_id
        if self.message_index is not None:
            payload["message_index"] = self.message_index
        if self.context_message_id is not None:
            payload["context_message_id"] = self.context_message_id
        if self.target_role is not None:
            payload["target_role"] = self.target_role
        if self.boundary_index is not None:
            payload["boundary_index"] = self.boundary_index
        if self.popped_user_message is not None:
            payload["popped_user_message"] = self.popped_user_message
        return payload


class SessionManager:
    """Coordinator that ties agent state to per-session disk layout.

    Typical lifecycle::

        sm = SessionManager()
        sm.attach(agent, runner)
        sm.new_session(name="my chat")
        # ... agent runs, checkpoints flow automatically ...
        sm.save_now()             # or rely on exit hook
        sm.detach()
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        writer: SessionWriter | None = None,
        time_provider: Callable[[], float] = time.time,
        keep_session_system_prompt: bool = True,
        manifest_metadata_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._root = Path(root).expanduser() if root else layout.DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._writer_owned = writer is None
        self._writer = writer or SessionWriter()
        self._time = time_provider
        self._keep_session_system_prompt = keep_session_system_prompt
        self._manifest_metadata_provider = manifest_metadata_provider
        self._lock = threading.RLock()

        self._agent: HawiAgent | None = None
        self._runner: AgentRunner | None = None
        self._event_bus: EventBus | None = None
        self._session_id: str | None = None
        self._session_name: str | None = None
        self._session_created_at: str | None = None
        self._session_has_visible_messages = False
        self._manager_id = uuid.uuid4().hex
        self._session_lock: SessionFileLock | None = None
        self._exit_hook_registered = False
        self._subscribed_event_types: tuple[str, ...] = ()

    # --- lifecycle -------------------------------------------------------

    def attach(
        self,
        agent: HawiAgent,
        runner: AgentRunner | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        """Wire SessionManager into an agent + runner.

        Subscribes to boundary events, starts the writer thread, and registers
        the final-flush exit hook.
        """
        with self._lock:
            self._agent = agent
            self._runner = runner
            bus = event_bus or getattr(agent, "_event_bus", None) or getattr(
                agent, "event_bus", None
            )
            self._event_bus = bus
            self._writer._event_bus = bus

            if self._writer_owned:
                self._writer.start()

            if bus is not None:
                self._subscribe(bus)
            self._configure_subagent_storage()

            if not self._exit_hook_registered:
                ExitHandler.get_instance().register_last(
                    self._final_flush, name="session-manager-final-flush"
                )
                self._exit_hook_registered = True

    def detach(self) -> None:
        """Unsubscribe from events and (if we own the writer) stop it."""
        with self._lock:
            if self._event_bus is not None and self._subscribed_event_types:
                try:
                    self._event_bus.unsubscribe(self._on_event)
                except Exception:
                    logger.debug("failed to unsubscribe", exc_info=True)
            self._subscribed_event_types = ()
            self._event_bus = None
            self._writer._event_bus = None
            self._agent = None
            self._runner = None
            if self._exit_hook_registered:
                ExitHandler.get_instance().unregister(self._final_flush)
                self._exit_hook_registered = False
            if self._writer_owned:
                self._writer.shutdown(timeout=5.0)
            else:
                self._writer.wait_idle(timeout=5.0)
            self._release_current_session_lock()

    # --- session API -----------------------------------------------------

    def new_session(
        self,
        name: str | None = None,
        *,
        session_id: str | None = None,
    ) -> str:
        """Create an in-memory session and make it current.

        The session is materialized on disk lazily, once it has at least one
        user-visible message. This keeps empty "New" clicks and startup
        placeholders out of the session directory.
        """
        session_id = session_id or self._new_unique_session_id()
        session_dir = layout.session_dir(self._root, session_id)
        if session_dir.exists():
            raise FileExistsError(f"session already exists: {session_id}")
        with self._lock:
            self._writer.wait_idle(timeout=10.0)
            self._release_current_session_lock()
            self._session_id = session_id
            self._session_name = name or session_id
            self._session_created_at = datetime.now().isoformat()
            self._session_has_visible_messages = False
            self._reset_agent_system_prompt_injection_hooks()
            self._set_agent_system_prompt_hooks_suppressed(False)
            self._configure_subagent_storage()
        return session_id

    @property
    def current_session_id(self) -> str | None:
        return self._session_id

    def list_sessions(self) -> list[SessionMeta]:
        """Return manifest metadata for every session under ``root``."""
        out: list[SessionMeta] = []
        if not self._root.exists():
            return out
        for child in sorted(self._root.iterdir(), key=lambda p: p.name):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            mp = layout.manifest_path(child)
            if not mp.exists():
                continue
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("skipping unreadable session manifest %s", mp)
                continue
            if not self._session_dir_has_visible_messages(child):
                continue
            lock_info = self._session_lock_info(data.get("session_id", child.name))
            out.append(
                SessionMeta(
                    session_id=data.get("session_id", child.name),
                    name=data.get("name", child.name),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    last_checkpoint_event=data.get("last_checkpoint_event"),
                    components_present=list(data.get("components_present", [])),
                    locked=lock_info.locked,
                    lock_owner=lock_info.owner if lock_info.locked else None,
                    gui_launch_profile=(
                        data.get("gui_launch_profile")
                        if isinstance(data.get("gui_launch_profile"), dict)
                        else None
                    ),
                    last_cwd=(
                        data.get("last_cwd")
                        if isinstance(data.get("last_cwd"), str)
                        else None
                    ),
                )
            )
        return sorted(out, key=lambda m: _parse_iso_timestamp(m.created_at), reverse=True)

    def load_session(self, session_id: str) -> None:
        """Load a session's on-disk state into the attached agent."""
        if self._agent is None:
            raise RuntimeError("SessionManager.load_session requires attach() first")
        session_dir = layout.session_dir(self._root, session_id)
        manifest_path = layout.manifest_path(session_dir)
        if not manifest_path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")

        next_lock: SessionFileLock | None = None
        previous_lock: SessionFileLock | None = None
        previous_session_id = self._session_id
        previous_session_name = self._session_name
        previous_created_at = self._session_created_at
        previous_has_visible_messages = self._session_has_visible_messages
        previous_system_prompt_hook_suppressed = (
            self._agent_system_prompt_hooks_suppressed()
        )
        try:
            with self._lock:
                if session_id != self._session_id:
                    next_lock = self._acquire_session_lock(session_id)
                    previous_lock = self._session_lock
                else:
                    self._ensure_current_session_lock()

                self._session_id = session_id
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self._session_name = manifest.get("name", session_id)
                self._session_created_at = (
                    manifest.get("created_at") or datetime.now().isoformat()
                )

                loaded: list[str] = []
                loaded_system_prompt = False

                ctx_path = layout.context_path(session_dir)
                if ctx_path.exists():
                    ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
                    self._agent.context.load_snapshot(ctx_data)
                    loaded.append(layout.COMPONENT_CONTEXT)
                    loaded_system_prompt = "system_prompt" in ctx_data

                queues_path = layout.queues_path(session_dir)
                if queues_path.exists() and self._runner is not None:
                    queues_data = json.loads(queues_path.read_text(encoding="utf-8"))
                    qm = self._runner_queue_manager()
                    if qm is not None and "runner" in queues_data:
                        qm.load_snapshot(queues_data["runner"])
                        qm.rebind_event_bus(self._event_bus)
                    if "pending_steer_inputs" in queues_data:
                        self._agent.load_steer(queues_data["pending_steer_inputs"])
                    # Restore runner control state (pause/resume) for v2+
                    runner_control = queues_data.get("runner_control")
                    if runner_control is not None and isinstance(runner_control, dict):
                        if runner_control.get("paused") and hasattr(self._runner, "pause"):
                            reason = runner_control.get("pause_reason", "session_restored")
                            self._runner.pause(
                                reason,
                                error_message=runner_control.get("last_error_message"),
                            )
                    loaded.append(layout.COMPONENT_QUEUES)

                runtime_path = layout.runtime_path(session_dir)
                if runtime_path.exists():
                    runtime_data = json.loads(runtime_path.read_text(encoding="utf-8"))
                    self._agent.load_runtime(runtime_data)
                    loaded.append(layout.COMPONENT_RUNTIME)

                # Synthesize error tool results for any in-flight tool calls that
                # were interrupted by the crash. Done at load time (not deferred to
                # the next run) so the context is provider-valid immediately and
                # GUI snapshots produced right after load show no orphan tool
                # nodes. add_missing_tool_results scans messages directly and is
                # idempotent.
                recovered = self._agent.context.add_missing_tool_results(
                    "Tool call interrupted before completion (reason: session restored)."
                )
                if recovered:
                    # Clear runtime tool-call list — these are now "answered" with
                    # synthetic results; nothing live should pick them up.
                    self._agent._current_tool_calls = []
                    self._agent._last_unsent_tool_results = []
                    logger.info(
                        "session %s load: recovered %d interrupted tool calls",
                        session_id,
                        len(recovered),
                    )

                plugins_dir = layout.plugins_dir(session_dir)
                if plugins_dir.exists():
                    self._load_plugins(plugins_dir, manifest)
                    loaded.append(layout.COMPONENT_PLUGINS)

                suppress_system_prompt_hooks = (
                    self._keep_session_system_prompt and loaded_system_prompt
                )
                if not suppress_system_prompt_hooks:
                    self._reset_agent_system_prompt_injection_hooks()
                self._set_agent_system_prompt_hooks_suppressed(
                    suppress_system_prompt_hooks
                )
                if self._event_bus is not None:
                    self._event_bus.publish(
                        SessionLoadedEvent.create(
                            session_id=session_id, components_loaded=loaded
                        )
                    )
                self._session_has_visible_messages = self._session_dir_has_visible_messages(
                    session_dir
                )
                self._configure_subagent_storage()
                if next_lock is not None:
                    self._session_lock = next_lock
                    if previous_lock is not None:
                        previous_lock.release()
                    next_lock = None
        except Exception:
            if next_lock is not None:
                next_lock.release()
            self._session_id = previous_session_id
            self._session_name = previous_session_name
            self._session_created_at = previous_created_at
            self._session_has_visible_messages = previous_has_visible_messages
            self._set_agent_system_prompt_hooks_suppressed(
                previous_system_prompt_hook_suppressed
            )
            raise

    def read_message_history(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read append-only, user-visible message history for a session."""
        sid = session_id or self._session_id
        if sid is None:
            return []
        entries = layout.read_jsonl(
            layout.message_history_path(layout.session_dir(self._root, sid))
        )
        if sid == self._session_id and self._agent is not None:
            messages = getattr(self._agent.context, "messages", None)
        else:
            context_snapshot = self._read_context_snapshot(sid)
            messages = context_snapshot.get("messages")
        if isinstance(messages, list):
            return self._history_with_context_indices(entries, messages)
        return entries

    def export_markdown(
        self,
        session_id: str | None = None,
        *,
        model: str | None = None,
        title: str | None = None,
    ) -> MarkdownExport:
        """Create a session-internal Markdown export bundle."""
        sid = session_id or self._session_id
        if sid is None:
            raise RuntimeError("No active session to export")
        if session_id is None or session_id == self._session_id:
            self.save_now()
        session_dir = layout.session_dir(self._root, sid)
        history_path = layout.message_history_path(session_dir)
        message_history = layout.read_jsonl(history_path)
        context_snapshot = self._read_context_snapshot(sid)
        manifest = self._read_manifest(sid)
        export = export_message_history_to_markdown(
            message_history,
            kind="session",
            subject_id=sid,
            title=title or f"Hawi Session {sid}",
            model=model,
            system_prompt=context_snapshot.get("system_prompt"),
            metadata={
                "name": manifest.get("name"),
                "active_plugins": manifest.get("active_plugins"),
                "message_count": len(message_history),
            },
            raw_history_path=str(history_path),
        )
        return write_markdown_export_bundle(
            export,
            export_dir=layout.export_dir(session_dir, export.export_id),
            source_jsonl_path=history_path,
            message_history=message_history,
        )

    def switch_to(self, session_id: str) -> None:
        """Save the current session, then load another."""
        previous = self._session_id
        if previous is not None and previous != session_id:
            self.save_now()
        self.load_session(session_id)
        if self._event_bus is not None:
            self._event_bus.publish(
                SessionSwitchedEvent.create(
                    from_session_id=previous, to_session_id=session_id
                )
            )

    def fork_session(self, session_id: str | None = None, name: str | None = None) -> str:
        """Copy an existing session into a new unlocked session and load it.

        The source session is treated as read-only and is deliberately not
        locked, so a second Hawi engine can fork a session that another engine
        is currently using instead of joining it.
        """
        return self._fork_session(
            session_id=session_id,
            name=name,
            after_message_index=None,
        ).session_id

    def fork_session_after_message(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
        after_message_index: int,
    ) -> SessionContextBranchResult:
        """Fork a session and truncate the fork at a user/assistant message."""
        return self._fork_session(
            session_id=session_id,
            name=name,
            after_message_index=after_message_index,
        )

    def fork_session_after_message_id(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
        context_message_id: str,
    ) -> SessionContextBranchResult:
        """Fork a session and truncate the fork at a stable message id."""
        return self._fork_session(
            session_id=session_id,
            name=name,
            after_message_index=None,
            after_context_message_id=context_message_id,
        )

    def _fork_session(
        self,
        *,
        session_id: str | None,
        name: str | None,
        after_message_index: int | None,
        after_context_message_id: str | None = None,
    ) -> SessionContextBranchResult:
        source_id = session_id or self._session_id
        if source_id is None:
            raise RuntimeError("No session available to fork")
        if self._session_id is not None:
            self.save_now()

        source_dir = layout.session_dir(self._root, source_id)
        if not layout.manifest_path(source_dir).exists():
            raise FileNotFoundError(f"session not found: {source_id}")

        fork_id = self._new_unique_session_id()
        fork_dir = layout.session_dir(self._root, fork_id)
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{fork_id}.",
                suffix=".fork",
                dir=str(self._root),
            )
        )
        branch_result: SessionContextBranchResult | None = None
        try:
            shutil.rmtree(temp_dir)
            shutil.copytree(
                source_dir,
                temp_dir,
                ignore=shutil.ignore_patterns(
                    layout.SESSION_LOCK_FILENAME,
                    "*.tmp",
                ),
            )
            if after_message_index is not None or after_context_message_id is not None:
                branch_result = self._apply_context_message_boundary(
                    temp_dir,
                    session_id=fork_id,
                    source_session_id=source_id,
                    message_index=after_message_index,
                    context_message_id=after_context_message_id,
                )
            self._rewrite_fork_manifest(
                temp_dir,
                fork_id=fork_id,
                source_id=source_id,
                name=name,
                branch_result=branch_result,
            )
            temp_dir.rename(fork_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(fork_dir, ignore_errors=True)
            raise

        self.load_session(fork_id)
        return branch_result or SessionContextBranchResult(
            session_id=fork_id,
            source_session_id=source_id,
        )

    def rewind_session_after_message(
        self,
        *,
        after_message_index: int,
    ) -> SessionContextBranchResult:
        """Rewind the current session to a user/assistant message boundary."""
        if self._session_id is None:
            raise RuntimeError("No active session to rewind")
        if self._agent is None:
            raise RuntimeError("SessionManager.rewind_session requires attach() first")

        self.save_now()
        session_id = self._session_id
        session_dir = layout.session_dir(self._root, session_id)
        if not layout.manifest_path(session_dir).exists():
            raise FileNotFoundError(f"session not found: {session_id}")

        self._ensure_current_session_lock()
        result = self._apply_context_message_boundary(
            session_dir,
            session_id=session_id,
            source_session_id=session_id,
            message_index=after_message_index,
            manifest_event="session_rewind",
        )
        self.load_session(session_id)
        self._session_has_visible_messages = self._session_dir_has_visible_messages(
            session_dir
        )
        return result

    def rewind_session_after_message_id(
        self,
        *,
        context_message_id: str,
    ) -> SessionContextBranchResult:
        """Rewind the current session to a stable message id boundary."""
        if self._session_id is None:
            raise RuntimeError("No active session to rewind")
        if self._agent is None:
            raise RuntimeError("SessionManager.rewind_session requires attach() first")

        self.save_now()
        session_id = self._session_id
        session_dir = layout.session_dir(self._root, session_id)
        if not layout.manifest_path(session_dir).exists():
            raise FileNotFoundError(f"session not found: {session_id}")

        self._ensure_current_session_lock()
        result = self._apply_context_message_boundary(
            session_dir,
            session_id=session_id,
            source_session_id=session_id,
            message_index=None,
            context_message_id=context_message_id,
            manifest_event="session_rewind",
        )
        self.load_session(session_id)
        self._session_has_visible_messages = self._session_dir_has_visible_messages(
            session_dir
        )
        return result

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a session directory.

        If the deleted session is currently active, the manager has no current
        session afterwards.
        """
        delete_lock: SessionFileLock | None = None
        if session_id != self._session_id:
            session_dir = layout.session_dir(self._root, session_id)
            if session_dir.exists():
                delete_lock = self._acquire_session_lock(session_id)
        layout.remove_session_dir(layout.session_dir(self._root, session_id))
        if delete_lock is not None:
            delete_lock.release()
        with self._lock:
            if self._session_id == session_id:
                self._release_current_session_lock()
                self._session_id = None
                self._session_name = None

    def save_now(self, *, fsync: bool = True) -> None:
        """Capture all components synchronously, enqueue + wait for the writer."""
        if self._session_id is None:
            return
        if not self._current_session_has_visible_messages():
            self._remove_current_session_dir_if_empty()
            return
        self._ensure_current_session_lock()
        snapshots, manifest_patch = self._capture_all("save_now")
        job = WriteJob(
            session_dir=layout.session_dir(self._root, self._session_id),
            snapshots=snapshots,
            manifest_patch=manifest_patch,
            fsync=fsync,
            component_set_key="save_now",
        )
        self._writer.submit(job)
        self._writer.wait_idle(timeout=10.0)

    # --- internal: subscriptions + capture ------------------------------

    def _subscribe(self, bus: EventBus) -> None:
        # Use blocking subscription so snapshots run on the event-emitting
        # thread BEFORE the agent can mutate state. The actual disk I/O is
        # offloaded to the writer thread; what runs synchronously is just a
        # cheap dataclass-to-dict capture.
        event_types = list(EVENT_ROUTING.keys())
        try:
            bus.subscribe_blocking(self._on_event, event_types)
        except Exception:
            logger.exception("failed to subscribe SessionManager to event bus")
            self._subscribed_event_types = ()
            return
        self._subscribed_event_types = tuple(event_types)

    def _on_event(self, event: Event) -> None:
        if self._session_id is None:
            return
        components = EVENT_ROUTING.get(event.type)
        if components is None:
            return
        message_history_entries: list[dict[str, Any]] = []
        entry = message_history_entry_from_event(event)
        if entry is not None:
            message_history_entries.append(entry)
            self._session_has_visible_messages = True
        if (
            not message_history_entries
            and not self._current_session_has_visible_messages()
        ):
            return
        try:
            self._ensure_current_session_lock()
            snapshots, manifest_patch = self._capture(event.type, components)
        except Exception:
            logger.exception(
                "session capture failed during event %s; skipping checkpoint",
                event.type,
            )
            return
        component_names = set(components)
        if message_history_entries:
            component_names.add(layout.COMPONENT_MESSAGE_HISTORY)
        job = WriteJob(
            session_dir=layout.session_dir(self._root, self._session_id),
            snapshots=snapshots,
            message_history_entries=message_history_entries,
            manifest_patch=manifest_patch,
            fsync=False,
            component_set_key=",".join(sorted(component_names)),
            drop_on_overflow=not message_history_entries,
        )
        self._writer.submit(job)

    def _capture_all(self, trigger: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._capture(
            trigger,
            (
                layout.COMPONENT_CONTEXT,
                layout.COMPONENT_QUEUES,
                layout.COMPONENT_RUNTIME,
                layout.COMPONENT_PLUGINS,
            ),
        )

    def _capture(
        self, trigger: str, components: tuple[str, ...] | list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._agent is None:
            raise RuntimeError("SessionManager not attached to an agent")
        snapshots: dict[str, Any] = {}

        if layout.COMPONENT_CONTEXT in components:
            snapshots[layout.COMPONENT_CONTEXT] = self._agent.context.snapshot()

        if layout.COMPONENT_QUEUES in components:
            queues_payload: dict[str, Any] = {"version": layout.QUEUES_VERSION}
            qm = self._runner_queue_manager()
            if qm is not None:
                queues_payload["runner"] = qm.snapshot()
            queues_payload["pending_steer_inputs"] = self._agent.snapshot_steer()
            queues_payload["pending_audit_tool_calls" if layout.QUEUES_VERSION >= 2 else "pending_audit_tool_calls"] = [
                {
                    "tool_call_id": p.tool_call_id,
                    "tool_name": p.tool_name,
                    "arguments": p.arguments,
                    "requested_at": p.requested_at,
                }
                for p in self._agent.context.get_pending_tool_calls()
            ]
            # Save runner control state (pause/resume) for v2+
            if layout.QUEUES_VERSION >= 2 and self._runner is not None:
                queues_payload["runner_control"] = self._runner.control_snapshot()
            snapshots[layout.COMPONENT_QUEUES] = queues_payload

        if layout.COMPONENT_RUNTIME in components:
            snapshots[layout.COMPONENT_RUNTIME] = self._agent.snapshot_runtime()

        if layout.COMPONENT_PLUGINS in components:
            plugin_states: dict[str, Any] = {}
            for plugin in self._iter_plugins():
                try:
                    state = plugin.save_state()
                except Exception:
                    logger.exception(
                        "plugin %s save_state raised; skipping",
                        getattr(plugin, "plugin_name", plugin.__class__.__name__),
                    )
                    continue
                if state is None:
                    continue
                key = getattr(plugin, "plugin_name", plugin.__class__.__name__)
                plugin_states[key] = {
                    "state": state,
                    "plugin_class": (
                        f"{plugin.__class__.__module__}.{plugin.__class__.__name__}"
                    ),
                }
            snapshots[layout.COMPONENT_PLUGINS] = plugin_states

        manifest_patch = {
            "session_id": self._session_id,
            "name": self._session_name or self._session_id,
            "created_at": self._session_created_at or datetime.now().isoformat(),
            "last_checkpoint_event": trigger,
            "active_plugins": self._active_plugin_names(),
        }
        if self._manifest_metadata_provider is not None:
            try:
                extra_metadata = self._manifest_metadata_provider()
            except Exception:
                logger.exception("session manifest metadata provider failed")
            else:
                if isinstance(extra_metadata, dict):
                    manifest_patch.update(extra_metadata)
        return snapshots, manifest_patch

    def _final_flush(self) -> None:
        """Exit hook: synchronously capture once more then drain the writer.

        No-op when the manager has already been detached or has no current
        session — the relevant state is already on disk.
        """
        try:
            if (
                self._session_id is not None
                and self._agent is not None
                and self._current_session_has_visible_messages()
            ):
                self._ensure_current_session_lock()
                snapshots, manifest_patch = self._capture_all("exit")
                job = WriteJob(
                    session_dir=layout.session_dir(self._root, self._session_id),
                    snapshots=snapshots,
                    manifest_patch=manifest_patch,
                    fsync=True,
                    component_set_key="final_flush",
                )
                self._writer.submit(job)
            elif self._session_id is not None:
                self._remove_current_session_dir_if_empty()
        except Exception:
            logger.exception("session final flush capture failed")
        finally:
            try:
                if self._writer_owned:
                    self._writer.shutdown(timeout=5.0)
                else:
                    self._writer.wait_idle(timeout=5.0)
                self._release_current_session_lock()
            except Exception:
                logger.exception("session writer shutdown raised")

    # --- helpers ---------------------------------------------------------

    def _new_unique_session_id(self) -> str:
        while True:
            timestamp = datetime.now().strftime(SESSION_ID_TIMESTAMP_FORMAT)
            session_id = f"session-{timestamp}-{uuid.uuid4().hex[:6]}"
            if not layout.session_dir(self._root, session_id).exists():
                return session_id

    def _apply_context_message_boundary(
        self,
        session_dir: Path,
        *,
        session_id: str,
        source_session_id: str,
        message_index: int | None,
        context_message_id: str | None = None,
        manifest_event: str | None = None,
    ) -> SessionContextBranchResult:
        ctx_path = layout.context_path(session_dir)
        if not ctx_path.exists():
            raise FileNotFoundError(
                f"context snapshot not found for session: {source_session_id}"
            )

        from hawi.agent.context import AgentContext

        ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
        context = AgentContext()
        context.load_snapshot(ctx_data)
        original_messages = deepcopy(context.messages)
        if context_message_id is not None:
            boundary_result = context.truncate_after_message_id(context_message_id)
        elif message_index is not None:
            boundary_result = context.truncate_after_message(message_index)
        else:
            raise ValueError("message_index or context_message_id is required")

        layout.atomic_write_text(
            ctx_path,
            json.dumps(context.snapshot(), ensure_ascii=False, indent=2),
            fsync=True,
        )

        history_path = layout.message_history_path(session_dir)
        history = layout.read_jsonl(history_path)
        pruned_history = self._prune_history_to_context_boundary(
            history,
            original_messages,
            boundary_result.boundary_index,
        )
        layout.write_jsonl(history_path, pruned_history, fsync=True)

        if manifest_event is not None:
            self._patch_manifest(
                session_dir,
                {
                    "last_checkpoint_event": manifest_event,
                    "rewound_after_message_index": boundary_result.message_index,
                    "rewound_after_context_message_id": (
                        boundary_result.context_message_id
                    ),
                    "rewind_boundary_index": boundary_result.boundary_index,
                    "rewind_target_role": boundary_result.target_role,
                },
                components=[
                    layout.COMPONENT_CONTEXT,
                    layout.COMPONENT_MESSAGE_HISTORY,
                ],
            )

        return SessionContextBranchResult(
            session_id=session_id,
            source_session_id=source_session_id,
            message_index=boundary_result.message_index,
            context_message_id=boundary_result.context_message_id,
            target_role=boundary_result.target_role,
            boundary_index=boundary_result.boundary_index,
            popped_user_message=(
                deepcopy(boundary_result.popped_user_message)
                if boundary_result.popped_user_message is not None
                else None
            ),
        )

    def _prune_history_to_context_boundary(
        self,
        entries: list[dict[str, Any]],
        messages: list[Any],
        boundary_index: int,
    ) -> list[dict[str, Any]]:
        if boundary_index <= 0:
            return []
        annotated = self._history_with_context_indices(entries, messages)
        last_included_pos = -1
        for pos, entry in enumerate(annotated):
            context_index = entry.get("context_message_index")
            if isinstance(context_index, int) and context_index < boundary_index:
                last_included_pos = pos
        if last_included_pos < 0:
            return []
        return [
            self._without_context_message_index(entry)
            for entry in annotated[: last_included_pos + 1]
        ]

    @classmethod
    def _history_with_context_indices(
        cls,
        entries: list[dict[str, Any]],
        messages: list[Any],
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        cursor = 0
        for entry in entries:
            record = dict(entry)
            if record.get("role") in {"user", "assistant", "tool"}:
                match = cls._find_matching_context_message_by_id(
                    record,
                    messages,
                    start=cursor,
                )
                if match is None:
                    match = cls._find_matching_context_message(
                        record,
                        messages,
                        start=cursor,
                        strict_metadata=True,
                    )
                if match is None:
                    match = cls._find_matching_context_message(
                        record,
                        messages,
                        start=cursor,
                        strict_metadata=False,
                    )
                if match is not None:
                    record["context_message_index"] = match
                    message = messages[match]
                    if isinstance(message, dict):
                        context_message_id = message.get("context_message_id")
                        if isinstance(context_message_id, str):
                            record["context_message_id"] = context_message_id
                    cursor = match + 1
            annotated.append(record)
        return annotated

    @classmethod
    def _find_matching_context_message_by_id(
        cls,
        record: dict[str, Any],
        messages: list[Any],
        *,
        start: int,
    ) -> int | None:
        context_message_id = record.get("context_message_id")
        if not isinstance(context_message_id, str) or not context_message_id:
            return None
        for index in range(start, len(messages)):
            message = messages[index]
            if (
                isinstance(message, dict)
                and message.get("context_message_id") == context_message_id
            ):
                return index
        return None

    @classmethod
    def _find_matching_context_message(
        cls,
        record: dict[str, Any],
        messages: list[Any],
        *,
        start: int,
        strict_metadata: bool,
    ) -> int | None:
        for index in range(start, len(messages)):
            message = messages[index]
            if cls._history_record_matches_context_message(
                record,
                message,
                strict_metadata=strict_metadata,
            ):
                return index
        return None

    @staticmethod
    def _history_record_matches_context_message(
        record: dict[str, Any],
        message: Any,
        *,
        strict_metadata: bool,
    ) -> bool:
        if not isinstance(message, dict):
            return False
        if record.get("role") != message.get("role"):
            return False
        if record.get("content") != message.get("content"):
            return False
        if not strict_metadata:
            return True
        record_metadata = record.get("metadata")
        if not isinstance(record_metadata, dict):
            record_metadata = None
        message_metadata = message.get("metadata")
        if not isinstance(message_metadata, dict):
            message_metadata = None
        return record_metadata == message_metadata

    @staticmethod
    def _without_context_message_index(entry: dict[str, Any]) -> dict[str, Any]:
        copy = dict(entry)
        copy.pop("context_message_index", None)
        return copy

    def _patch_manifest(
        self,
        session_dir: Path,
        patch: dict[str, Any],
        *,
        components: list[str] | tuple[str, ...] = (),
    ) -> None:
        manifest_path = layout.manifest_path(session_dir)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        manifest.setdefault("version", layout.MANIFEST_VERSION)
        manifest.update(patch)
        manifest["updated_at"] = datetime.now().isoformat()
        components_present = set(manifest.get("components_present", []))
        components_present.update(components)
        manifest["components_present"] = sorted(components_present)
        layout.atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2),
            fsync=True,
        )

    def _rewrite_fork_manifest(
        self,
        session_dir: Path,
        *,
        fork_id: str,
        source_id: str,
        name: str | None,
        branch_result: SessionContextBranchResult | None = None,
    ) -> None:
        manifest_path = layout.manifest_path(session_dir)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        created_at = datetime.now().isoformat()
        source_name = manifest.get("name") or source_id
        manifest.update(
            {
                "version": layout.MANIFEST_VERSION,
                "session_id": fork_id,
                "name": name or f"{source_name} fork",
                "created_at": created_at,
                "updated_at": created_at,
                "forked_from_session_id": source_id,
                "forked_at": created_at,
                "last_checkpoint_event": "session_fork",
            }
        )
        if branch_result is not None:
            manifest.update(
                {
                    "forked_after_message_index": branch_result.message_index,
                    "forked_after_context_message_id": (
                        branch_result.context_message_id
                    ),
                    "fork_boundary_index": branch_result.boundary_index,
                    "fork_target_role": branch_result.target_role,
                }
            )
        layout.atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2),
            fsync=True,
        )

    def _session_lock_info(self, session_id: str) -> SessionLockInfo:
        session_dir = layout.session_dir(self._root, session_id)
        path = layout.session_lock_path(session_dir)
        if (
            self._session_lock is not None
            and self._session_lock.path.resolve() == path.resolve()
        ):
            return SessionLockInfo(
                locked=False,
                owner=read_lock_owner(path),
                owned_by_self=True,
            )
        return probe_session_lock(path, owner_token=self._manager_id)

    def _acquire_session_lock(self, session_id: str) -> SessionFileLock:
        session_dir = layout.session_dir(self._root, session_id)
        lock_path = layout.session_lock_path(session_dir)
        lock = SessionFileLock(
            lock_path,
            owner_token=self._manager_id,
            metadata=make_lock_metadata(self._manager_id),
        )
        try:
            return lock.acquire()
        except SessionLockUnavailable as exc:
            raise SessionLockedError(
                session_id,
                owner=read_lock_owner(lock_path),
            ) from exc

    def _ensure_current_session_lock(self) -> None:
        if self._session_id is None:
            return
        session_dir = layout.session_dir(self._root, self._session_id)
        lock_path = layout.session_lock_path(session_dir)
        if (
            self._session_lock is not None
            and self._session_lock.path.resolve() == lock_path.resolve()
        ):
            return
        previous_lock = self._session_lock
        self._session_lock = self._acquire_session_lock(self._session_id)
        if previous_lock is not None:
            previous_lock.release()

    def _release_current_session_lock(self) -> None:
        lock = self._session_lock
        self._session_lock = None
        if lock is not None:
            lock.release()

    def _current_session_has_visible_messages(self) -> bool:
        if self._session_has_visible_messages:
            return True
        if self._agent is not None and self._context_has_visible_messages():
            return True
        if self._session_id is None:
            return False
        session_dir = layout.session_dir(self._root, self._session_id)
        return self._session_dir_has_visible_messages(session_dir)

    def _context_has_visible_messages(self) -> bool:
        if self._agent is None:
            return False
        context = getattr(self._agent, "context", None)
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return False
        return self._messages_have_visible_entries(messages)

    def _session_dir_has_visible_messages(self, session_dir: Path) -> bool:
        history_path = layout.message_history_path(session_dir)
        try:
            if history_path.exists() and layout.read_jsonl(history_path):
                return True
        except (OSError, json.JSONDecodeError):
            logger.warning("could not inspect message history %s", history_path)

        ctx_path = layout.context_path(session_dir)
        if not ctx_path.exists():
            return False
        try:
            ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("could not inspect context snapshot %s", ctx_path)
            return False
        messages = ctx_data.get("messages")
        return isinstance(messages, list) and self._messages_have_visible_entries(
            messages
        )

    def _remove_current_session_dir_if_empty(self) -> None:
        if self._session_id is None:
            return
        session_dir = layout.session_dir(self._root, self._session_id)
        if session_dir.exists() and not self._session_dir_has_visible_messages(
            session_dir
        ):
            layout.remove_session_dir(session_dir)

    @classmethod
    def _messages_have_visible_entries(cls, messages: list[Any]) -> bool:
        return any(cls._message_is_visible_entry(message) for message in messages)

    @classmethod
    def _message_is_visible_entry(cls, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in {"user", "assistant", "tool"}:
            return False
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return False
        return should_persist_message(message.get("metadata"))

    def _runner_queue_manager(self) -> Any:
        if self._runner is None:
            return None
        for attr in ("queue_manager", "queues", "_queue_manager"):
            qm = getattr(self._runner, attr, None)
            if qm is not None and hasattr(qm, "snapshot"):
                return qm
        return None

    def _iter_plugins(self) -> list[Any]:
        if self._agent is None:
            return []
        manager = getattr(self._agent, "_plugin_manager", None) or getattr(
            self._agent, "plugins", None
        )
        if manager is None:
            return []
        for attr in ("plugins", "_plugins"):
            pl = getattr(manager, attr, None)
            if isinstance(pl, dict):
                return list(pl.values())
            if isinstance(pl, list):
                return list(pl)
        return []

    def _active_plugin_names(self) -> list[str]:
        return [
            getattr(p, "plugin_name", p.__class__.__name__) for p in self._iter_plugins()
        ]

    def _read_context_snapshot(self, session_id: str) -> dict[str, Any]:
        ctx_path = layout.context_path(layout.session_dir(self._root, session_id))
        if not ctx_path.exists():
            if session_id == self._session_id and self._agent is not None:
                return self._agent.context.snapshot()
            return {}
        try:
            data = json.loads(ctx_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to read context snapshot %s", ctx_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _read_manifest(self, session_id: str) -> dict[str, Any]:
        manifest_path = layout.manifest_path(layout.session_dir(self._root, session_id))
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to read session manifest %s", manifest_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _agent_system_prompt_hooks_suppressed(self) -> bool:
        if self._agent is None:
            return False
        return bool(getattr(self._agent, "_suppress_system_prompt_hooks", False))

    def _set_agent_system_prompt_hooks_suppressed(self, suppress: bool) -> None:
        if self._agent is None:
            return
        setter = getattr(self._agent, "suppress_system_prompt_hooks", None)
        if callable(setter):
            setter(suppress)
        else:
            setattr(self._agent, "_suppress_system_prompt_hooks", suppress)

    def _reset_agent_system_prompt_injection_hooks(self) -> None:
        if self._agent is None:
            return
        resetter = getattr(self._agent, "reset_system_prompt_injection_hooks", None)
        if callable(resetter):
            resetter()

    def _configure_subagent_storage(self) -> None:
        subagents = getattr(self._agent, "subagents", None)
        if subagents is None or not hasattr(subagents, "configure_session_storage"):
            return
        subagents.configure_session_storage(
            root=self._root,
            session_id_provider=lambda: self._session_id,
        )

    def _load_plugins(self, plugins_dir_: Path, manifest: dict[str, Any]) -> None:
        existing_by_name = {
            getattr(p, "plugin_name", p.__class__.__name__): p
            for p in self._iter_plugins()
        }
        for state_file in plugins_dir_.glob("*.json"):
            try:
                wrapper = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("skipping unreadable plugin state %s", state_file)
                continue
            name = wrapper.get("plugin_name", state_file.stem)
            plugin = existing_by_name.get(name)
            if plugin is None:
                logger.warning(
                    "session manifest references plugin %r but it is not active; skipping",
                    name,
                )
                continue
            try:
                plugin.load_state(wrapper.get("state", {}))
            except Exception:
                logger.exception("plugin %s load_state raised", name)

        # Warn on plugins that were active when saved but are not loadable now.
        active_at_save = set(manifest.get("active_plugins", []))
        for missing in active_at_save - set(existing_by_name):
            logger.warning(
                "manifest active plugin %r is not present in the current "
                "agent; that plugin will start fresh",
                missing,
            )


def _parse_iso_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0
