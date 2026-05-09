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
import threading
import time
import uuid
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
from .writer import SessionWriter, WriteJob

if TYPE_CHECKING:
    from hawi.agent.agent import HawiAgent
    from hawi.agent.scheduler.scheduler import HawiScheduler

logger = logging.getLogger(__name__)


# Boundary events whose firing triggers a checkpoint, mapped to the components
# that get re-snapshotted. Components are written through the SessionWriter,
# which handles atomic rename + (optionally) fsync.
EVENT_ROUTING: dict[str, tuple[str, ...]] = {
    "agent.run_start": (layout.COMPONENT_RUNTIME, layout.COMPONENT_QUEUES),
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
    "scheduler.enqueue": (layout.COMPONENT_QUEUES,),
    "scheduler.dequeue": (layout.COMPONENT_QUEUES,),
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


class SessionManager:
    """Coordinator that ties agent state to per-session disk layout.

    Typical lifecycle::

        sm = SessionManager()
        sm.attach(agent, scheduler)
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
    ) -> None:
        self._root = Path(root).expanduser() if root else layout.DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._writer_owned = writer is None
        self._writer = writer or SessionWriter()
        self._time = time_provider
        self._lock = threading.RLock()

        self._agent: HawiAgent | None = None
        self._scheduler: HawiScheduler | None = None
        self._event_bus: EventBus | None = None
        self._session_id: str | None = None
        self._session_name: str | None = None
        self._exit_hook_registered = False
        self._subscribed_event_types: tuple[str, ...] = ()

    # --- lifecycle -------------------------------------------------------

    def attach(
        self,
        agent: HawiAgent,
        scheduler: HawiScheduler | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        """Wire SessionManager into an agent + scheduler.

        Subscribes to boundary events, starts the writer thread, and registers
        the final-flush exit hook.
        """
        with self._lock:
            self._agent = agent
            self._scheduler = scheduler
            bus = event_bus or getattr(agent, "_event_bus", None) or getattr(
                agent, "event_bus", None
            )
            self._event_bus = bus
            if self._writer._event_bus is None and bus is not None:
                self._writer._event_bus = bus

            if self._writer_owned:
                self._writer.start()

            if bus is not None:
                self._subscribe(bus)

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
            self._agent = None
            self._scheduler = None
            if self._exit_hook_registered:
                ExitHandler.get_instance().unregister(self._final_flush)
                self._exit_hook_registered = False
            if self._writer_owned:
                self._writer.shutdown(timeout=5.0)

    # --- session API -----------------------------------------------------

    def new_session(self, name: str | None = None) -> str:
        """Create an empty session directory and make it the current session."""
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._session_id = session_id
            self._session_name = name or session_id
            session_dir = layout.session_dir(self._root, session_id)
            layout.ensure_session_layout(session_dir)
            now = datetime.now().isoformat()
            manifest = {
                "version": layout.MANIFEST_VERSION,
                "session_id": session_id,
                "name": self._session_name,
                "created_at": now,
                "updated_at": now,
                "last_checkpoint_event": None,
                "components_present": [],
                "active_plugins": self._active_plugin_names(),
            }
            layout.atomic_write_text(
                layout.manifest_path(session_dir),
                json.dumps(manifest, ensure_ascii=False, indent=2),
                fsync=True,
            )
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
            out.append(
                SessionMeta(
                    session_id=data.get("session_id", child.name),
                    name=data.get("name", child.name),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    last_checkpoint_event=data.get("last_checkpoint_event"),
                    components_present=list(data.get("components_present", [])),
                )
            )
        return out

    def load_session(self, session_id: str) -> None:
        """Load a session's on-disk state into the attached agent."""
        if self._agent is None:
            raise RuntimeError("SessionManager.load_session requires attach() first")
        session_dir = layout.session_dir(self._root, session_id)
        manifest_path = layout.manifest_path(session_dir)
        if not manifest_path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")

        with self._lock:
            self._session_id = session_id
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._session_name = manifest.get("name", session_id)

            loaded: list[str] = []

            ctx_path = layout.context_path(session_dir)
            if ctx_path.exists():
                ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
                self._agent.context.load_snapshot(ctx_data)
                loaded.append(layout.COMPONENT_CONTEXT)

            queues_path = layout.queues_path(session_dir)
            if queues_path.exists() and self._scheduler is not None:
                queues_data = json.loads(queues_path.read_text(encoding="utf-8"))
                qm = self._scheduler_queue_manager()
                if qm is not None and "scheduler" in queues_data:
                    qm.load_snapshot(queues_data["scheduler"])
                    qm.rebind_event_bus(self._event_bus)
                if "pending_steer_inputs" in queues_data:
                    self._agent.load_steer(queues_data["pending_steer_inputs"])
                loaded.append(layout.COMPONENT_QUEUES)

            runtime_path = layout.runtime_path(session_dir)
            if runtime_path.exists():
                runtime_data = json.loads(runtime_path.read_text(encoding="utf-8"))
                self._agent.load_runtime(runtime_data)
                loaded.append(layout.COMPONENT_RUNTIME)

            plugins_dir = layout.plugins_dir(session_dir)
            if plugins_dir.exists():
                self._load_plugins(plugins_dir, manifest)
                loaded.append(layout.COMPONENT_PLUGINS)

            if self._event_bus is not None:
                self._event_bus.publish(
                    SessionLoadedEvent.create(
                        session_id=session_id, components_loaded=loaded
                    )
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

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a session directory.

        If the deleted session is currently active, the manager has no current
        session afterwards.
        """
        layout.remove_session_dir(layout.session_dir(self._root, session_id))
        with self._lock:
            if self._session_id == session_id:
                self._session_id = None
                self._session_name = None

    def save_now(self, *, fsync: bool = True) -> None:
        """Capture all components synchronously, enqueue + wait for the writer."""
        if self._session_id is None:
            return
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
        if not components:
            return
        try:
            snapshots, manifest_patch = self._capture(event.type, components)
        except Exception:
            logger.exception(
                "session capture failed during event %s; skipping checkpoint",
                event.type,
            )
            return
        job = WriteJob(
            session_dir=layout.session_dir(self._root, self._session_id),
            snapshots=snapshots,
            manifest_patch=manifest_patch,
            fsync=False,
            component_set_key=",".join(sorted(components)),
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
            qm = self._scheduler_queue_manager()
            if qm is not None:
                queues_payload["scheduler"] = qm.snapshot()
            queues_payload["pending_steer_inputs"] = self._agent.snapshot_steer()
            queues_payload["pending_audit_tool_calls"] = [
                {
                    "tool_call_id": p.tool_call_id,
                    "tool_name": p.tool_name,
                    "arguments": p.arguments,
                    "requested_at": p.requested_at,
                }
                for p in self._agent.context.get_pending_tool_calls()
            ]
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
            "last_checkpoint_event": trigger,
            "active_plugins": self._active_plugin_names(),
        }
        return snapshots, manifest_patch

    def _final_flush(self) -> None:
        """Exit hook: synchronously capture once more then drain the writer.

        No-op when the manager has already been detached or has no current
        session — the relevant state is already on disk.
        """
        try:
            if self._session_id is not None and self._agent is not None:
                snapshots, manifest_patch = self._capture_all("exit")
                job = WriteJob(
                    session_dir=layout.session_dir(self._root, self._session_id),
                    snapshots=snapshots,
                    manifest_patch=manifest_patch,
                    fsync=True,
                    component_set_key="final_flush",
                )
                self._writer.submit(job)
        except Exception:
            logger.exception("session final flush capture failed")
        finally:
            try:
                if self._writer_owned:
                    self._writer.shutdown(timeout=5.0)
            except Exception:
                logger.exception("session writer shutdown raised")

    # --- helpers ---------------------------------------------------------

    def _scheduler_queue_manager(self) -> Any:
        if self._scheduler is None:
            return None
        for attr in ("_queue_manager", "queue_manager", "queues"):
            qm = getattr(self._scheduler, attr, None)
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
