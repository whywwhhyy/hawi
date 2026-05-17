from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

from hawi.events import EventBus, PluginEvent, PluginEventType

from .types import PluginHooks

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool
    from .resource import HawiResource


@dataclass
class PluginRuntimeContext:
    """Execution context automatically attached to plugin-emitted events."""

    run_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    iteration: int | None = None

class HawiPlugin:
    """Base class for Hawi plugins.

    Plugins can provide:
    - hooks: Lifecycle hooks
    - tools: Custom tools for agents to use
    - resources: Contextual data/resources for agents (MCP-compatible)

    For fork/clone support, plugins can be used in two ways:
    1. **Clone mode**: Implement `clone()` to return a copy of the plugin.
       Default implementation returns self (safe for stateless plugins).
    2. **Factory mode**: Pass `plugin_factories` to HawiAgent instead of `plugins`.
       Factories are called during init and clone to create fresh instances.
    """
    name: ClassVar[str | None] = None
    display_name: ClassVar[str | None] = None
    dependencies: ClassVar[Sequence[str]] = ()
    _cached_hooks:PluginHooks
    _cached_tools:Sequence[AgentTool]
    _event_bus: EventBus | None = None
    _plugin_id: str | None = None
    _plugin_name: str | None = None
    _runtime_context: PluginRuntimeContext | None = None

    def _collect_items(self):
        from hawi.tool import tool as create_tool

        self._cached_hooks = {}
        self._cached_tools = []

        # Skip these properties to avoid triggering recursion
        _skip_names = {"hooks", "tools", "resources", "_cached_hooks", "_cached_tools"}

        for name in dir(self):
            if name in _skip_names:
                continue
            member = getattr(self, name, None)
            if getattr(member, "_is_hook", None) is True:
                hook_type = getattr(member, "_hook_type")
                self._cached_hooks[hook_type] = member
            if getattr(member, "_is_agent_tool", None) is True and callable(member):
                agent_tools_kwargs = getattr(member, "_agent_tool_parameters", {})
                agent_tool = create_tool(**agent_tools_kwargs)(member)
                setattr(agent_tool, "_hawi_plugin", self)
                self._cached_tools.append(agent_tool)

    @property
    def hooks(self) -> PluginHooks:
        """Lifecycle hooks."""
        if not hasattr(self, "_cached_hooks"):
            self._collect_items()
        return self._cached_hooks

    @property
    def tools(self) -> Sequence[AgentTool]:
        """Tools provided by this plugin."""
        if not hasattr(self, "_cached_tools"):
            self._collect_items()
        return self._cached_tools

    @property
    def resources(self) -> Sequence[HawiResource]:
        """Resources provided by this plugin (MCP-compatible).

        Resources provide contextual data to agents, identified by URI.
        They can be text or binary, static or dynamic.
        """
        return []

    def clone(self) -> HawiPlugin:
        """Create a copy of this plugin for agent fork/clone operations.

        Default implementation returns self, which is safe for stateless plugins.
        Stateful plugins (e.g., with subprocess, file handles) should override
        this to return a fresh instance or a proper deep copy.

        Returns:
            A plugin instance for the cloned agent. Default returns self.
        """
        return self

    def save_state(self) -> dict[str, Any] | None:
        """Return JSON-serializable plugin state for session persistence.

        Default returns ``None`` — the SessionManager will skip writing a state
        file for this plugin. Plugins that hold non-reproducible state (open
        documents, paused workflows, plan items, etc.) should override and
        return a plain ``dict`` whose values survive ``json.dumps``.

        Implementations must NOT include live references (subprocess handles,
        sockets, ``EventBus`` instances). Capture only the data needed to
        rebuild equivalent state in :py:meth:`load_state`.
        """
        return None

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore plugin state previously produced by :py:meth:`save_state`.

        Default is a no-op. Plugins that override ``save_state`` should
        override this counterpart. The agent guarantees that ``load_state``
        is called before the plugin is used by any active run.
        """
        return None

    @property
    def plugin_id(self) -> str:
        """Stable plugin identifier used in plugin.* events."""
        return self._plugin_id or self.name or self.__class__.__name__

    @property
    def plugin_name(self) -> str:
        """Human-readable plugin name used in plugin.* events."""
        return self._plugin_name or self.display_name or self.__class__.__name__

    def bind_event_bus(self, event_bus: EventBus | None) -> None:
        """Bind the event bus used by ``emit_*`` helper methods."""
        self._event_bus = event_bus

    def bind_plugin_identity(
        self,
        *,
        plugin_id: str | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Bind GUI-facing plugin identity.

        Core integrations can use this to expose a stable key such as
        ``hawi/filesystem`` while keeping the class name as the display label.
        """
        self._plugin_id = plugin_id or self._plugin_id
        self._plugin_name = plugin_name or self._plugin_name

    @contextlib.contextmanager
    def plugin_event_context(
        self,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        iteration: int | None = None,
    ):
        previous = self._runtime_context
        self._runtime_context = PluginRuntimeContext(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            iteration=iteration,
        )
        try:
            yield
        finally:
            self._runtime_context = previous

    def emit_plugin_event(
        self,
        event_type: PluginEventType,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> PluginEvent | None:
        """Emit one ``plugin.*`` event synchronously.

        Returns the event object when an event bus is bound; otherwise returns
        ``None`` so standalone plugin usage stays frictionless.
        """
        event = self._make_plugin_event(
            event_type,
            payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
            message_id=message_id,
        )
        if self._event_bus is None:
            return None
        self._event_bus.publish(event)
        return event

    async def aemit_plugin_event(
        self,
        event_type: PluginEventType,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> PluginEvent | None:
        """Emit one ``plugin.*`` event and wait for async subscribers."""
        event = self._make_plugin_event(
            event_type,
            payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
            message_id=message_id,
        )
        if self._event_bus is None:
            return None
        await self._event_bus.publish_async(event)
        return event

    def emit_message(
        self,
        message: str,
        *,
        level: Literal["debug", "info", "warning", "error"] = "info",
        title: str | None = None,
        data: Any | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        payload: dict[str, Any] = {"level": level, "message": message}
        if title:
            payload["title"] = title
        if data is not None:
            payload["data"] = data
        return self.emit_plugin_event(
            "plugin.message",
            payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
            message_id=f"plugin-msg-{uuid.uuid4().hex[:10]}",
        )

    def emit_status(
        self,
        status: str,
        *,
        label: str | None = None,
        message: str | None = None,
        progress: float | None = None,
        data: Any | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        payload: dict[str, Any] = {"status": status}
        if label:
            payload["label"] = label
        if message:
            payload["message"] = message
        if progress is not None:
            payload["progress"] = progress
        if data is not None:
            payload["data"] = data
        return self.emit_plugin_event(
            "plugin.status",
            payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def emit_tool_progress(
        self,
        *,
        progress: float | None = None,
        status: str | None = None,
        message: str | None = None,
        label: str | None = None,
        data: Any | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        payload: dict[str, Any] = {}
        if progress is not None:
            payload["progress"] = progress
        if status:
            payload["status"] = status
        if message:
            payload["message"] = message
        if label:
            payload["label"] = label
        if data is not None:
            payload["data"] = data
        return self.emit_plugin_event(
            "plugin.tool_progress",
            payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def upsert_artifact(
        self,
        artifact_id: str,
        *,
        artifact_type: str,
        title: str,
        content: str | None = None,
        data: Any | None = None,
        mime_type: str | None = None,
        language: str | None = None,
        uri: str | None = None,
        path: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        artifact: dict[str, Any] = {
            "id": artifact_id,
            "type": artifact_type,
            "title": title,
        }
        optional = {
            "content": content,
            "data": data,
            "mime_type": mime_type,
            "language": language,
            "uri": uri,
            "path": path,
            "description": description,
            "status": status,
            "metadata": metadata,
        }
        artifact.update({key: value for key, value in optional.items() if value is not None})
        return self.emit_plugin_event(
            "plugin.artifact.upsert",
            {"artifact": artifact},
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def append_artifact(
        self,
        artifact_id: str,
        delta: str,
        *,
        field: str = "content",
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        return self.emit_plugin_event(
            "plugin.artifact.delta",
            {
                "artifact_id": artifact_id,
                "field": field,
                "delta": delta,
            },
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def remove_artifact(
        self,
        artifact_id: str,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        return self.emit_plugin_event(
            "plugin.artifact.remove",
            {"artifact_id": artifact_id},
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def clear_artifacts(
        self,
        *,
        scope: Literal["plugin", "all"] = "plugin",
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> PluginEvent | None:
        return self.emit_plugin_event(
            "plugin.artifact.clear",
            {"scope": scope},
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def _make_plugin_event(
        self,
        event_type: PluginEventType,
        payload: dict[str, Any] | None,
        *,
        run_id: str | None,
        tool_call_id: str | None,
        message_id: str | None,
    ) -> PluginEvent:
        context = self._runtime_context
        return PluginEvent.create(
            event_type,
            plugin_id=self.plugin_id,
            plugin_name=self.plugin_name,
            payload=payload,
            run_id=run_id if run_id is not None else context.run_id if context else None,
            tool_call_id=(
                tool_call_id
                if tool_call_id is not None
                else context.tool_call_id
                if context
                else None
            ),
            message_id=message_id,
        )

    @classmethod
    def gui_config_schema(cls) -> dict[str, Any]:
        """JSON schema used by GUI to render plugin configuration fields.

        Default: no configuration options.
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict[str, Any]:
        """Default GUI configuration values for this plugin."""
        return {}
