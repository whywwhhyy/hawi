"""Runtime for the Hawi core process."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, cast

from hawi.agent import AutoCompactConfig, HawiAgent, HawiScheduler
from hawi.agent.context import AgentContext, ToolCallContext
from hawi.events import Event
from hawi.models import model_registry
from hawi.session import SessionManager
from hawi.tool import ToolParameterInjection

from .blob import BlobStore
from .blob.commands import dispatch_blob_command
from .event_mapper import SemanticEventMapper
from .protocol import (
    CoreCommand,
    ProtocolError,
    make_ack,
    make_error,
    make_frame,
    parse_frame,
    to_json_safe,
)

logger = logging.getLogger(__name__)

QueueKind = Literal["normal", "high_prio", "urgent"]

DEFAULT_SYSTEM_PROMPT = "你是Hawi，一个通用agent"

PLUGIN_FILESYSTEM = "filesystem"
PLUGIN_SHELL = "shell"
PLUGIN_WEB = "web"
PLUGIN_SKILLS = "skills"
PLUGIN_PYTHON_INTERPRETER = "python_interpreter"
PLUGIN_MCP = "mcp"
PLUGIN_PLAN = "plan"
PLUGIN_WORKFLOW = "workflow"
PLUGIN_ENVIRON_PROMPT = "environ_prompt"

KNOWN_PLUGINS = {
    PLUGIN_FILESYSTEM,
    PLUGIN_SHELL,
    PLUGIN_WEB,
    PLUGIN_SKILLS,
    PLUGIN_PYTHON_INTERPRETER,
    PLUGIN_MCP,
    PLUGIN_PLAN,
    PLUGIN_WORKFLOW,
    PLUGIN_ENVIRON_PROMPT,
}

PLUGIN_LABELS = {
    PLUGIN_FILESYSTEM: "FileSystemPlugin",
    PLUGIN_SHELL: "ShellPlugin",
    PLUGIN_WEB: "WebPlugin",
    PLUGIN_SKILLS: "SkillsPlugin",
    PLUGIN_PYTHON_INTERPRETER: "PythonInterpreterPlugin",
    PLUGIN_MCP: "MCPPlugin",
    PLUGIN_PLAN: "PlanPlugin",
    PLUGIN_WORKFLOW: "WorkflowPlugin",
    PLUGIN_ENVIRON_PROMPT: "EnvironPromptPlugin",
}

_EXTRA_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SERVER_CAPS: frozenset[str] = frozenset({"last_event_id"})
"""Capabilities the server advertises during hello negotiation. Plans 3-5 grow this set."""


@dataclass(frozen=True)
class ExtraToolParameter:
    """CLI-provided framework parameter to inject into every tool schema."""

    name: str
    type_name: str
    description: str
    schema: dict[str, Any]


def parse_extra_tool_parameter(raw: Sequence[str]) -> ExtraToolParameter:
    """Parse one ``--extra-tool-parameter name type description`` value."""
    if isinstance(raw, str) or len(raw) != 3:
        raise ValueError(
            "--extra-tool-parameter must use <name> <type> <description> format"
        )
    name, type_name, description = (part.strip() for part in raw)
    if not name or not type_name or not description:
        raise ValueError(
            "--extra-tool-parameter must use <name> <type> <description> format"
        )
    if not _EXTRA_PARAMETER_NAME_RE.match(name):
        raise ValueError(
            "--extra-tool-parameter name must start with a letter or underscore "
            "and contain only letters, numbers, and underscores"
        )
    schema = _extra_tool_parameter_schema(type_name)
    return ExtraToolParameter(
        name=name,
        type_name=type_name.lower(),
        description=description,
        schema=schema,
    )


def parse_extra_tool_parameters(raw_values: Sequence[Sequence[str]]) -> list[ExtraToolParameter]:
    """Parse stacked ``--extra-tool-parameter`` values and reject duplicates."""
    parameters = [parse_extra_tool_parameter(value) for value in raw_values]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for parameter in parameters:
        if parameter.name in seen:
            duplicates.add(parameter.name)
        seen.add(parameter.name)
    if duplicates:
        raise ValueError(
            "Duplicate --extra-tool-parameter name(s): "
            + ", ".join(sorted(duplicates))
        )
    return parameters


def _extra_tool_parameter_schema(type_name: str) -> dict[str, Any]:
    normalized = type_name.lower()
    type_map: dict[str, dict[str, Any]] = {
        "str": {"type": "string"},
        "string": {"type": "string"},
        "int": {"type": "integer"},
        "integer": {"type": "integer"},
        "float": {"type": "number"},
        "number": {"type": "number"},
        "bool": {"type": "boolean"},
        "boolean": {"type": "boolean"},
        "object": {"type": "object"},
        "array": {"type": "array"},
    }
    if normalized not in type_map:
        raise ValueError(
            "Unsupported --extra-tool-parameter type "
            f"'{type_name}'. Supported types: str, int, float, bool, object, array"
        )
    return dict(type_map[normalized])


class RuntimeClient(Protocol):
    """Minimal client interface used by CoreRuntime."""

    id: str
    authenticated: bool
    negotiated_caps: set[str]

    async def send(self, frame: dict[str, Any]) -> None:
        """Queue a frame for this client."""

    async def close(self) -> None:
        """Close the client transport."""


class CoreRuntime:
    """Owns the Hawi agent scheduler and command/event protocol handling."""

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        selected_plugins: list[str] | None = None,
        plugin_configs: dict[str, dict[str, Any]] | None = None,
        extra_tool_parameters: list[ExtraToolParameter] | None = None,
        max_context_tokens: int | None = None,
        token: str | None = None,
        status_interval: float = 0.3,
        broadcast_queue_size: int = 1000,
        blob_store: BlobStore | None = None,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self._selected_plugins = list(selected_plugins or [])
        self._plugin_configs = {
            name: dict(cfg) for name, cfg in (plugin_configs or {}).items()
        }
        self._extra_tool_parameters = list(extra_tool_parameters or [])
        self._max_context_tokens = max_context_tokens
        self._token = token
        self._blob_store: BlobStore | None = blob_store
        self._status_interval = status_interval

        self._scheduler: HawiScheduler | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._broadcast_task: asyncio.Task | None = None
        self._plugins: list[Any] = []
        self._session_manager: SessionManager | None = None

        self._clients: set[RuntimeClient] = set()
        self._mapper = SemanticEventMapper()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=broadcast_queue_size
        )
        self._shutdown_requested = asyncio.Event()
        self._started = False

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    async def start(self) -> None:
        """Build the agent scheduler and start background runtime tasks."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        if self._blob_store is not None:
            await self._blob_store.start()
        scheduler, scheduler_task, plugins = await self._build_scheduler(
            model_name=self.model_name,
            selected_plugins=self._selected_plugins,
            plugin_configs=self._plugin_configs,
            context_to_restore=None,
        )
        self._scheduler = scheduler
        self._scheduler_task = scheduler_task
        self._plugins = plugins
        self._session_manager = SessionManager()
        self._session_manager.attach(
            scheduler.agent,
            scheduler,
            event_bus=getattr(scheduler.agent, "_event_bus", None),
        )
        # Auto-create an initial in-memory session id. SessionManager writes it
        # to disk lazily once the conversation has a user-visible message, so
        # startup and empty "New" clicks do not leave blank sessions behind.
        try:
            self._session_manager.new_session()
        except Exception:
            logger.exception("failed to auto-create initial session")
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._status_task = asyncio.create_task(self._status_loop())
        self._started = True

    async def stop(self) -> None:
        """Stop scheduler, clients, plugins, and runtime tasks."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()

        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
            await asyncio.gather(self._status_task, return_exceptions=True)

        if self._session_manager is not None:
            try:
                self._session_manager.save_now()
            except Exception:
                logger.exception("session save_now failed during shutdown")
            try:
                self._session_manager.detach()
            except Exception:
                logger.exception("session detach failed during shutdown")
            self._session_manager = None

        await self._stop_scheduler(self._scheduler, self._scheduler_task, self._plugins)
        self._scheduler = None
        self._scheduler_task = None
        self._plugins = []

        for client in list(self._clients):
            await client.close()

        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            await asyncio.gather(self._broadcast_task, return_exceptions=True)

        if self._blob_store is not None:
            try:
                await self._blob_store.close()
            except Exception:
                logger.exception("blob store close failed during shutdown")

    async def wait_shutdown(self) -> None:
        await self._shutdown_requested.wait()

    async def register_client(self, client: RuntimeClient) -> None:
        self._clients.add(client)
        if self._token is None:
            client.authenticated = True
            await client.send(make_frame("core.ready", self._ready_payload()))

    async def unregister_client(self, client: RuntimeClient) -> None:
        self._clients.discard(client)

    async def handle_frame(self, client: RuntimeClient, raw: str | bytes) -> None:
        """Parse and process one incoming client frame."""
        try:
            command = parse_frame(raw)
        except ProtocolError as exc:
            await client.send(make_error(str(exc), code=exc.code))
            return
        await self.handle_command(client, command)

    async def handle_command(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        """Process one validated client command."""
        logger.debug(
            "Handling command type=%s id=%s client=%s payload=%r",
            command.type,
            command.id,
            client.id,
            to_json_safe(command.payload),
        )
        try:
            if command.type == "hello":
                await self._handle_hello(client, command)
                return

            if not client.authenticated:
                await client.send(
                    make_error(
                        "Client must send a successful hello command first.",
                        request_id=command.id,
                        code="unauthenticated",
                    )
                )
                return

            if command.type == "ping":
                await client.send(
                    make_frame("pong", {"ok": True}, request_id=command.id)
                )
                return

            if command.type == "enqueue":
                await self._handle_enqueue(client, command)
            elif command.type == "interrupt":
                await self._handle_interrupt(client, command)
            elif command.type == "clear_context":
                await self._handle_clear_context(client, command)
            elif command.type == "clear_queue":
                await self._handle_clear_queue(client, command)
            elif command.type == "set_system_prompt":
                await self._handle_set_system_prompt(client, command)
            elif command.type == "switch_model":
                await self._handle_switch_model(client, command)
            elif command.type == "apply_plugins":
                await self._handle_apply_plugins(client, command)
            elif command.type == "get_status":
                await client.send(
                    make_frame(
                        "core.status",
                        self._status_payload(),
                        request_id=command.id,
                    )
                )
            elif command.type == "session_list":
                await self._handle_session_list(client, command)
            elif command.type == "session_new":
                await self._handle_session_new(client, command)
            elif command.type == "session_load":
                await self._handle_session_load(client, command)
            elif command.type == "session_switch":
                await self._handle_session_switch(client, command)
            elif command.type == "session_delete":
                await self._handle_session_delete(client, command)
            elif command.type == "session_save_now":
                await self._handle_session_save_now(client, command)
            elif command.type == "session_history":
                await self._handle_session_history(client, command)
            elif command.type == "shutdown":
                await client.send(make_ack("shutdown", request_id=command.id))
                await self.stop()
            elif command.type.startswith("blob."):
                if self._blob_store is None:
                    await client.send(
                        make_error(
                            "Blob store is disabled on this engine.",
                            request_id=command.id,
                            code="blob_disabled",
                        )
                    )
                    return
                await dispatch_blob_command(client, command, store=self._blob_store)
            else:
                await client.send(
                    make_error(
                        f"Unsupported command: {command.type}",
                        request_id=command.id,
                        code="unknown_command",
                    )
                )
        except Exception as exc:
            logger.exception("Command failed: %s", command.type)
            await client.send(
                make_error(
                    str(exc),
                    request_id=command.id,
                    code="command_failed",
                    details={"command": command.type, "class": exc.__class__.__name__},
                )
            )

    def emit(self, frame: dict[str, Any]) -> None:
        """Schedule a semantic event for broadcast from any thread."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._enqueue_broadcast, frame)

    def _enqueue_broadcast(self, frame: dict[str, Any]) -> None:
        if self._shutdown_requested.is_set():
            return
        if frame.get("type") != "core.status":
            logger.debug(
                "Queueing event type=%s id=%s payload=%r",
                frame.get("type"),
                frame.get("id"),
                to_json_safe(frame.get("payload", {})),
            )
        try:
            self._broadcast_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Dropping core event because broadcast queue is full")

    async def _broadcast_loop(self) -> None:
        while not self._shutdown_requested.is_set():
            frame = await self._broadcast_queue.get()
            await self._broadcast(frame)

    async def _broadcast(self, frame: dict[str, Any]) -> None:
        clients = list(self._clients)
        for client in clients:
            if client.authenticated:
                await client.send(frame)

    async def _status_loop(self) -> None:
        while not self._shutdown_requested.is_set():
            await asyncio.sleep(self._status_interval)
            self.emit(make_frame("core.status", self._status_payload()))

    async def _handle_hello(self, client: RuntimeClient, command: CoreCommand) -> None:
        if self._token is not None:
            token = command.payload.get("token")
            if token != self._token:
                await client.send(
                    make_error(
                        "Invalid authentication token.",
                        request_id=command.id,
                        code="unauthorized",
                    )
                )
                return

        client_caps_raw = command.payload.get("client_caps", [])
        if not isinstance(client_caps_raw, list) or not all(
            isinstance(c, str) for c in client_caps_raw
        ):
            await client.send(
                make_error(
                    "'hello.payload.client_caps' must be a list of strings.",
                    request_id=command.id,
                    code="bad_request",
                )
            )
            return
        client_caps = set(client_caps_raw)
        active_caps = set(SERVER_CAPS)
        if self._blob_store is not None:
            active_caps.add("blob_v1")
        negotiated = client_caps & active_caps
        client.negotiated_caps = negotiated

        was_authenticated = client.authenticated
        client.authenticated = True
        await client.send(
            make_ack(
                "hello",
                request_id=command.id,
                payload={
                    "authenticated": True,
                    "server_caps": sorted(active_caps),
                    "negotiated": sorted(negotiated),
                },
            )
        )
        if not was_authenticated:
            await client.send(make_frame("core.ready", self._ready_payload()))

    async def _handle_enqueue(self, client: RuntimeClient, command: CoreCommand) -> None:
        scheduler = self._require_scheduler()
        content = command.payload.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError("'enqueue.payload.content' must be a string or content part list")
        queue = self._queue_kind(command.payload.get("queue", "normal"))
        metadata = command.payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("'enqueue.payload.metadata' must be an object")
        metadata = {**metadata, "queue_kind": queue}
        message_id = scheduler.enqueue(content, queue, metadata=metadata)
        await client.send(
            make_ack(
                "enqueue",
                request_id=command.id,
                payload={"message_id": message_id, "queue": queue},
            )
        )

    async def _handle_interrupt(self, client: RuntimeClient, command: CoreCommand) -> None:
        scheduler = self._require_scheduler()
        reason = command.payload.get("reason", "user")
        if not isinstance(reason, str):
            raise ValueError("'interrupt.payload.reason' must be a string")
        interrupted_ids = await scheduler.interrupt(reason)
        await client.send(
            make_ack(
                "interrupt",
                request_id=command.id,
                payload={"interrupted_tool_calls": interrupted_ids},
            )
        )

    async def _handle_clear_context(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        self._require_scheduler().agent.context.clear()
        await client.send(make_ack("clear_context", request_id=command.id))

    async def _handle_clear_queue(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        scheduler = self._require_scheduler()
        queue = command.payload.get("queue", "all")
        if queue == "all":
            cleared = scheduler.clear_all_queues()
        else:
            cleared = scheduler.clear_queue(self._queue_kind(queue))
        await client.send(
            make_ack(
                "clear_queue",
                request_id=command.id,
                payload={"cleared": cleared},
            )
        )

    async def _handle_set_system_prompt(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        system_prompt = command.payload.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise ValueError("'set_system_prompt.payload.system_prompt' must be a string")
        self.system_prompt = system_prompt
        self._require_scheduler().agent.context.set_system_prompt(system_prompt)
        await client.send(make_ack("set_system_prompt", request_id=command.id))

    async def _handle_switch_model(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        model_name = command.payload.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("'switch_model.payload.model_name' must be a non-empty string")
        self._require_scheduler().agent.set_model(model_name)
        self.model_name = model_name
        await client.send(
            make_ack(
                "switch_model",
                request_id=command.id,
                payload={"model_name": model_name},
            )
        )
        self.emit(make_frame("core.ready", self._ready_payload()))

    async def _handle_apply_plugins(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        scheduler = self._require_scheduler()
        if not scheduler._executor.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Apply plugins when the scheduler is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return

        selected_plugins = command.payload.get("selected_plugins", [])
        if selected_plugins is None:
            selected_plugins = []
        if not isinstance(selected_plugins, list) or not all(
            isinstance(name, str) for name in selected_plugins
        ):
            raise ValueError("'apply_plugins.payload.selected_plugins' must be a string list")
        unknown = sorted(set(selected_plugins) - KNOWN_PLUGINS)
        if unknown:
            raise ValueError(f"Unknown plugin key(s): {', '.join(unknown)}")

        plugin_configs = command.payload.get("plugin_configs", {})
        if plugin_configs is None:
            plugin_configs = {}
        if not isinstance(plugin_configs, dict):
            raise ValueError("'apply_plugins.payload.plugin_configs' must be an object")

        context_copy = scheduler.agent.context.copy()
        await self._replace_scheduler(
            model_name=self.model_name,
            selected_plugins=list(selected_plugins),
            plugin_configs={
                str(name): dict(cfg) if isinstance(cfg, dict) else {}
                for name, cfg in plugin_configs.items()
            },
            preserve_context=context_copy,
        )
        await client.send(
            make_ack(
                "apply_plugins",
                request_id=command.id,
                payload={
                    "selected_plugins": list(self._selected_plugins),
                    "plugin_configs": to_json_safe(self._plugin_configs),
                },
            )
        )

    def _require_session_manager(self) -> SessionManager:
        if self._session_manager is None:
            raise RuntimeError("SessionManager not initialized; runtime not started")
        return self._session_manager

    @staticmethod
    def _optional_session_id(command: CoreCommand) -> str | None:
        session_id = command.payload.get("session_id")
        if session_id is None:
            return None
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                "payload.session_id must be a non-empty string when present"
            )
        return session_id

    async def _handle_session_list(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        sessions = [
            {
                "session_id": m.session_id,
                "name": m.name,
                "created_at": m.created_at,
                "updated_at": m.updated_at,
                "last_checkpoint_event": m.last_checkpoint_event,
                "components_present": m.components_present,
            }
            for m in sm.list_sessions()
        ]
        await client.send(
            make_ack(
                "session_list",
                request_id=command.id,
                payload={
                    "sessions": sessions,
                    "current_session_id": sm.current_session_id,
                },
            )
        )

    async def _handle_session_new(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        scheduler = self._require_scheduler()
        if not scheduler._executor.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Create a new session when the scheduler is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return
        name = command.payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("'session_new.payload.name' must be a string when present")
        scheduler.agent.context.clear()
        scheduler.clear_all_queues()
        scheduler.agent.load_steer([])
        scheduler.agent.load_runtime(
            {
                "version": 1,
                "current_tool_calls": [],
                "interrupted_tool_call_ids": [],
                "last_unsent_tool_results": [],
                "last_interrupt_reason": None,
            }
        )
        session_id = sm.new_session(name=name)
        await client.send(
            make_ack(
                "session_new",
                request_id=command.id,
                payload={"session_id": session_id, "name": name or session_id},
            )
        )

    async def _handle_session_load(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        session_id = command.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("'session_load.payload.session_id' must be a non-empty string")
        sm.load_session(session_id)
        message_history = sm.read_message_history(session_id)
        context_usage = self._agent_context_usage()
        await client.send(
            make_ack(
                "session_load",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "message_history": message_history,
                    "context_usage": context_usage,
                },
            )
        )

    async def _handle_session_switch(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        session_id = command.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                "'session_switch.payload.session_id' must be a non-empty string"
            )
        sm.switch_to(session_id)
        message_history = sm.read_message_history(session_id)
        context_usage = self._agent_context_usage()
        await client.send(
            make_ack(
                "session_switch",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "message_history": message_history,
                    "context_usage": context_usage,
                },
            )
        )

    async def _handle_session_delete(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        session_id = command.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                "'session_delete.payload.session_id' must be a non-empty string"
            )
        if session_id == sm.current_session_id:
            await client.send(
                make_error(
                    "Cannot delete the current session.",
                    request_id=command.id,
                    code="invalid_session_delete",
                )
            )
            return
        sm.delete_session(session_id)
        await client.send(
            make_ack(
                "session_delete",
                request_id=command.id,
                payload={"session_id": session_id},
            )
        )

    async def _handle_session_save_now(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        sm.save_now()
        await client.send(
            make_ack(
                "session_save_now",
                request_id=command.id,
                payload={"session_id": sm.current_session_id},
            )
        )

    async def _handle_session_history(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        requested_session_id = self._optional_session_id(command)
        session_id = requested_session_id or sm.current_session_id
        message_history = sm.read_message_history(requested_session_id)
        context_usage = self._agent_context_usage() if requested_session_id is None else None
        await client.send(
            make_ack(
                "session_history",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "message_history": message_history,
                    "context_usage": context_usage,
                },
            )
        )

    async def _replace_scheduler(
        self,
        *,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
        preserve_context: AgentContext | None,
    ) -> None:
        new_scheduler, new_task, new_plugins = await self._build_scheduler(
            model_name=model_name,
            selected_plugins=selected_plugins,
            plugin_configs=plugin_configs,
            context_to_restore=preserve_context,
        )
        await self._stop_scheduler(self._scheduler, self._scheduler_task, self._plugins)
        self._scheduler = new_scheduler
        self._scheduler_task = new_task
        self._plugins = new_plugins
        self._selected_plugins = list(selected_plugins)
        self._plugin_configs = {name: dict(cfg) for name, cfg in plugin_configs.items()}
        self.emit(make_frame("core.ready", self._ready_payload()))

    async def _build_scheduler(
        self,
        *,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
        context_to_restore: AgentContext | None,
    ) -> tuple[HawiScheduler, asyncio.Task, list[Any]]:
        model_overrides: dict[str, Any] = {}
        if self._max_context_tokens is not None:
            model_overrides["max_context_tokens"] = self._max_context_tokens
        model = model_registry.create_model(model_name, **model_overrides)
        auto_compact = None
        if self._max_context_tokens is not None:
            auto_compact = AutoCompactConfig(
                enabled=True,
                max_context_tokens=self._max_context_tokens,
            )
        plugins = await self._create_plugins(selected_plugins, plugin_configs)
        agent = HawiAgent(
            model=model,
            plugins=plugins,
            system_prompt=self.system_prompt,
            max_iterations=None,
            streaming=True,
            auto_compact=auto_compact,
        )
        self._apply_extra_tool_parameters(agent)
        if context_to_restore is not None:
            agent.set_context(context_to_restore.copy())
            agent.context.tool_call_context = ToolCallContext(agent)

        scheduler = HawiScheduler(agent)
        agent.event_bus.subscribe(self._on_hawi_event)
        scheduler_task = asyncio.create_task(scheduler.run_forever(poll_interval=0.1))
        return scheduler, scheduler_task, plugins

    def _apply_extra_tool_parameters(self, agent: HawiAgent) -> None:
        for parameter in self._extra_tool_parameters:
            agent.plugins.add_tool_parameter_injection(
                ToolParameterInjection(
                    name=parameter.name,
                    schema=self._extra_tool_parameter_json_schema(parameter),
                    required=True,
                )
            )
        if self._extra_tool_parameters:
            defs = agent.plugins.get_tool_definitions()
            agent.context.tool_definitions = defs if defs else None

    @staticmethod
    def _extra_tool_parameter_json_schema(parameter: ExtraToolParameter) -> dict[str, Any]:
        return {
            **parameter.schema,
            "description": parameter.description,
        }

    async def _stop_scheduler(
        self,
        scheduler: HawiScheduler | None,
        scheduler_task: asyncio.Task | None,
        plugins: list[Any],
    ) -> None:
        if scheduler is not None:
            scheduler.agent.event_bus.unsubscribe(self._on_hawi_event)
            scheduler.stop()
        if scheduler_task and not scheduler_task.done():
            scheduler_task.cancel()
            await asyncio.gather(scheduler_task, return_exceptions=True)
        if scheduler is not None:
            try:
                scheduler.agent.event_bus.close(wait=True, timeout=2.0)
            except Exception:
                logger.exception("Failed to close scheduler event bus")
        await self._close_plugins(plugins)

    async def _create_plugins(
        self,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
    ) -> list[Any]:
        plugins: list[Any] = []
        for plugin_key in selected_plugins:
            cfg = dict(plugin_configs.get(plugin_key, {}))
            plugin: Any
            if plugin_key == PLUGIN_FILESYSTEM:
                from hawi_plugins.filesystem_plugin import FileSystemPlugin

                plugin = FileSystemPlugin()
            elif plugin_key == PLUGIN_SHELL:
                from hawi_plugins.shell_plugin import ShellPlugin

                plugin = ShellPlugin()
            elif plugin_key == PLUGIN_WEB:
                from hawi_plugins.web import WebPlugin

                plugin = WebPlugin()
            elif plugin_key == PLUGIN_SKILLS:
                from hawi_plugins.skills_plugin import SkillsPlugin

                skills_dir = str(cfg.get("skills_dir") or ".skills")
                plugin = SkillsPlugin(skills_dir=skills_dir)
            elif plugin_key == PLUGIN_PYTHON_INTERPRETER:
                from hawi_plugins.python_interpreter import PythonInterpreterPlugin

                work_dir_raw = cfg.get("work_dir")
                work_dir = str(work_dir_raw).strip() if isinstance(work_dir_raw, str) else None
                plugin = PythonInterpreterPlugin(
                    work_dir=work_dir or None,
                    print_execution=bool(cfg.get("print_execution", False)),
                )
            elif plugin_key == PLUGIN_MCP:
                from hawi_plugins.mcp_plugin import MCPPlugin

                config_path = str(cfg.get("config_path") or "").strip()
                if not config_path:
                    raise ValueError("MCP plugin requires 'config_path'.")
                plugin = MCPPlugin(config_path=config_path)
                await plugin.connect()
            elif plugin_key == PLUGIN_PLAN:
                from hawi_plugins.plan_plugin import PlanPlugin

                plugin = PlanPlugin(
                    fold_completed_tasks=bool(cfg.get("fold_completed_tasks", False))
                )
            elif plugin_key == PLUGIN_WORKFLOW:
                from hawi_plugins.workflow_plugin import WorkflowPlugin

                plugin = WorkflowPlugin()
            elif plugin_key == PLUGIN_ENVIRON_PROMPT:
                from hawi_plugins.environ_prompt_plugin import EnvironPromptPlugin

                config_path = str(cfg.get("config_path") or "").strip() or None
                plugin = EnvironPromptPlugin(config_path=config_path)
            else:
                raise ValueError(f"Unknown plugin key: {plugin_key}")
            if hasattr(plugin, "bind_plugin_identity"):
                plugin.bind_plugin_identity(
                    plugin_id=plugin_key,
                    plugin_name=PLUGIN_LABELS.get(plugin_key, plugin_key),
                )
            plugins.append(plugin)
        return plugins

    async def _close_plugins(self, plugins: list[Any]) -> None:
        for plugin in plugins:
            try:
                if hasattr(plugin, "disconnect"):
                    result = plugin.disconnect()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(plugin, "close"):
                    result = plugin.close()
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                logger.exception("Failed to close plugin %s", plugin.__class__.__name__)

    def _on_hawi_event(self, event: Event) -> None:
        for frame in self._mapper.map(event):
            self.emit(frame)

    def _require_scheduler(self) -> HawiScheduler:
        if self._scheduler is None:
            raise RuntimeError("Core runtime is not ready")
        return self._scheduler

    @staticmethod
    def _queue_kind(value: Any) -> QueueKind:
        if value not in {"normal", "high_prio", "urgent"}:
            raise ValueError("queue must be one of: normal, high_prio, urgent")
        return cast(QueueKind, value)

    def _ready_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "selected_plugins": list(self._selected_plugins),
            "plugin_configs": to_json_safe(self._plugin_configs),
            "status": self._status_payload(),
        }

    def _status_payload(self) -> dict[str, Any]:
        if self._scheduler is None:
            return {
                "ready": False,
                "scheduler_state": "STOPPED",
                "agent_state": "STOPPED",
                "queue_lengths": {"normal": 0, "high_prio": 0, "urgent": 0},
                "queue_messages": {"normal": [], "high_prio": [], "urgent": []},
                "model_name": self.model_name,
            }
        queue_messages_getter = getattr(self._scheduler, "get_queue_messages", None)
        queue_messages = (
            queue_messages_getter()
            if callable(queue_messages_getter)
            else {"normal": [], "high_prio": [], "urgent": []}
        )
        queue_messages = {**queue_messages, "urgent": []}
        pending_input_getter = getattr(
            self._scheduler.agent,
            "get_pending_input_messages",
            None,
        )
        if callable(pending_input_getter):
            pending_inputs = pending_input_getter()
            if pending_inputs:
                queue_messages = {**queue_messages}
                for pending in pending_inputs:
                    queue = pending.get("queue")
                    if queue not in {"normal", "high_prio", "urgent"}:
                        queue = "high_prio"
                    queue_messages[queue] = [
                        *queue_messages.get(queue, []),
                        pending,
                    ]
        payload = {
            "ready": True,
            "scheduler_state": self._scheduler.state.name,
            "agent_state": self._scheduler._executor.state.name,
            "queue_lengths": self._scheduler.get_queue_lengths(),
            "queue_messages": queue_messages,
            "model_name": self.model_name,
        }
        context_usage = self._agent_context_usage()
        if context_usage is not None:
            payload["context_usage"] = context_usage
        return payload

    def _agent_context_usage(self) -> dict[str, Any] | None:
        if self._scheduler is None:
            return None
        context = getattr(self._scheduler.agent, "context", None)
        saved_getter = getattr(context, "context_usage_snapshot", None)
        if callable(saved_getter):
            saved_snapshot = saved_getter()
            to_dict = getattr(saved_snapshot, "to_dict", None)
            if callable(to_dict):
                return to_json_safe(to_dict())
        getter = getattr(self._scheduler.agent, "context_usage", None)
        if not callable(getter):
            return None
        snapshot = getter()
        to_dict = getattr(snapshot, "to_dict", None)
        if callable(to_dict):
            return to_json_safe(to_dict())
        return None


def load_model_configs(extra_paths: list[str] | None = None) -> list[Path]:
    """Load model configs in core-cli order and return paths that existed."""
    loaded: list[Path] = []
    candidates = [
        Path.home() / ".hawi" / "models.yaml",
        Path.cwd() / ".hawi" / "models.yaml",
        Path.cwd() / "models.yaml",
    ]
    candidates.extend(Path(path) for path in (extra_paths or []))

    for path in candidates:
        if path.exists():
            model_registry.load_config(path, quiet=True)
            loaded.append(path)
    return loaded


def token_from_arg_or_env(token: str | None) -> str | None:
    """Resolve the core transport token from CLI or environment."""
    if token is not None:
        return token
    env_token = os.environ.get("HAWI_CORE_TOKEN")
    return env_token if env_token else None
