"""Runtime for the Hawi core process."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, cast

from hawi.agent import AutoCompactConfig, HawiAgent, AgentRunner
from hawi.agent.context import AgentContext, ToolCallContext
from hawi.agent.agent import SKIP_BEFORE_CONVERSATION_HOOKS_METADATA_KEY
from hawi.events import Event
from hawi.models import model_registry
from hawi.review import RuntimeReviewBroker, RuntimeReviewDecision
from hawi.session import SessionLockedError, SessionManager
from hawi.tool import ToolParameterInjection, ToolResult
from hawi.utils.debug import debug_assert
from hawi.utils.workspace import find_git_root

from .blob import BlobStore
from .blob.commands import dispatch_blob_command
from .blob.resolver import resolve_blob_references_for_model
from .event_mapper import SemanticEventMapper
from .plugin_registry import (
    KNOWN_PLUGINS,
    create_plugin,
    expand_plugin_dependencies,
    get_plugin_descriptor,
)
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

DEFAULT_RESUME_PROMPT = "继续"

_EXTRA_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SERVER_CAPS: frozenset[str] = frozenset({
    "last_event_id",
    "tlv_v1",
    "resume_v1",
    "runner_pause_v1",
    "queue_edit_v1",
    "message_intent_v1",
    "context_branch_v1",
    "manual_context_compact_v1",
    "auto_compact_config_v1",
    "plugin_action_v1",
    "runtime_review_v1",
})
"""Capabilities the server advertises during hello negotiation. Plans 3-5 grow this set."""

PLUGIN_ACTION_METHODS: frozenset[str] = frozenset({
    "approve_taskflow_review",
    "reject_taskflow_review",
    "approve_workflow_node",
    "reject_workflow_node",
    "approve_permission_review",
    "reject_permission_review",
})


@dataclass(frozen=True)
class ExtraToolParameter:
    """CLI-provided framework parameter to inject into every tool schema."""

    name: str
    type_name: str
    description: str
    schema: dict[str, Any]
    required: bool = True


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


def parse_extra_tool_parameter_json(raw: str) -> ExtraToolParameter:
    """Parse one JSON ``--extra-tool-parameter-json`` directive."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--extra-tool-parameter-json must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--extra-tool-parameter-json must be a JSON object")

    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("--extra-tool-parameter-json.name must be a non-empty string")
    name = name.strip()
    if not _EXTRA_PARAMETER_NAME_RE.match(name):
        raise ValueError(
            "--extra-tool-parameter-json.name must start with a letter or "
            "underscore and contain only letters, numbers, and underscores"
        )

    schema_raw = parsed.get("schema")
    type_raw = parsed.get("type", parsed.get("type_name"))
    if isinstance(schema_raw, dict):
        schema = dict(schema_raw)
        type_name = str(schema.get("type") or type_raw or "object").lower()
    elif isinstance(type_raw, str) and type_raw.strip():
        type_name = type_raw.strip().lower()
        schema = _extra_tool_parameter_schema(type_name)
    else:
        raise ValueError(
            "--extra-tool-parameter-json must include either schema or type"
        )

    description_raw = parsed.get("description", schema.get("description"))
    if not isinstance(description_raw, str) or not description_raw.strip():
        raise ValueError(
            "--extra-tool-parameter-json.description must be a non-empty string"
        )
    description = description_raw.strip()
    schema.setdefault("description", description)

    required = parsed.get("required", True)
    if not isinstance(required, bool):
        raise ValueError("--extra-tool-parameter-json.required must be a boolean")

    return ExtraToolParameter(
        name=name,
        type_name=type_name,
        description=description,
        schema=schema,
        required=required,
    )


def parse_extra_tool_parameters(
    raw_values: Sequence[Sequence[str]],
    raw_json_values: Sequence[str] = (),
) -> list[ExtraToolParameter]:
    """Parse stacked extra tool parameter directives and reject duplicates."""
    parameters = [
        *[parse_extra_tool_parameter(value) for value in raw_values],
        *[parse_extra_tool_parameter_json(value) for value in raw_json_values],
    ]
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
    """Owns the Hawi agent runner and command/event protocol handling."""

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        selected_plugins: list[str] | None = None,
        plugin_configs: dict[str, dict[str, Any]] | None = None,
        extra_tool_parameters: list[ExtraToolParameter] | None = None,
        max_context_tokens: int | None = None,
        keep_session_system_prompt: bool = True,
        gui_launch_profile: dict[str, Any] | None = None,
        initial_session_id: str | None = None,
        initial_session_name: str | None = None,
        token: str | None = None,
        status_interval: float = 0.3,
        broadcast_queue_size: int = 1000,
        blob_store: BlobStore | None = None,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self._selected_plugins = expand_plugin_dependencies(selected_plugins or [])
        self._plugin_configs = {
            name: dict(cfg) for name, cfg in (plugin_configs or {}).items()
            if name in KNOWN_PLUGINS
        }
        self._extra_tool_parameters = list(extra_tool_parameters or [])
        self._max_context_tokens = max_context_tokens
        self._keep_session_system_prompt = keep_session_system_prompt
        self._gui_launch_profile = (
            dict(gui_launch_profile) if isinstance(gui_launch_profile, dict) else None
        )
        self._initial_session_id = initial_session_id
        self._initial_session_name = initial_session_name
        self._token = token
        self._blob_store: BlobStore | None = blob_store
        self._status_interval = status_interval

        self._runner: AgentRunner | None = None
        self._runner_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._broadcast_task: asyncio.Task | None = None
        self._plugins: list[Any] = []
        self._session_manager: SessionManager | None = None
        self._review_broker = RuntimeReviewBroker()

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
        """Build the agent runner and start background runtime tasks."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        if self._blob_store is not None:
            await self._blob_store.start()
        runner, runner_task, plugins = await self._build_runner(
            model_name=self.model_name,
            selected_plugins=self._selected_plugins,
            plugin_configs=self._plugin_configs,
            context_to_restore=None,
        )
        self._runner = runner
        self._runner_task = runner_task
        self._plugins = plugins
        self._session_manager = SessionManager(
            keep_session_system_prompt=self._keep_session_system_prompt,
            manifest_metadata_provider=self._session_manifest_metadata,
        )
        self._session_manager.attach(
            runner.agent,
            runner,
            event_bus=getattr(runner.agent, "_event_bus", None),
        )
        # Auto-create an initial in-memory session id. SessionManager writes it
        # to disk lazily once the conversation has a user-visible message, so
        # startup and empty "New" clicks do not leave blank sessions behind.
        try:
            self._session_manager.new_session(
                name=self._initial_session_name,
                session_id=self._initial_session_id,
            )
        except Exception:
            logger.exception("failed to auto-create initial session")
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._status_task = asyncio.create_task(self._status_loop())
        self._started = True

    async def stop(self) -> None:
        """Stop runner, clients, plugins, and runtime tasks."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()

        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
            await asyncio.gather(self._status_task, return_exceptions=True)

        await self._stop_runner(self._runner, self._runner_task, self._plugins)
        self._runner = None
        self._runner_task = None
        self._plugins = []

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
            elif command.type == "stop":
                await self._handle_stop(client, command)
            elif command.type == "resume":
                await self._handle_resume(client, command)
            elif command.type == "queue_task_add":
                await self._handle_queue_task_add(client, command)
            elif command.type == "queue_task_update":
                await self._handle_queue_task_update(client, command)
            elif command.type == "queue_task_remove":
                await self._handle_queue_task_remove(client, command)
            elif command.type == "queue_task_reorder":
                await self._handle_queue_task_reorder(client, command)
            elif command.type == "queue_message_remove":
                await self._handle_queue_message_remove(client, command)
            elif command.type == "queue_message_promote":
                await self._handle_queue_message_promote(client, command)
            elif command.type == "clear_context":
                await self._handle_clear_context(client, command)
            elif command.type == "compact_context":
                await self._handle_compact_context(client, command)
            elif command.type == "set_auto_compact":
                await self._handle_set_auto_compact(client, command)
            elif command.type == "clear_queue":
                await self._handle_clear_queue(client, command)
            elif command.type == "set_system_prompt":
                await self._handle_set_system_prompt(client, command)
            elif command.type == "switch_model":
                await self._handle_switch_model(client, command)
            elif command.type == "refresh_models":
                await self._handle_refresh_models(client, command)
            elif command.type == "apply_plugins":
                await self._handle_apply_plugins(client, command)
            elif command.type == "plugin_action":
                await self._handle_plugin_action(client, command)
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
            elif command.type == "session_fork":
                await self._handle_session_fork(client, command)
            elif command.type == "session_rewind":
                await self._handle_session_rewind(client, command)
            elif command.type == "session_load":
                await self._handle_session_load(client, command)
            elif command.type == "session_switch":
                await self._handle_session_switch(client, command)
            elif command.type == "session_delete":
                await self._handle_session_delete(client, command)
            elif command.type == "session_rename":
                await self._handle_session_rename(client, command)
            elif command.type == "session_save_now":
                await self._handle_session_save_now(client, command)
            elif command.type == "session_history":
                await self._handle_session_history(client, command)
            elif command.type == "session_export_markdown":
                await self._handle_session_export_markdown(client, command)
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
        except SessionLockedError as exc:
            await client.send(
                make_error(
                    str(exc),
                    request_id=command.id,
                    code="session_locked",
                    details=exc.to_dict(),
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
        runner = self._require_runner()
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
        message_id = runner.enqueue(content, queue, metadata=metadata)
        await client.send(
            make_ack(
                "enqueue",
                request_id=command.id,
                payload={"message_id": message_id, "queue": queue},
            )
        )

    async def _handle_interrupt(self, client: RuntimeClient, command: CoreCommand) -> None:
        runner = self._require_runner()
        reason = command.payload.get("reason", "user")
        if not isinstance(reason, str):
            raise ValueError("'interrupt.payload.reason' must be a string")
        pause = command.payload.get("pause", False)
        if not isinstance(pause, bool):
            raise ValueError("'interrupt.payload.pause' must be a boolean when present")
        interrupted_ids = await runner.interrupt(reason, pause=pause)
        await client.send(
            make_ack(
                "interrupt",
                request_id=command.id,
                payload={
                    "interrupted_tool_calls": interrupted_ids,
                    "control": runner.control_snapshot(),
                },
            )
        )

    async def _handle_stop(self, client: RuntimeClient, command: CoreCommand) -> None:
        runner = self._require_runner()
        reason = str(command.payload.get("reason", "user"))
        message = command.payload.get("message")
        if message is not None and not isinstance(message, (str, list)):
            raise ValueError("'stop.payload.message' must be a string or content part list when present")
        metadata = command.payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("'stop.payload.metadata' must be an object when present")
        result = await runner.stop_execution(
            reason=reason,
            message=message,
            event_bus=None,
            metadata=dict(metadata),
        )
        await client.send(
            make_ack(
                "stop",
                request_id=command.id,
                payload={
                    "interrupted_tool_calls": result["interrupted_tool_calls"],
                    "message_id": result["message_id"],
                    "control": result["control"],
                },
            )
        )

    async def _handle_resume(self, client: RuntimeClient, command: CoreCommand) -> None:
        runner = self._require_runner()
        message = command.payload.get("message")
        if message is not None and not isinstance(message, (str, list)):
            raise ValueError("'resume.payload.message' must be a string or content part list when present")

        if message is None and self._runner_has_pending_immediate_work(runner):
            runner.resume()
            await client.send(
                make_ack(
                    "resume",
                    request_id=command.id,
                    payload={
                        "message_id": None,
                        "queue": None,
                        "resumed_existing_work": True,
                        "control": runner.control_snapshot(),
                    },
                )
            )
            return

        content = message if message else DEFAULT_RESUME_PROMPT
        msg_id = runner.submit_immediate_message(
            content,
            intent="resume",
            metadata={
                "intent": "resume",
                "display_message_type": "resume",
                "auto_generated": message is None,
                **(
                    {SKIP_BEFORE_CONVERSATION_HOOKS_METADATA_KEY: True}
                    if message is None
                    else {}
                ),
            },
        )
        await client.send(
            make_ack(
                "resume",
                request_id=command.id,
                payload={
                    "message_id": msg_id,
                    "queue": "high_prio",
                },
            )
        )

    @staticmethod
    def _runner_has_pending_immediate_work(runner: Any) -> bool:
        getter = getattr(runner, "has_pending_immediate_work", None)
        if callable(getter):
            return bool(getter())
        lengths_getter = getattr(runner, "get_queue_lengths", None)
        if not callable(lengths_getter):
            return False
        lengths = lengths_getter()
        if not isinstance(lengths, dict):
            return False
        return any(
            int(lengths.get(queue, 0) or 0) > 0
            for queue in ("urgent", "high_prio")
        )

    async def _handle_queue_task_add(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        content = command.payload.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError("'queue_task_add.payload.content' must be a string or content part list")
        msg = runner._queue_manager.enqueue_normal(
            content,
            metadata={"intent": "queue_task", "source": "gui_queue_panel"},
        )
        await client.send(
            make_ack(
                "queue_task_add",
                request_id=command.id,
                payload={"message_id": msg.id, "queue": "normal"},
            )
        )

    async def _handle_queue_task_update(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        message_id = command.payload.get("message_id")
        if not isinstance(message_id, str):
            raise ValueError("'queue_task_update.payload.message_id' must be a string")
        content = command.payload.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError("'queue_task_update.payload.content' must be a string or content part list when present")
        metadata = command.payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("'queue_task_update.payload.metadata' must be an object when present")
        ok = runner._queue_manager.update_message(
            message_id,
            content=content,
            metadata=metadata,
        )
        if not ok:
            raise ValueError(f"Message {message_id!r} not found or cannot be updated")
        await client.send(
            make_ack("queue_task_update", request_id=command.id, payload={"message_id": message_id})
        )

    async def _handle_queue_task_remove(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        message_id = command.payload.get("message_id")
        if not isinstance(message_id, str):
            raise ValueError("'queue_task_remove.payload.message_id' must be a string")
        ok = runner._queue_manager.remove_message(message_id)
        if not ok:
            raise ValueError(f"Message {message_id!r} not found")
        await client.send(
            make_ack("queue_task_remove", request_id=command.id, payload={"message_id": message_id})
        )

    async def _handle_queue_task_reorder(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        message_ids = command.payload.get("message_ids")
        if not isinstance(message_ids, list) or not all(isinstance(mid, str) for mid in message_ids):
            raise ValueError("'queue_task_reorder.payload.message_ids' must be a list of strings")
        from hawi.agent.runner.queue import QueueType
        new_order = runner._queue_manager.reorder_queue(QueueType.NORMAL, message_ids)
        await client.send(
            make_ack(
                "queue_task_reorder",
                request_id=command.id,
                payload={"message_ids": new_order},
            )
        )

    async def _handle_queue_message_remove(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        message_id = command.payload.get("message_id")
        if not isinstance(message_id, str):
            raise ValueError("'queue_message_remove.payload.message_id' must be a string")
        ok = runner._queue_manager.remove_message(message_id)
        if not ok:
            raise ValueError(f"Message {message_id!r} not found or already sent")
        await client.send(
            make_ack(
                "queue_message_remove",
                request_id=command.id,
                payload={"message_id": message_id},
            )
        )

    async def _handle_queue_message_promote(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        message_id = command.payload.get("message_id")
        if not isinstance(message_id, str):
            raise ValueError("'queue_message_promote.payload.message_id' must be a string")
        ok = runner._queue_manager.promote_normal_to_high_prio(message_id)
        if not ok:
            raise ValueError(
                f"Message {message_id!r} not found, already sent, or cannot be promoted"
            )
        await client.send(
            make_ack(
                "queue_message_promote",
                request_id=command.id,
                payload={"message_id": message_id, "queue": "high_prio"},
            )
        )

    async def _handle_clear_context(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        self._require_runner().agent.context.clear()
        await client.send(make_ack("clear_context", request_id=command.id))

    async def _handle_compact_context(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        runner = self._require_runner()
        if not runner.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Compact context when the runner is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return

        run_id = f"manual-compact-{command.id or uuid.uuid4().hex}"
        record = await runner.agent.acompact(run_id=run_id, mode="manual")
        sm.save_now()
        context_usage = self._agent_context_usage()
        message_history = sm.read_message_history(sm.current_session_id)
        record_payload = None
        if record is not None:
            to_dict = getattr(record, "to_dict", None)
            record_payload = to_json_safe(to_dict() if callable(to_dict) else record)
        await client.send(
            make_ack(
                "compact_context",
                request_id=command.id,
                payload={
                    "session_id": sm.current_session_id,
                    "status": "success" if record is not None else "skipped",
                    "record": record_payload,
                    "message_history": message_history,
                    "context_usage": context_usage,
                },
            )
        )

    async def _handle_set_auto_compact(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        cfg = getattr(runner.agent, "_auto_compact", None)
        if not isinstance(cfg, AutoCompactConfig):
            raise ValueError("Current agent does not support auto compact configuration")

        updates: dict[str, Any] = {}
        if "enabled" in command.payload:
            enabled = command.payload["enabled"]
            if not isinstance(enabled, bool):
                raise ValueError("'set_auto_compact.payload.enabled' must be a boolean")
            updates["enabled"] = enabled

        if "trigger_tokens" in command.payload:
            trigger_tokens = command.payload["trigger_tokens"]
            if trigger_tokens is None:
                updates["trigger_tokens"] = None
            else:
                if isinstance(trigger_tokens, bool) or not isinstance(trigger_tokens, int):
                    raise ValueError(
                        "'set_auto_compact.payload.trigger_tokens' must be a positive integer or null"
                    )
                if trigger_tokens <= 0:
                    raise ValueError(
                        "'set_auto_compact.payload.trigger_tokens' must be positive"
                    )
                if cfg.max_context_tokens > 0 and trigger_tokens > cfg.max_context_tokens:
                    raise ValueError(
                        "'set_auto_compact.payload.trigger_tokens' must not exceed max_context_tokens"
                    )
                updates["trigger_tokens"] = trigger_tokens

        if "trigger_ratio" in command.payload:
            trigger_ratio = command.payload["trigger_ratio"]
            if (
                isinstance(trigger_ratio, bool)
                or not isinstance(trigger_ratio, (int, float))
                or not math.isfinite(float(trigger_ratio))
            ):
                raise ValueError("'set_auto_compact.payload.trigger_ratio' must be a number")
            trigger_ratio = float(trigger_ratio)
            if trigger_ratio <= 0 or trigger_ratio > 1:
                raise ValueError(
                    "'set_auto_compact.payload.trigger_ratio' must be greater than 0 and at most 1"
                )
            updates["trigger_ratio"] = trigger_ratio

        if not updates:
            raise ValueError(
                "'set_auto_compact.payload' must include enabled, trigger_tokens, or trigger_ratio"
            )

        runner.agent._auto_compact = replace(cfg, **updates)
        await client.send(
            make_ack(
                "set_auto_compact",
                request_id=command.id,
                payload={
                    "auto_compact": self._agent_auto_compact(),
                    "context_usage": self._agent_context_usage(),
                },
            )
        )

    async def _handle_clear_queue(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        queue = command.payload.get("queue", "all")
        if queue == "all":
            cleared = runner.clear_all_queues()
        else:
            cleared = runner.clear_queue(self._queue_kind(queue))
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
        self._require_runner().agent.context.set_system_prompt(system_prompt)
        self._update_gui_launch_profile(system_prompt=system_prompt)
        await client.send(make_ack("set_system_prompt", request_id=command.id))

    async def _handle_switch_model(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        model_name = command.payload.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("'switch_model.payload.model_name' must be a non-empty string")
        self._require_runner().agent.set_model(model_name)
        self.model_name = model_name
        self._update_gui_launch_profile(model_name=model_name)
        await client.send(
            make_ack(
                "switch_model",
                request_id=command.id,
                payload={"model_name": model_name},
            )
        )
        self.emit(make_frame("core.ready", self._ready_payload()))

    async def _handle_refresh_models(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        provider = command.payload.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("'refresh_models.payload.provider' must be a non-empty string")
        provider = provider.strip()
        loop = asyncio.get_running_loop()
        models = await loop.run_in_executor(
            None,
            model_registry.refresh_provider_models,
            provider,
        )
        await client.send(
            make_ack(
                "refresh_models",
                request_id=command.id,
                payload={
                    "provider": provider,
                    "models": models,
                    "all_models": model_registry.list_models(),
                },
            )
        )

    async def _handle_apply_plugins(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        runner = self._require_runner()
        if not runner.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Apply plugins when the runner is idle.",
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
        selected_plugins = expand_plugin_dependencies(selected_plugins)

        plugin_configs = command.payload.get("plugin_configs", {})
        if plugin_configs is None:
            plugin_configs = {}
        if not isinstance(plugin_configs, dict):
            raise ValueError("'apply_plugins.payload.plugin_configs' must be an object")

        context_copy = runner.agent.context.copy()
        normalized_plugin_configs = {
            str(name): dict(cfg) if isinstance(cfg, dict) else {}
            for name, cfg in plugin_configs.items()
            if str(name) in KNOWN_PLUGINS
        }
        await self._replace_runner(
            model_name=self.model_name,
            selected_plugins=list(selected_plugins),
            plugin_configs=normalized_plugin_configs,
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

    async def _handle_plugin_action(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        plugin_id = command.payload.get("plugin_id")
        action = command.payload.get("action")
        arguments = command.payload.get("arguments", {})
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise ValueError("'plugin_action.payload.plugin_id' must be a non-empty string")
        if not isinstance(action, str) or action not in PLUGIN_ACTION_METHODS:
            raise ValueError("'plugin_action.payload.action' is not allowed")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("'plugin_action.payload.arguments' must be an object")

        review_result = self._resolve_runtime_review_action(
            plugin_id=plugin_id.strip(),
            action=action,
            arguments=arguments,
        )
        if review_result is not None:
            if review_result.get("success") is False:
                await client.send(
                    make_error(
                        str(review_result.get("error") or "Plugin action failed."),
                        request_id=command.id,
                        code="plugin_action_failed",
                        details=review_result,
                    )
                )
                return
            await client.send(
                make_ack(
                    "plugin_action",
                    request_id=command.id,
                    payload={
                        "plugin_id": plugin_id,
                        "action": action,
                        "result": review_result,
                        "resume_message_id": None,
                    },
                )
            )
            return

        plugin = self._find_plugin(plugin_id.strip())
        if plugin is None:
            await client.send(
                make_error(
                    f"Plugin is not active: {plugin_id}",
                    request_id=command.id,
                    code="plugin_not_active",
                )
            )
            return

        method = getattr(plugin, action, None)
        if not callable(method):
            await client.send(
                make_error(
                    f"Plugin action is not available: {action}",
                    request_id=command.id,
                    code="plugin_action_unavailable",
                )
            )
            return

        result = method(**arguments)
        if inspect.isawaitable(result):
            result = await result
        result_payload = self._plugin_action_result_payload(result)
        if result_payload.get("success") is False:
            await client.send(
                make_error(
                    str(result_payload.get("error") or "Plugin action failed."),
                    request_id=command.id,
                    code="plugin_action_failed",
                    details=result_payload,
                )
            )
            return

        resume_message_id = self._enqueue_plugin_action_next_message(
            action,
            result_payload,
            command.payload,
        )
        await client.send(
            make_ack(
                "plugin_action",
                request_id=command.id,
                payload={
                    "plugin_id": plugin_id,
                    "action": action,
                    "result": result_payload,
                    "resume_message_id": resume_message_id,
                },
            )
        )

    def _resolve_runtime_review_action(
        self,
        *,
        plugin_id: str,
        action: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        if action not in PLUGIN_ACTION_METHODS:
            return None
        review_id = arguments.get("review_id")
        if not isinstance(review_id, str) or not review_id.strip():
            return None
        request = self._review_broker.get(review_id.strip())
        if request is None:
            return None
        if request.plugin_id and plugin_id and request.plugin_id != plugin_id:
            return {
                "success": False,
                "output": None,
                "error": (
                    f"Review {review_id!r} belongs to plugin "
                    f"{request.plugin_id!r}, not {plugin_id!r}."
                ),
            }

        approved = action.startswith("approve_")
        feedback = str(arguments.get("feedback") or "")
        if not approved and not feedback.strip():
            return {
                "success": False,
                "output": None,
                "error": "feedback is required when rejecting.",
            }
        decision = RuntimeReviewDecision(
            approved=approved,
            feedback=feedback,
            modified_output=(
                str(arguments["modified_output"])
                if arguments.get("modified_output") is not None
                else None
            ),
            next_step_id=(
                str(arguments["next_step_id"])
                if arguments.get("next_step_id") is not None
                else None
            ),
            metadata={
                key: to_json_safe(value)
                for key, value in arguments.items()
                if key
                not in {"review_id", "feedback", "modified_output", "next_step_id"}
            },
        )
        if not self._review_broker.resolve(review_id.strip(), decision):
            return None
        return {
            "success": True,
            "output": {
                "review_id": review_id,
                "approved": approved,
                "rejected": not approved,
            },
            "error": "",
        }

    def _find_plugin(self, plugin_id: str) -> Any | None:
        for plugin in self._plugins:
            candidates = {
                str(getattr(plugin, "plugin_id", "") or ""),
                str(getattr(plugin, "name", "") or ""),
                str(getattr(plugin, "plugin_name", "") or ""),
                plugin.__class__.__name__,
            }
            if plugin_id in candidates:
                return plugin
        return None

    @staticmethod
    def _plugin_action_result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, ToolResult):
            return {
                "success": result.success,
                "output": to_json_safe(result.output),
                "error": result.error,
            }
        if isinstance(result, dict):
            return {"success": True, "output": to_json_safe(result), "error": ""}
        return {"success": True, "output": to_json_safe(result), "error": ""}

    def _enqueue_plugin_action_next_message(
        self,
        action: str,
        result_payload: dict[str, Any],
        command_payload: dict[str, Any],
    ) -> str | None:
        if command_payload.get("enqueue_next_message", True) is False:
            return None
        output = result_payload.get("output")
        if not isinstance(output, dict):
            return None
        next_message = output.get("next_message")
        if not isinstance(next_message, (str, list)) or not next_message:
            return None
        runner = self._runner
        if runner is None:
            return None
        return runner.submit_immediate_message(
            next_message,
            intent="plugin_action",
            metadata={
                "intent": "plugin_action",
                "display_message_type": "resume",
                "plugin_action": action,
                "auto_generated": True,
            },
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

    @staticmethod
    def _optional_message_index(command: CoreCommand) -> int | None:
        raw = command.payload.get("message_index")
        if raw is None:
            raw = command.payload.get("after_message_index")
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(
                "payload.message_index must be a non-negative integer when present"
            )
        return raw

    @staticmethod
    def _optional_context_message_id(command: CoreCommand) -> str | None:
        raw = command.payload.get("context_message_id")
        if raw is None:
            raw = command.payload.get("after_context_message_id")
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                "payload.context_message_id must be a non-empty string when present"
            )
        return raw

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
                "locked": m.locked,
                "lock_owner": m.lock_owner,
                "gui_launch_profile": to_json_safe(m.gui_launch_profile),
                "last_cwd": m.last_cwd,
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
        runner = self._require_runner()
        if not runner.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Create a new session when the runner is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return
        name = command.payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("'session_new.payload.name' must be a string when present")
        runner.agent.context.clear()
        runner.clear_all_queues()
        runner.agent.load_steer([])
        runner.agent.load_runtime(
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

    async def _handle_session_fork(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        runner = self._require_runner()
        if not runner.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Fork a session when the runner is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return
        source_session_id = command.payload.get("session_id")
        if source_session_id is not None and (
            not isinstance(source_session_id, str) or not source_session_id
        ):
            raise ValueError(
                "'session_fork.payload.session_id' must be a non-empty string when present"
            )
        name = command.payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("'session_fork.payload.name' must be a string when present")
        forked_from = source_session_id or sm.current_session_id
        context_message_id = self._optional_context_message_id(command)
        message_index = (
            None if context_message_id is not None else self._optional_message_index(command)
        )
        branch_result = None
        if message_index is None and context_message_id is None:
            session_id = sm.fork_session(session_id=source_session_id, name=name)
        elif context_message_id is not None:
            branch_result = sm.fork_session_after_message_id(
                session_id=source_session_id,
                name=name,
                context_message_id=context_message_id,
            )
            session_id = branch_result.session_id
            self._reset_runner_volatile_state()
            sm.save_now()
        else:
            branch_result = sm.fork_session_after_message(
                session_id=source_session_id,
                name=name,
                after_message_index=message_index,
            )
            session_id = branch_result.session_id
            self._reset_runner_volatile_state()
            sm.save_now()
        message_history = sm.read_message_history(session_id)
        context_usage = self._agent_context_usage()
        branch_payload = (
            self._context_branch_payload(branch_result)
            if branch_result is not None
            else {}
        )
        await client.send(
            make_ack(
                "session_fork",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "forked_from_session_id": forked_from,
                    "name": name or session_id,
                    "message_history": message_history,
                    "context_usage": context_usage,
                    **branch_payload,
                },
            )
        )

    async def _handle_session_rewind(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        runner = self._require_runner()
        if not runner.is_idle:
            await client.send(
                make_error(
                    "Agent is running. Rewind a session when the runner is idle.",
                    request_id=command.id,
                    code="busy",
                )
            )
            return
        context_message_id = self._optional_context_message_id(command)
        message_index = (
            None if context_message_id is not None else self._optional_message_index(command)
        )
        if context_message_id is not None:
            branch_result = sm.rewind_session_after_message_id(
                context_message_id=context_message_id,
            )
        else:
            if message_index is None:
                raise ValueError(
                    "'session_rewind.payload.context_message_id' or "
                    "'session_rewind.payload.message_index' is required"
                )
            branch_result = sm.rewind_session_after_message(
                after_message_index=message_index,
            )
        self._reset_runner_volatile_state()
        sm.save_now()
        message_history = sm.read_message_history(branch_result.session_id)
        context_usage = self._agent_context_usage()
        await client.send(
            make_ack(
                "session_rewind",
                request_id=command.id,
                payload={
                    "session_id": branch_result.session_id,
                    "message_history": message_history,
                    "context_usage": context_usage,
                    **self._context_branch_payload(branch_result),
                },
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

    async def _handle_session_rename(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        session_id = command.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                "'session_rename.payload.session_id' must be a non-empty string"
            )
        name = command.payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'session_rename.payload.name' must be a non-empty string")
        next_name = name.strip()
        sm.rename_session(session_id, next_name)
        await client.send(
            make_ack(
                "session_rename",
                request_id=command.id,
                payload={"session_id": session_id, "name": next_name},
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

    async def _handle_session_export_markdown(
        self,
        client: RuntimeClient,
        command: CoreCommand,
    ) -> None:
        sm = self._require_session_manager()
        requested_session_id = self._optional_session_id(command)
        session_id = requested_session_id or sm.current_session_id
        if session_id is None:
            raise ValueError("No active session to export")
        export = sm.export_markdown(
            requested_session_id,
            model=self.model_name,
        )
        await client.send(
            make_ack(
                "session_export_markdown",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "export": export.to_dict(include_markdown=True),
                },
            )
        )

    def _reset_runner_volatile_state(self) -> None:
        """Clear queues/runtime state after context is branched or rewound."""
        runner = self._require_runner()
        runner.clear_all_queues()
        runner.agent.load_steer([])
        runner.agent.load_runtime(
            {
                "version": 1,
                "current_tool_calls": [],
                "interrupted_tool_call_ids": [],
                "last_unsent_tool_results": [],
                "last_interrupt_reason": None,
            }
        )

    def _context_branch_payload(self, branch_result: Any | None) -> dict[str, Any]:
        if branch_result is None:
            return {}
        popped_user_message = getattr(branch_result, "popped_user_message", None)
        payload = {
            "message_index": getattr(branch_result, "message_index", None),
            "context_message_id": getattr(
                branch_result,
                "context_message_id",
                None,
            ),
            "target_role": getattr(branch_result, "target_role", None),
            "boundary_index": getattr(branch_result, "boundary_index", None),
            "popped_user_message": popped_user_message,
        }
        popped_text = self._message_plain_text(popped_user_message)
        if popped_text is not None:
            payload["popped_user_text"] = popped_text
        return payload

    @classmethod
    def _message_plain_text(cls, message: Any) -> str | None:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, list):
            return None
        text = cls._content_plain_text(content).strip()
        return text or None

    @classmethod
    def _content_plain_text(cls, content: list[Any]) -> str:
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                chunks.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                chunks.append(str(part.get("text") or ""))
            elif part_type == "steer" and isinstance(part.get("content"), list):
                chunks.append(cls._content_plain_text(part["content"]))
            elif part_type == "tool_result":
                nested = part.get("content")
                if isinstance(nested, list):
                    chunks.append(cls._content_plain_text(nested))
                elif nested is not None:
                    chunks.append(str(nested))
        return "\n\n".join(chunk for chunk in chunks if chunk)

    async def _replace_runner(
        self,
        *,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
        preserve_context: AgentContext | None,
    ) -> None:
        selected_plugins = expand_plugin_dependencies(selected_plugins)
        new_runner, new_task, new_plugins = await self._build_runner(
            model_name=model_name,
            selected_plugins=selected_plugins,
            plugin_configs=plugin_configs,
            context_to_restore=preserve_context,
        )
        session_manager = self._session_manager
        if session_manager is not None:
            session_manager.detach()
        await self._stop_runner(self._runner, self._runner_task, self._plugins)
        self._runner = new_runner
        self._runner_task = new_task
        self._plugins = new_plugins
        self._selected_plugins = list(selected_plugins)
        self._plugin_configs = {name: dict(cfg) for name, cfg in plugin_configs.items()}
        self._update_gui_launch_profile(
            selected_plugins=self._selected_plugins,
            plugin_configs=self._plugin_configs,
        )
        if session_manager is not None:
            session_manager.attach(
                new_runner.agent,
                new_runner,
                event_bus=getattr(new_runner.agent, "_event_bus", None),
            )
        self.emit(make_frame("core.ready", self._ready_payload()))

    async def _build_runner(
        self,
        *,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
        context_to_restore: AgentContext | None,
    ) -> tuple[AgentRunner, asyncio.Task, list[Any]]:
        model_overrides: dict[str, Any] = {}
        if self._max_context_tokens is not None:
            model_overrides["max_context_tokens"] = self._max_context_tokens
        model = model_registry.create_model(model_name, **model_overrides)
        auto_compact = AutoCompactConfig(
            enabled=True,
            max_context_tokens=self._explicit_context_limit_for_model(model),
        )
        # Debug assert: the GUI/core runtime should never rely on HawiAgent's
        # implicit auto-compact default.
        debug_assert(
            auto_compact is not None,
            "CoreRuntime must pass an explicit AutoCompactConfig to HawiAgent",
        )
        plugins = await self._create_plugins(selected_plugins, plugin_configs)
        blob_store = self._blob_store
        agent = HawiAgent(
            model=model,
            plugins=plugins,
            system_prompt=self.system_prompt,
            max_iterations=None,
            streaming=True,
            auto_compact=auto_compact,
            model_input_resolver=(
                (lambda request: resolve_blob_references_for_model(request, blob_store))
                if blob_store is not None
                else None
            ),
        )
        agent.review_broker = self._review_broker
        self._apply_extra_tool_parameters(agent)
        if context_to_restore is not None:
            agent.set_context(context_to_restore.copy())
            agent.context.tool_call_context = ToolCallContext(agent)

        runner = AgentRunner(agent)
        agent.event_bus.subscribe(self._on_hawi_event)
        runner_task = asyncio.create_task(runner.run_forever(poll_interval=0.1))
        return runner, runner_task, plugins

    def _explicit_context_limit_for_model(self, model: Any) -> int:
        if self._max_context_tokens is not None:
            return self._max_context_tokens
        getter = getattr(model, "get_max_context_tokens", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
        return AutoCompactConfig().max_context_tokens

    def _apply_extra_tool_parameters(self, agent: HawiAgent) -> None:
        for parameter in self._extra_tool_parameters:
            agent.plugins.add_tool_parameter_injection(
                ToolParameterInjection(
                    name=parameter.name,
                    schema=self._extra_tool_parameter_json_schema(parameter),
                    required=parameter.required,
                )
            )
        if self._extra_tool_parameters:
            defs = agent.plugins.get_tool_definitions()
            agent.context.tool_definitions = defs if defs else None

    @staticmethod
    def _extra_tool_parameter_json_schema(parameter: ExtraToolParameter) -> dict[str, Any]:
        schema = {
            **parameter.schema,
        }
        schema.setdefault("description", parameter.description)
        if parameter.name == "tool_call_purpose":
            schema["default"] = None
        return schema

    async def _stop_runner(
        self,
        runner: AgentRunner | None,
        runner_task: asyncio.Task | None,
        plugins: list[Any],
    ) -> None:
        if runner is not None:
            try:
                await runner.interrupt("shutdown")
            except Exception:
                logger.exception("Failed to interrupt runner during shutdown")
            runner.stop()
        if runner_task and not runner_task.done():
            runner_task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)
        if runner is not None:
            runner.agent.event_bus.unsubscribe(self._on_hawi_event)
        if runner is not None:
            try:
                runner.agent.event_bus.close(wait=True, timeout=2.0)
            except Exception:
                logger.exception("Failed to close runner event bus")
        await self._close_plugins(plugins)

    async def _create_plugins(
        self,
        selected_plugins: list[str],
        plugin_configs: dict[str, dict[str, Any]],
    ) -> list[Any]:
        plugins: list[Any] = []
        for plugin_key in selected_plugins:
            cfg = dict(plugin_configs.get(plugin_key, {}))
            plugin = await create_plugin(plugin_key, cfg)
            descriptor = get_plugin_descriptor(plugin_key)
            if hasattr(plugin, "bind_plugin_identity"):
                plugin.bind_plugin_identity(
                    plugin_id=plugin_key,
                    plugin_name=descriptor.display_name,
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

    def _require_runner(self) -> AgentRunner:
        if self._runner is None:
            raise RuntimeError("Core runtime is not ready")
        return self._runner

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
        if self._runner is None:
            return {
                "ready": False,
                "runner_state": "STOPPED",
                "agent_state": "STOPPED",
                "queue_lengths": {"normal": 0, "high_prio": 0, "urgent": 0},
                "queue_messages": {"normal": [], "high_prio": [], "urgent": []},
                "model_name": self.model_name,
            }
        queue_messages_getter = getattr(self._runner, "get_queue_messages", None)
        queue_messages: dict[str, list[dict[str, Any]]] = cast(
            dict[str, list[dict[str, Any]]],
            queue_messages_getter()
            if callable(queue_messages_getter)
            else {"normal": [], "high_prio": [], "urgent": []}
        )
        queue_messages = {**queue_messages, "urgent": []}
        pending_input_getter = getattr(
            self._runner.agent,
            "get_pending_input_messages",
            None,
        )
        if callable(pending_input_getter):
            pending_inputs = cast(list[dict[str, Any]], pending_input_getter())
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
            "runner_state": self._runner.state.name,
            "agent_state": self._runner.agent_state.name,
            "queue_lengths": self._runner.get_queue_lengths(),
            "queue_messages": queue_messages,
            "model_name": self.model_name,
            "control": self._runner.control_snapshot(),
        }
        context_usage = self._agent_context_usage()
        if context_usage is not None:
            payload["context_usage"] = context_usage
        auto_compact = self._agent_auto_compact()
        if auto_compact is not None:
            payload["auto_compact"] = auto_compact
        return payload

    def _session_manifest_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"last_cwd": str(Path.cwd().resolve())}
        if self._gui_launch_profile is None:
            return metadata
        profile = self._effective_gui_launch_profile()
        if profile:
            metadata["gui_launch_profile"] = to_json_safe(profile)
        return metadata

    def _effective_gui_launch_profile(self) -> dict[str, Any] | None:
        if self._gui_launch_profile is not None:
            return dict(self._gui_launch_profile)
        return {
            "version": 1,
            "modelName": self.model_name,
            "systemPrompt": self.system_prompt,
            "selectedPlugins": list(self._selected_plugins),
            "pluginConfigs": to_json_safe(self._plugin_configs),
        }

    def _update_gui_launch_profile(self, **updates: Any) -> None:
        if self._gui_launch_profile is None:
            return
        profile = self._effective_gui_launch_profile() or {"version": 1}
        for key, value in updates.items():
            if key == "model_name":
                profile["modelName"] = value
            elif key == "system_prompt":
                profile["systemPrompt"] = value
            elif key == "selected_plugins":
                profile["selectedPlugins"] = list(value or [])
            elif key == "plugin_configs":
                profile["pluginConfigs"] = to_json_safe(value or {})
            else:
                profile[key] = to_json_safe(value)
        profile.setdefault("version", 1)
        profile.setdefault("modelName", self.model_name)
        profile.setdefault("systemPrompt", self.system_prompt)
        profile.setdefault("selectedPlugins", list(self._selected_plugins))
        profile.setdefault("pluginConfigs", to_json_safe(self._plugin_configs))
        self._gui_launch_profile = profile

    def _agent_context_usage(self) -> dict[str, Any] | None:
        if self._runner is None:
            return None
        context = getattr(self._runner.agent, "context", None)
        saved_getter = getattr(context, "context_usage_snapshot", None)
        if callable(saved_getter):
            saved_snapshot = saved_getter()
            to_dict = getattr(saved_snapshot, "to_dict", None)
            if callable(to_dict):
                return to_json_safe(to_dict())
        getter = getattr(self._runner.agent, "context_usage", None)
        if not callable(getter):
            return None
        snapshot = getter()
        to_dict = getattr(snapshot, "to_dict", None)
        if callable(to_dict):
            return to_json_safe(to_dict())
        return None

    def _agent_auto_compact(self) -> dict[str, Any] | None:
        if self._runner is None:
            return None
        cfg = getattr(self._runner.agent, "_auto_compact", None)
        if not isinstance(cfg, AutoCompactConfig):
            return None
        token_limit = cfg.token_limit()
        token_limit_ratio = (
            token_limit / cfg.max_context_tokens
            if cfg.max_context_tokens > 0
            else None
        )
        return to_json_safe({
            "enabled": cfg.enabled,
            "max_context_tokens": cfg.max_context_tokens,
            "trigger_tokens": cfg.trigger_tokens,
            "trigger_ratio": cfg.trigger_ratio,
            "max_trigger_ratio": cfg.max_trigger_ratio,
            "compression_budget": cfg.compression_budget,
            "token_limit": token_limit,
            "token_limit_ratio": token_limit_ratio,
        })


def load_model_configs(
    extra_paths: list[str] | None = None,
    *,
    include_user: bool = True,
) -> list[Path]:
    """Load model configs in core-cli order and return paths that existed."""
    loaded: list[Path] = []
    # ModelRegistry has its own lazy auto-loader for ~/.hawi and workspace configs.
    # The engine owns this chain explicitly so GUI and CLI metadata see the
    # same deterministic order.
    model_registry._auto_load_needed = False  # type: ignore[attr-defined]
    workspace_root = find_git_root(Path.cwd())
    candidates = [
        workspace_root / ".hawi" / "models.yaml",
        workspace_root / "models.yaml",
    ]
    if include_user:
        candidates.append(Path.home() / ".hawi" / "models.yaml")
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
