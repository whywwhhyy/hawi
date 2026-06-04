"""Command entry point for `hawi-engine`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import readline  # noqa: F401  Enables input line editing/history on supported terminals.
except ImportError:
    readline = None  # type: ignore[assignment]

from rich.console import Console

from hawi.agent import AutoCompactConfig, HawiAgent
from hawi.agent.printers.rich import RichPrinter
from hawi.models import model_registry
from hawi.utils.config_loader import load_config_file

from .init import prepare_hawi_dir
from .runtime import (
    DEFAULT_SYSTEM_PROMPT,
    CoreRuntime,
    load_model_configs,
    parse_extra_tool_parameters,
    token_from_arg_or_env,
)
from .inspect import build_inspect_payload, build_plugin_tool_preview_payload
from .protocol import json_dumps
from .gateway import GATEWAY_REGISTRY, discover_gateways
from . import builtin_gateways  # noqa: F401  side-effect import: registers built-in gateways
from . import http_gateway  # noqa: F401  side-effect import: registers HttpGateway

warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args)
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"hawi-engine: {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hawi-engine",
        description="Run Hawi as an always-on JSON protocol core process.",
    )
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser(
        "init",
        help="Create a starter Hawi environment config",
        description="Create a starter .hawi directory without overwriting existing files.",
    )
    init_parser.add_argument(
        "hawi_dir",
        nargs="?",
        default=None,
        help=(
            "Existing Hawi config directory to use. "
            "If omitted, the bundled template is copied into the nearest "
            "Git root's .hawi, or ./.hawi when no Git root is found."
        ),
    )
    parser.add_argument("--model", default=None, help="Model factory name from models.yaml")
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Run a minimal streaming Markdown chat CLI instead of the JSON "
            "gateway. Plugins are not loaded in this mode."
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help=(
            "Override the selected model's context window for automatic "
            "context compaction."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print GUI metadata JSON and exit",
    )
    parser.add_argument(
        "--inspect-plugin",
        default=None,
        help="With --inspect, temporarily load one plugin and print its tool metadata JSON",
    )
    parser.add_argument(
        "--readonly",
        action="store_true",
        help="Run a read-only session browser/search engine without loading a model.",
    )
    parser.add_argument(
        "--session-root",
        default=None,
        help="Session root for --readonly. Defaults to ~/.hawi/sessions.",
    )
    discover_gateways()
    parser.add_argument(
        "--gateway",
        choices=sorted(GATEWAY_REGISTRY.keys()),
        default="stdio",
        help="Gateway to use (built-in: stdio, tcp, http; plus any installed plugins)",
    )
    # Backward-compat alias for one release. Maps onto the same dest.
    parser.add_argument(
        "--transport",
        choices=sorted(GATEWAY_REGISTRY.keys()),
        default=None,
        dest="transport",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for tcp/http gateways")
    parser.add_argument("--port", type=int, default=None, help="Port for tcp/http gateways")

    # Let each gateway register its own args.
    for gateway in GATEWAY_REGISTRY.values():
        gateway.register_args(parser)
    parser.add_argument(
        "--token",
        default=None,
        help="Optional client token. Defaults to HAWI_CORE_TOKEN when set.",
    )
    parser.add_argument(
        "--models-config",
        action="append",
        default=[],
        help="Extra models.yaml path. May be passed more than once.",
    )
    parser.add_argument(
        "--model-provider-config",
        action="append",
        default=[],
        help=(
            "JSON/YAML/TOML file containing temporary provider property "
            "overrides keyed by provider name. May be passed more than once."
        ),
    )
    parser.add_argument(
        "--refresh-provider",
        action="append",
        default=[],
        help=(
            "Temporarily refresh a provider's model list from its remote API. "
            "May be passed more than once."
        ),
    )
    parser.add_argument(
        "--no-user-models",
        action="store_true",
        help="Do not load ~/.hawi/models.yaml; use workspace and explicit model configs only.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Initial system prompt",
    )
    parser.add_argument(
        "--plugins",
        default="",
        help="Comma-separated plugin names to enable at startup, e.g. hawi/filesystem,hawi/shell",
    )
    parser.add_argument(
        "--extra-tool-parameter",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "TYPE", "DESCRIPTION"),
        help=(
            "Framework-level tool parameter to expose to every tool schema and "
            "strip before tool execution. Injected parameters are required in "
            "tool call schemas. May be passed more than once. "
            "Use quotes around DESCRIPTION when it contains spaces. "
            "Supported types: str, int, float, bool, object, array."
        ),
    )
    parser.add_argument(
        "--extra-tool-parameter-json",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "Framework-level tool parameter directive as JSON. Expected fields: "
            "name, schema or type, description, and optional required. May be "
            "passed more than once."
        ),
    )
    parser.add_argument(
        "--plugin-config",
        default=None,
        help="JSON/YAML/TOML file containing plugin config object keyed by plugin name",
    )
    parser.add_argument(
        "--keep-session-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When loading a saved session, keep the persisted system prompt "
            "instead of regenerating declared system-prompt hook content."
        ),
    )
    parser.add_argument(
        "--profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request provider profiling data from models that support it.",
    )
    parser.add_argument(
        "--gui-launch-profile",
        default=None,
        help="JSON object with GUI session launch profile metadata to persist in session manifests.",
    )
    parser.add_argument(
        "--initial-session-id",
        default=None,
        help="Optional initial in-memory session id for GUI-managed engines.",
    )
    parser.add_argument(
        "--initial-session-name",
        default=None,
        help="Optional initial in-memory session name for GUI-managed engines.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.3,
        help="Seconds between core.status broadcasts",
    )
    parser.add_argument(
        "--outbound-queue-size",
        type=int,
        default=100,
        help="Per-client outbound frame queue size",
    )
    parser.add_argument(
        "--max-frame-size",
        type=int,
        default=16 * 1024 * 1024,
        help="Max TLV frame body size in bytes (stdio/tcp only). Default 16 MiB.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append backend debug logs to this file. Logs never go to stdout.",
    )
    parser.add_argument(
        "--blob-dir",
        default=".hawi/blobs",
        help="Directory under which inbound/ and outbound/ blob sandboxes live. Default '.hawi/blobs'.",
    )
    parser.add_argument(
        "--blob-quota-mb",
        type=int,
        default=1024,
        help="Per-direction quota in MiB. Default 1024 (1 GiB).",
    )
    parser.add_argument(
        "--blob-disabled",
        action="store_true",
        help="Disable the blob store entirely. blob.* commands return blob_disabled errors.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr")
    return parser


def configure_logging(args: argparse.Namespace) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    stderr_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stderr_handler]
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    if args.log_file:
        logging.getLogger(__name__).info("Writing hawi-engine backend log to %s", args.log_file)


async def async_main(args: argparse.Namespace) -> None:
    if getattr(args, "command", None) == "init":
        result = prepare_hawi_dir(hawi_dir=args.hawi_dir)
        if args.hawi_dir:
            print(f"Using Hawi config directory: {result.config_dir}")
        elif result.changed:
            print(f"Initialized Hawi config directory: {result.config_dir}")
            for item in result.files:
                if item.action == "created":
                    print(f"{item.action}: {item.path}")
            if result.skipped:
                print("Some existing files were left unchanged.")
            print("Set the provider API key environment variable before running a model.")
        else:
            print(f"Hawi config directory already exists: {result.config_dir}")
        return

    if args.inspect:
        if args.inspect_plugin:
            plugin_configs = load_plugin_config(args.plugin_config)
            plugin_key = args.inspect_plugin.strip()
            print(json_dumps(await build_plugin_tool_preview_payload(
                plugin_key,
                plugin_configs.get(plugin_key, {}),
            )))
            return
        load_model_configs(args.models_config, include_user=not args.no_user_models)
        apply_model_provider_config_overrides(args.model_provider_config)
        refresh_model_providers(args.refresh_provider)
        print(json_dumps(build_inspect_payload()))
        return

    if args.readonly:
        from .readonly import ReadOnlyRuntime

        runtime = ReadOnlyRuntime(
            session_root=args.session_root,
            token=token_from_arg_or_env(args.token),
        )
        await runtime.start()
        gateway_name = args.transport if args.transport else args.gateway
        gateway = GATEWAY_REGISTRY.get(gateway_name)
        if gateway is None:
            raise RuntimeError(f"Unsupported gateway: {gateway_name}")
        await gateway.serve(runtime, args)
        return

    loaded = load_model_configs(args.models_config, include_user=not args.no_user_models)
    apply_model_provider_config_overrides(args.model_provider_config)
    refresh_model_providers(args.refresh_provider)
    available = model_registry.list_models()

    if not args.model:
        raise RuntimeError("--model is required unless --inspect is used")

    if args.model not in available:
        loaded_text = ", ".join(str(path) for path in loaded) or "none"
        if available:
            available_text = ", ".join(available)
            raise RuntimeError(
                f"Unknown model '{args.model}'. Loaded configs: {loaded_text}. "
                f"Available models: {available_text}"
            )
        raise RuntimeError(
            f"No model configurations available. Loaded configs: {loaded_text}. "
            "Create ~/.hawi/models.yaml, <workspace>/.hawi/models.yaml, "
            "<workspace>/models.yaml, "
            "pass --models-config PATH, or run `hawi-engine init`."
        )

    if args.chat:
        await run_chat_cli(args)
        return

    selected_plugins = parse_plugins(args.plugins)
    plugin_configs = load_plugin_config(args.plugin_config)
    extra_tool_parameters = parse_extra_tool_parameters(
        args.extra_tool_parameter,
        args.extra_tool_parameter_json,
    )

    blob_store = None
    if not args.blob_disabled:
        from hawi.engine.blob import BlobStore

        blob_store = BlobStore(
            root=Path(args.blob_dir).expanduser().resolve(),
            quota_bytes=args.blob_quota_mb * 1024 * 1024,
        )

    runtime = CoreRuntime(
        model_name=args.model,
        system_prompt=args.system_prompt,
        selected_plugins=selected_plugins,
        plugin_configs=plugin_configs,
        extra_tool_parameters=extra_tool_parameters,
        max_context_tokens=args.max_context_tokens,
        keep_session_system_prompt=args.keep_session_system_prompt,
        profiling=args.profiling,
        gui_launch_profile=parse_gui_launch_profile(args.gui_launch_profile),
        initial_session_id=args.initial_session_id,
        initial_session_name=args.initial_session_name,
        model_config_paths=loaded,
        token=token_from_arg_or_env(args.token),
        status_interval=args.status_interval,
        blob_store=blob_store,
    )
    await runtime.start()

    gateway_name = args.transport if args.transport else args.gateway
    gateway = GATEWAY_REGISTRY.get(gateway_name)
    if gateway is None:
        raise RuntimeError(f"Unsupported gateway: {gateway_name}")
    await gateway.serve(runtime, args)


class ChatRichPrinter(RichPrinter):
    """Minimal Markdown renderer for the interactive chat CLI."""

    def _print_usage(self, usage: Any) -> None:
        return None

    def stop_live(self) -> None:
        self._stop_live()


async def run_chat_cli(args: argparse.Namespace) -> None:
    """Run a minimal interactive streaming chat loop."""
    console = Console()
    model_overrides: dict[str, Any] = {}
    if args.max_context_tokens is not None:
        model_overrides["max_context_tokens"] = args.max_context_tokens
    model = model_registry.create_model(args.model, **model_overrides)
    auto_compact = (
        AutoCompactConfig(enabled=True, max_context_tokens=args.max_context_tokens)
        if args.max_context_tokens is not None
        else True
    )
    agent = HawiAgent(
        model=model,
        plugins=[],
        system_prompt=args.system_prompt,
        max_iterations=None,
        streaming=True,
        auto_compact=auto_compact,
        profiling=args.profiling,
    )
    printer = ChatRichPrinter(
        show_reasoning=False,
        show_tools=False,
        show_error_stack=False,
        streaming=True,
        console=console,
    )
    agent.subscribe(printer.handle)

    console.print(f"[dim]Hawi chat · model: {args.model} · Ctrl+C interrupts, /exit quits[/dim]")
    while True:
        try:
            prompt = await asyncio.to_thread(console.input, "[bold cyan]>>> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q", "/exit", "/quit", "/q"}:
            return
        await _run_chat_turn(agent, prompt, console, printer)


async def _run_chat_turn(
    agent: HawiAgent,
    prompt: str,
    console: Console,
    printer: ChatRichPrinter,
) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(agent.arun(prompt))
    interrupted = False

    def request_interrupt() -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        agent.interrupt("user")
        task.cancel()

    with _temporary_sigint_handler(loop, request_interrupt), _suppress_stdin_echo():
        try:
            await task
        except asyncio.CancelledError:
            if not interrupted:
                raise
        except KeyboardInterrupt:
            request_interrupt()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        finally:
            if interrupted:
                printer.stop_live()
                agent.clear_interrupt_state()
                console.print("\n[dim]Interrupted.[/dim]")
            else:
                console.print()


@contextlib.contextmanager
def _temporary_sigint_handler(
    loop: asyncio.AbstractEventLoop,
    callback: Any,
) -> Iterator[None]:
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        loop.add_signal_handler(signal.SIGINT, callback)
        restore_with_loop = True
    except (NotImplementedError, RuntimeError):
        restore_with_loop = False

        def handler(signum: int, frame: Any) -> None:
            loop.call_soon_threadsafe(callback)

        signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        if restore_with_loop:
            loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_handler)


@contextlib.contextmanager
def _suppress_stdin_echo() -> Iterator[None]:
    if not sys.stdin.isatty():
        yield
        return
    try:
        import termios
    except ImportError:
        yield
        return

    fd = sys.stdin.fileno()
    try:
        previous = termios.tcgetattr(fd)
    except termios.error:
        yield
        return
    next_attrs = previous[:]
    next_attrs[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, next_attrs)
        yield
    finally:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        with contextlib.suppress(termios.error):
            termios.tcflush(fd, termios.TCIFLUSH)


def parse_plugins(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def refresh_model_providers(providers: list[str]) -> None:
    for provider in providers:
        provider = provider.strip()
        if provider:
            model_registry.refresh_provider_models(provider)


def load_plugin_config(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = load_config_file(Path(path))
    if not isinstance(data, dict):
        raise RuntimeError("--plugin-config must point to a config object")
    return {
        str(name): dict(cfg) if isinstance(cfg, dict) else {}
        for name, cfg in data.items()
    }


def apply_model_provider_config_overrides(paths: list[str] | None) -> None:
    overrides: dict[str, dict[str, Any]] = {}
    for raw_path in paths or []:
        data = load_config_file(Path(raw_path))
        if not isinstance(data, dict):
            raise RuntimeError("--model-provider-config must point to a config object")
        for provider, cfg in _iter_model_provider_config_overrides(data):
            current = overrides.setdefault(provider, {})
            current.update(cfg)
    if overrides:
        model_registry.apply_provider_config_overrides(overrides)


def _iter_model_provider_config_overrides(
    data: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    providers = data.get("providers")
    if isinstance(providers, dict):
        for name, cfg in providers.items():
            if not isinstance(cfg, dict):
                continue
            properties = cfg.get("properties") if "properties" in cfg else cfg
            if isinstance(properties, dict):
                yield str(name), dict(properties)
        return

    for name, cfg in data.items():
        if name == "providers":
            continue
        if isinstance(cfg, dict):
            properties = cfg.get("properties") if "properties" in cfg else cfg
            if isinstance(properties, dict):
                yield str(name), dict(properties)


def parse_gui_launch_profile(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("--gui-launch-profile must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("--gui-launch-profile must be a JSON object")
    return parsed


if __name__ == "__main__":
    main()
