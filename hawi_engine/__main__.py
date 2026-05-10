"""Command entry point for `hawi-engine`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

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
from .inspect import build_inspect_payload
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
            "If omitted, the bundled template is copied into ./.hawi."
        ),
    )
    parser.add_argument("--model", default=None, help="Model factory name from models.yaml")
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
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Initial system prompt",
    )
    parser.add_argument(
        "--plugins",
        default="",
        help="Comma-separated plugin keys to enable at startup",
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
        "--plugin-config",
        default=None,
        help="JSON/YAML/TOML file containing plugin config object keyed by plugin name",
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

    loaded = load_model_configs(args.models_config)
    available = model_registry.list_models()
    if args.inspect:
        print(json_dumps(build_inspect_payload()))
        return

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
            "Create ~/.hawi/models.yaml, ./.hawi/models.yaml, ./models.yaml, "
            "pass --models-config PATH, or run `hawi-engine init`."
        )

    selected_plugins = parse_plugins(args.plugins)
    plugin_configs = load_plugin_config(args.plugin_config)
    extra_tool_parameters = parse_extra_tool_parameters(args.extra_tool_parameter)

    blob_store = None
    if not args.blob_disabled:
        from hawi_engine.blob import BlobStore

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


def parse_plugins(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


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


if __name__ == "__main__":
    main()
