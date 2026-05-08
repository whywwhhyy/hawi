"""EnvironPromptPlugin — inject environment context into system and user prompts.

This plugin reads a YAML config file from ``.hawi/environ_prompt.yaml``
(or falls back to hardcoded defaults) and enriches the conversation with:

- **System prompt** (one-time at session start): session start date, OS
  platform, timezone, hardware info, and user-specified text/file content.
- **User prompt** (before each user message): current working directory,
  files modified since the last user prompt, and user-specified text/file
  content.

All injected content is clearly demarcated as framework-provided so the
model can distinguish it from actual user input.
"""

from __future__ import annotations

import datetime
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

from hawi.plugin import (
    HawiPlugin,
    HookContext,
    before_conversation,
    before_session,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

ENVIRON_TAG_BEGIN = "<hawi-environ>"
ENVIRON_TAG_END = "</hawi-environ>"
"""XML-like tags that wrap every block of injected environment info.

These make it easy for the model (and for future prompts) to identify and
potentially ignore/replace framework-injected content.
"""

# ---------------------------------------------------------------------------
# Default configuration (used when no YAML config file is found)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "system_prompt": {
        "enabled": True,
        "include_session_info": True,
        "text": "",
        "files": [],
    },
    "user_prompt": {
        "enabled": True,
        "include_cwd": True,
        "include_modified_files": True,
        "text": "",
        "files": [],
    },
}

# ---------------------------------------------------------------------------
# Config file search paths (first match wins)
# ---------------------------------------------------------------------------

CONFIG_FILENAME = "environ_prompt.yaml"
CONFIG_CANDIDATES = [
    Path(".hawi") / CONFIG_FILENAME,       # project-local
    Path.home() / ".hawi" / CONFIG_FILENAME,  # user-global
]


# ===================================================================
# Plugin implementation
# ===================================================================

class EnvironPromptPlugin(HawiPlugin):
    """Inject environment information into system and user prompts.

    Reads its configuration from ``.hawi/environ_prompt.yaml`` (project-local
    first, then ``~/.hawi/environ_prompt.yaml``). When neither file exists
    the hardcoded default configuration is used, which enables both system-
    and user-prompt injection.

    **System-prompt injection** (``before_session`` hook, runs once):
        Appends session-level metadata (start date, platform, timezone,
        hardware info) and any user-specified static text or file content
        to ``agent.context.system_prompt``.

    **User-prompt injection** (``before_conversation`` hook, per turn):
        Inserts a user-role message carrying dynamic information (current
        working directory, files modified since the last user prompt) and
        any user-specified text or file content **before** the actual user
        message in the conversation context.

    All injected blocks are wrapped in ``<hawi-environ>…</hawi-environ>``
    markers and prefixed with a note declaring them as framework-injected.
    """

    def __init__(self) -> None:
        self._config = self._load_config()
        self._last_prompt_ts: float = 0.0
        """Timestamp (seconds since epoch) of the last user-prompt injection.

        Used to compute "files modified since last user prompt".
        """
        self._session_started: bool = False
        """Whether the ``before_session`` hook has already run."""

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load configuration from YAML file or fall back to defaults.

        Searches ``.hawi/environ_prompt.yaml`` in the current working
        directory, then ``~/.hawi/environ_prompt.yaml``.  If neither file
        exists the hardcoded :attr:`DEFAULT_CONFIG` is returned.
        """
        for candidate in CONFIG_CANDIDATES:
            resolved = candidate.resolve()
            if resolved.is_file():
                try:
                    return _parse_yaml_config(resolved)
                except Exception:
                    logger.exception(
                        "Failed to parse environ prompt config from %s; "
                        "falling back to defaults",
                        resolved,
                    )
                    break
        return dict(DEFAULT_CONFIG)  # shallow copy is sufficient here

    # ------------------------------------------------------------------
    # GUI registration metadata (optional)
    # ------------------------------------------------------------------

    @classmethod
    def gui_config_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "title": "Enabled",
                    "description": "Enable environment prompt injection",
                    "default": True,
                },
            },
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict[str, Any]:
        return {"enabled": True}

    # ------------------------------------------------------------------
    # Clone support
    # ------------------------------------------------------------------

    def clone(self) -> EnvironPromptPlugin:
        new = EnvironPromptPlugin()
        # Copy runtime state that matters for the cloned agent
        new._last_prompt_ts = self._last_prompt_ts
        new._session_started = self._session_started
        return new

    # ==============================================================
    # Hook: before_session — inject static env info into system prompt
    # ==============================================================

    @before_session
    def inject_system_prompt_env(
        self,
        agent: Any,
        ctx: HookContext,
    ) -> None:
        """Append static environment info to the system prompt (once)."""
        if self._session_started:
            return
        self._session_started = True

        cfg = (self._config.get("system_prompt") or {})
        if not cfg.get("enabled", True):
            return

        parts: list[str] = []

        # -- session-level information ---------------------------------
        if cfg.get("include_session_info", True):
            parts.append(_format_session_info())

        # -- user-specified static text --------------------------------
        text = cfg.get("text")
        if text and isinstance(text, str) and text.strip():
            parts.append(text.strip())

        # -- user-specified static file content -------------------------
        files = cfg.get("files")
        if files and isinstance(files, list):
            for entry in files:
                content = _read_file_entry(entry)
                if content is not None:
                    parts.append(content)

        if not parts:
            return

        body = "\n\n".join(parts)
        stamped = _stamp_environ_block(body)

        # Append to the end of system_prompt
        current = list(agent.context.system_prompt or [])
        # If there's already an <hawi-environ> block, replace it in-place
        # so we don't pile up stale blocks across session restarts.
        cleaned = _strip_existing_environ_blocks(current)
        cleaned.append({"type": "text", "text": stamped})
        agent.context.system_prompt = cleaned

    # ==============================================================
    # Hook: before_conversation — inject dynamic env info before user msg
    # ==============================================================

    @before_conversation
    def inject_user_prompt_env(
        self,
        agent: Any,
        ctx: HookContext,
    ) -> None:
        """Insert a user-role message with dynamic env info before the
        actual user message in the conversation context."""
        cfg = (self._config.get("user_prompt") or {})
        if not cfg.get("enabled", True):
            return

        # Build dynamic env info
        parts: list[str] = []

        if cfg.get("include_cwd", True):
            parts.append(_format_cwd())

        if cfg.get("include_modified_files", True):
            parts.append(_format_modified_files(self._last_prompt_ts))

        text = cfg.get("text")
        if text and isinstance(text, str) and text.strip():
            parts.append(text.strip())

        files = cfg.get("files")
        if files and isinstance(files, list):
            for entry in files:
                content = _read_file_entry(entry)
                if content is not None:
                    parts.append(content)

        if not parts:
            return

        body = "\n\n".join(parts)
        stamped = _stamp_environ_block(body)

        # Update timestamp for next "modified since" calculation
        self._last_prompt_ts = time.time()

        # Find the last user message (the one just added by the framework)
        # and insert our env-info message right before it.
        messages = agent.context.messages
        insert_index = _find_last_user_insert_index(messages)
        env_message: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": stamped}],
            "name": None,
            "metadata": {"source": "environ_prompt_plugin"},
        }
        agent.context.inject(env_message, position=insert_index)


# ===================================================================
# Internal helpers
# ===================================================================


def _parse_yaml_config(path: Path) -> dict[str, Any]:
    """Parse a YAML config file, falling back to JSON if PyYAML is absent."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        try:
            import json

            with path.open("r", encoding="utf-8") as fh:
                return dict(json.load(fh))
        except Exception:
            raise
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return dict(data)


def _stamp_environ_block(body: str) -> str:
    """Wrap *body* in environment markers with a framework note."""
    header = (
        "[Environment Information — auto-injected by "
        "EnvironPromptPlugin. This is NOT user input.]"
    )
    return (
        f"\n\n{ENVIRON_TAG_BEGIN}\n"
        f"{header}\n\n"
        f"{body}\n"
        f"{ENVIRON_TAG_END}\n"
    )


def _strip_existing_environ_blocks(
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove any content parts that contain ``<hawi-environ>`` markers."""
    return [
        part
        for part in parts
        if not (
            isinstance(part, dict)
            and part.get("type") == "text"
            and ENVIRON_TAG_BEGIN in str(part.get("text", ""))
        )
    ]


def _find_last_user_insert_index(messages: list[dict[str, Any]]) -> int:
    """Return the index before the most recent user message.

    If no user message exists, returns ``len(messages)`` (append).
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages)


# ------------------------------------------------------------------
# Fact helpers
# ------------------------------------------------------------------


def _format_session_info() -> str:
    """Return session-level environment facts."""
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    tz_name = now.strftime("%Z")
    tz_offset = now.strftime("%z")

    lines: list[str] = [
        f"Session started: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Operating system: {platform.system()}",
        f"Platform: {platform.platform(terse=True)}",
        f"Architecture: {platform.machine()}",
        f"Timezone: {tz_name} (UTC{tz_offset})",
    ]

    cpu = os.cpu_count()
    if cpu is not None:
        lines.append(f"CPU cores: {cpu}")

    node = platform.node()
    if node:
        lines.append(f"Hostname: {node}")

    return "Session environment:\n" + "\n".join(f"  {line}" for line in lines)


def _format_cwd() -> str:
    """Return the current working directory."""
    try:
        cwd = Path.cwd().resolve()
    except Exception:
        cwd = Path(".").resolve()
    return f"Current working directory: {cwd}"


def _format_modified_files(since_ts: float) -> str:
    """Return a bullet list of files modified since *since_ts* under CWD.

    When no files are found (or *since_ts* is 0, meaning first call),
    a brief notice is returned instead.
    """
    if since_ts <= 0:
        return "Recent file changes: (none — first user prompt in this session)"

    try:
        cwd = Path.cwd().resolve()
    except Exception:
        return "Recent file changes: (unable to determine working directory)"

    modified: list[str] = []
    # Walk common source directories; limit depth to avoid huge scans.
    for root, _dirs, files in os.walk(cwd, topdown=True):
        # Skip hidden directories and common caches
        rel = Path(root).relative_to(cwd)
        parts = rel.parts
        if parts and parts[0].startswith("."):
            continue
        if parts and parts[0] in {"node_modules", "__pycache__", ".git",
                                   ".venv", "venv", ".tox", "dist",
                                   "build", ".egg-info"}:
            continue

        for name in files:
            if name.startswith("."):
                continue
            fpath = Path(root) / name
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                continue
            if mtime >= since_ts:
                try:
                    rel_path = fpath.relative_to(cwd)
                except ValueError:
                    rel_path = fpath
                modified.append(str(rel_path))

        # Limit depth: don't walk deeper than 4 levels
        depth = len(rel.parts)
        if depth >= 3:
            _dirs[:] = [d for d in _dirs if not d.startswith(".")]

    if not modified:
        return "Recent file changes: (no files modified since last user prompt)"

    # Sort by modification time (most recent first)
    try:
        modified.sort(
            key=lambda p: (cwd / p).stat().st_mtime,
            reverse=True,
        )
    except OSError:
        pass

    max_items = 30
    shown = modified[:max_items]
    lines = ["Recent file changes (since last user prompt):"]
    for item in shown:
        try:
            mtime = (cwd / item).stat().st_mtime
            age = _format_age(time.time() - mtime)
            lines.append(f"  - {item} ({age})")
        except OSError:
            lines.append(f"  - {item}")

    if len(modified) > max_items:
        lines.append(f"  … and {len(modified) - max_items} more")

    return "\n".join(lines)


def _format_age(seconds: float) -> str:
    """Format a duration delta as a human-readable string."""
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    minutes_rem = minutes % 60
    if hours < 24:
        return f"{hours}h {minutes_rem}m ago" if minutes_rem else f"{hours}h ago"
    days = int(hours // 24)
    return f"{days}d ago"


def _read_file_entry(entry: Any) -> str | None:
    """Read file content from a config entry.

    *entry* can be:
    - A string (treated as a file path)
    - A dict with ``path`` and optional ``label`` key

    Returns ``None`` if the file does not exist or cannot be read.
    """
    if isinstance(entry, str):
        filepath = entry
        label = None
    elif isinstance(entry, dict):
        filepath = entry.get("path", "")
        label = entry.get("label")
    else:
        return None

    if not filepath or not isinstance(filepath, str):
        return None

    try:
        resolved = Path(filepath).resolve(strict=False)
        if not resolved.is_file():
            logger.debug("EnvironPromptPlugin: file not found — %s", resolved)
            return None
        content = resolved.read_text(encoding="utf-8")
    except Exception:
        logger.exception("EnvironPromptPlugin: failed to read file — %s", filepath)
        return None

    lines = [f"File: {label or filepath}", "```", content.rstrip("\n"), "```"]
    return "\n".join(lines)
