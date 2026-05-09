"""Environment initialization helpers for ``hawi-core``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

TemplateAction = Literal["created", "skipped"]

DEFAULT_TEMPLATE_NAME = "hawi_template"
HAWI_DIR_NAME = ".hawi"
TEMPLATE_TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


@dataclass(frozen=True)
class InitFileResult:
    """Result for one rendered template file."""

    path: Path
    action: TemplateAction


@dataclass(frozen=True)
class InitResult:
    """Result of initializing a Hawi environment directory."""

    config_dir: Path
    files: tuple[InitFileResult, ...]

    @property
    def changed(self) -> bool:
        return any(item.action == "created" for item in self.files)

    @property
    def skipped(self) -> bool:
        return any(item.action == "skipped" for item in self.files)


def prepare_hawi_dir(
    *,
    hawi_dir: str | Path | None = None,
    template_name: str = DEFAULT_TEMPLATE_NAME,
) -> InitResult:
    """Resolve or initialize the Hawi config directory.

    If ``hawi_dir`` is explicit it must already exist. Without an explicit
    directory, the bundled template is copied into the current working tree.
    """

    if hawi_dir is not None:
        config_dir = Path(hawi_dir).expanduser().resolve()
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Hawi config directory not found: {config_dir}")
        return InitResult(config_dir=config_dir, files=())

    destination_root = Path.cwd().resolve()
    config_dir = destination_root / HAWI_DIR_NAME
    values = _template_values(destination_root, config_dir)
    template_root = files("hawi_core_cli").joinpath("templates", template_name)
    if not template_root.is_dir():
        raise FileNotFoundError(f"Hawi init template not found: {template_name}")

    file_results: list[InitFileResult] = []
    for source_file, relative_path in _iter_template_files(template_root):
        rendered_path = _render_relative_path(relative_path, values)
        destination = destination_root / rendered_path
        action = _render_template_file(
            source_file=source_file,
            destination=destination,
            values=values,
        )
        file_results.append(InitFileResult(path=destination, action=action))

    return InitResult(config_dir=config_dir, files=tuple(file_results))


def render_template_text(text: str, values: dict[str, str]) -> str:
    """Replace known ``{{TOKEN}}`` placeholders in template text."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, match.group(0))

    return TEMPLATE_TOKEN_RE.sub(replace, text)


def _template_values(target_dir: Path, config_dir: Path) -> dict[str, str]:
    project_name = target_dir.name or "hawi"
    return {
        "HAWI_PROJECT_NAME": project_name,
        "HAWI_TARGET_DIR": str(target_dir),
        "HAWI_CONFIG_DIR": str(config_dir),
        "HAWI_MODELS_CONFIG_PATH": str(config_dir / "models.yaml"),
    }


def _iter_template_files(template_root, prefix: Path = Path()):
    for child in template_root.iterdir():
        relative_path = prefix / child.name
        if child.is_dir():
            yield from _iter_template_files(child, relative_path)
        elif child.is_file():
            yield child, relative_path


def _render_relative_path(path: Path, values: dict[str, str]) -> Path:
    return Path(*(render_template_text(part, values) for part in path.parts))


def _render_template_file(
    *,
    source_file,
    destination: Path,
    values: dict[str, str],
) -> TemplateAction:
    if destination.exists():
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)

    raw = source_file.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        destination.write_bytes(raw)
    else:
        destination.write_text(render_template_text(text, values), encoding="utf-8")
    return "created"
