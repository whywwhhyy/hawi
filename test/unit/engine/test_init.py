from __future__ import annotations

from pathlib import Path

import pytest

from hawi.engine.__main__ import build_parser
from hawi.engine.init import prepare_hawi_dir, render_template_text


def test_parser_accepts_optional_hawi_dir() -> None:
    args = build_parser().parse_args(["init", "/tmp/.hawi"])

    assert args.command == "init"
    assert args.hawi_dir == "/tmp/.hawi"


def test_prepare_hawi_dir_copies_template_when_no_dir_is_given(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAWI_NO_AUTO_LOAD", "1")

    result = prepare_hawi_dir()
    config_path = tmp_path / ".hawi" / "models.yaml"

    assert result.config_dir == tmp_path / ".hawi"
    assert result.changed is True
    assert config_path.exists()

    text = config_path.read_text(encoding="utf-8")
    assert "{{HAWI_PROJECT_NAME}}" not in text
    assert "providers:" in text
    assert "# providers:" in text


def test_prepare_hawi_dir_uses_nearest_git_root_when_no_dir_is_given(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    nested = repo / "packages" / "demo"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.setenv("HAWI_NO_AUTO_LOAD", "1")

    result = prepare_hawi_dir()

    assert result.config_dir == repo / ".hawi"
    assert (repo / ".hawi" / "models.yaml").exists()
    assert not (nested / ".hawi").exists()


def test_prepare_hawi_dir_leaves_existing_template_files_alone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".hawi" / "models.yaml"
    config_path.parent.mkdir()
    config_path.write_text("already here", encoding="utf-8")

    result = prepare_hawi_dir()

    assert result.changed is False
    assert result.skipped is True
    assert config_path.read_text(encoding="utf-8") == "already here"


def test_prepare_hawi_dir_accepts_existing_explicit_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / ".hawi"
    config_dir.mkdir()

    result = prepare_hawi_dir(hawi_dir=config_dir)

    assert result.config_dir == config_dir
    assert result.files == ()


def test_prepare_hawi_dir_rejects_missing_explicit_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Hawi config directory not found"):
        prepare_hawi_dir(hawi_dir=tmp_path / "missing" / ".hawi")


def test_render_template_text_replaces_known_tokens_only() -> None:
    rendered = render_template_text(
        "{{HAWI_PROJECT_NAME}} {{UNKNOWN_TOKEN}}",
        {"HAWI_PROJECT_NAME": "demo"},
    )

    assert rendered == "demo {{UNKNOWN_TOKEN}}"
