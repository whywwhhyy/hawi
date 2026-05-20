from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hawi.agent.context import AgentContext
from hawi.plugin import HookContext
from hawi.builtin_plugins.environ_prompt_plugin import EnvironPromptPlugin


def test_project_steering_loads_scoped_files_for_first_matching_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (repo / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("outside agents", encoding="utf-8")
    (repo / "AGENTS.md").write_text("root agents", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("root claude", encoding="utf-8")
    (package / "AGENTS.md").write_text("pkg agents", encoding="utf-8")
    (package / "CLAUDE.md").write_text("pkg claude", encoding="utf-8")

    monkeypatch.chdir(package)
    plugin = EnvironPromptPlugin(config_path=str(_project_steering_config(tmp_path)))

    text = _inject_system_prompt(plugin)

    assert "outside agents" not in text
    assert "root agents" in text
    assert "pkg agents" in text
    assert "root claude" not in text
    assert "pkg claude" not in text
    assert "Selected filename: AGENTS.md" in text
    assert f"Scope: {repo}" in text
    assert f"Scope: {package}" in text
    assert text.index("root agents") < text.index("pkg agents")


def test_project_steering_falls_back_to_later_filename_when_first_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text("root claude", encoding="utf-8")
    (package / "CLAUDE.md").write_text("pkg claude", encoding="utf-8")

    monkeypatch.chdir(package)
    plugin = EnvironPromptPlugin(config_path=str(_project_steering_config(tmp_path)))

    text = _inject_system_prompt(plugin)

    assert "Selected filename: CLAUDE.md" in text
    assert "root claude" in text
    assert "pkg claude" in text


def test_project_steering_stops_at_cwd_without_project_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "AGENTS.md").write_text("parent agents", encoding="utf-8")
    (work / "AGENTS.md").write_text("work agents", encoding="utf-8")

    monkeypatch.chdir(work)
    plugin = EnvironPromptPlugin(config_path=str(_project_steering_config(tmp_path)))

    text = _inject_system_prompt(plugin)

    assert "work agents" in text
    assert "parent agents" not in text


def test_project_steering_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("root agents", encoding="utf-8")
    config_path = tmp_path / "environ_prompt.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "system_prompt": {
                    "enabled": True,
                    "include_session_info": False,
                    "include_project_steering": False,
                    "text": "static prompt",
                    "files": [],
                },
                "user_prompt": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    plugin = EnvironPromptPlugin(config_path=str(config_path))

    text = _inject_system_prompt(plugin)

    assert "static prompt" in text
    assert "root agents" not in text


def test_project_steering_truncates_large_files(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("abcdef", encoding="utf-8")
    config_path = _project_steering_config(tmp_path, max_file_bytes=3)

    monkeypatch.chdir(repo)
    plugin = EnvironPromptPlugin(config_path=str(config_path))

    text = _inject_system_prompt(plugin)

    assert "abc" in text
    assert "def" not in text
    assert "Truncated by EnvironPromptPlugin at 3 bytes" in text


def test_system_prompt_orders_stable_project_content_before_session_info(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("stable project rules", encoding="utf-8")
    config_path = _project_steering_config(tmp_path, include_session_info=True)

    monkeypatch.chdir(repo)
    plugin = EnvironPromptPlugin(config_path=str(config_path))

    text = _inject_system_prompt(plugin)

    assert "stable project rules" in text
    assert "Session environment:" in text
    assert text.index("stable project rules") < text.index("Session environment:")


def _project_steering_config(
    tmp_path: Path,
    max_file_bytes: int = 65536,
    *,
    include_session_info: bool = False,
) -> Path:
    path = tmp_path / "environ_prompt.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "system_prompt": {
                    "enabled": True,
                    "include_session_info": include_session_info,
                    "include_project_steering": True,
                    "project_steering": {
                        "filenames": ["AGENTS.md", "CLAUDE.md"],
                        "project_root_markers": [".git", ".hawi"],
                        "max_file_bytes": max_file_bytes,
                    },
                    "text": "",
                    "files": [],
                },
                "user_prompt": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def _inject_system_prompt(plugin: EnvironPromptPlugin) -> str:
    agent = SimpleNamespace(context=AgentContext(system_prompt=[]))
    plugin.inject_system_prompt_env(
        agent,
        HookContext(run_id="test-run", iteration=0),
    )
    return "\n".join(
        str(part.get("text", ""))
        for part in agent.context.system_prompt or []
        if isinstance(part, dict)
    )
