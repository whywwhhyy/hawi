from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from hawi.utils.config_loader import (
    Config,
    ConfigLoader,
    ConfigLoaderError,
    ConfigSubstitutionError,
    load_config_file,
    load_config_from_directory_chain,
    substitute_config,
)


def test_load_skips_missing_dirs_and_files(tmp_path: Path) -> None:
    existing = tmp_path / "exists"
    existing.mkdir()

    config = load_config_from_directory_chain(
        [tmp_path / "missing", existing],
        "hawi.yaml",
    )

    assert config.raw == {}
    assert config.data == {}


def test_loads_existing_files_in_reverse_order_and_earlier_dirs_override(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    project = tmp_path / "project"
    agent = tmp_path / "agent"
    for directory in [root, project, agent]:
        directory.mkdir()

    _write_yaml(
        root / "hawi.yaml",
        """
        name: root
        shared: root
        nested:
          a: root-a
          shared: root-nested
        list_value:
          - root
        """,
    )
    _write_yaml(
        project / "hawi.yaml",
        """
        name: project
        project_only: true
        nested:
          b: project-b
          shared: project-nested
        """,
    )
    _write_yaml(
        agent / "hawi.yaml",
        """
        name: agent
        agent_only: true
        nested:
          c: agent-c
          shared: agent-nested
        list_value:
          - agent
        """,
    )

    config = load_config_from_directory_chain(
        [root, project, agent],
        "hawi.yaml",
    )

    result = config.data
    assert result["name"] == "root"
    assert result["shared"] == "root"
    assert result["project_only"] is True
    assert result["agent_only"] is True
    assert result["nested"] == {
        "a": "root-a",
        "b": "project-b",
        "c": "agent-c",
        "shared": "root-nested",
    }
    assert result["list_value"] == ["root"]


def test_preserves_raw_scalar_types_when_merging(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_yaml(
        config_dir / "hawi.yaml",
        """
        int_value: 3
        float_value: 1.5
        bool_value: true
        none_value: null
        list_value:
          - a
          - b
        """,
    )

    result = load_config_from_directory_chain([config_dir], "hawi.yaml").data

    assert result["int_value"] == 3
    assert result["float_value"] == 1.5
    assert result["bool_value"] is True
    assert result["none_value"] is None
    assert result["list_value"] == ["a", "b"]


def test_substitution_root_path_resolves_from_root() -> None:
    result = substitute_config(
        {
            "agent": {"name": "alice"},
            "message": "hello {agent.name}",
            "nested": {"message": "agent is {agent.name}"},
        }
    )

    assert result["message"] == "hello alice"
    assert result["nested"]["message"] == "agent is alice"


def test_substitution_dot_path_resolves_from_current_parent() -> None:
    result = substitute_config(
        {
            "agent": {"name": "root-agent"},
            "tools": {
                "python": {
                    "name": "python",
                    "label": "tool {.name}",
                },
            },
        }
    )

    assert result["tools"]["python"]["label"] == "tool python"


def test_substitution_parent_dots_walk_parent_chain() -> None:
    result = substitute_config(
        {
            "agent": {"name": "root-agent"},
            "groups": {
                "dev": {
                    "name": "backend",
                    "shell": {
                        "name": "shell",
                        "label": "{..name}/{.name}",
                    },
                },
            },
        }
    )

    assert result["groups"]["dev"]["shell"]["label"] == "backend/shell"


def test_substitution_supports_multiple_tokens_and_recursive_values() -> None:
    result = substitute_config(
        {
            "agent": {
                "first": "alice",
                "name": "{.first}",
            },
            "env": {
                "name": "prod",
                "url": "https://{.name}.example.com/{agent.name}",
            },
        }
    )

    assert result["agent"]["name"] == "alice"
    assert result["env"]["url"] == "https://prod.example.com/alice"


def test_substitution_only_references_scalar_values() -> None:
    with pytest.raises(ConfigSubstitutionError, match="scalar"):
        substitute_config(
            {
                "agent": {
                    "metadata": {"role": "admin"},
                    "tags": ["a", "b"],
                },
                "bad": "{agent.metadata}",
            }
        )

    with pytest.raises(ConfigSubstitutionError, match="scalar"):
        substitute_config(
            {
                "agent": {"tags": ["a", "b"]},
                "bad": "{agent.tags}",
            }
        )


def test_substitution_missing_reference_raises_clear_error() -> None:
    with pytest.raises(ConfigSubstitutionError, match="agent.name"):
        substitute_config({"message": "hello {agent.name}"})


def test_substitution_escaped_open_brace_is_literal() -> None:
    result = substitute_config(
        {
            "agent": {"name": "alice"},
            "literal": r"\{agent.name}",
            "mixed": r"literal \{agent.name}, real {agent.name}",
        }
    )

    assert result["literal"] == "{agent.name}"
    assert result["mixed"] == "literal {agent.name}, real alice"


def test_substitution_runs_after_merge(tmp_path: Path) -> None:
    root = tmp_path / "root"
    agent = tmp_path / "agent"
    root.mkdir()
    agent.mkdir()
    _write_json(agent / "hawi.json", {"message": "hello {agent.name}", "agent": {"name": "base"}})
    _write_json(root / "hawi.json", {"agent": {"name": "override"}})

    config = load_config_from_directory_chain([root, agent], "hawi.json")
    result = config.data

    assert result["agent"]["name"] == "override"
    assert result["message"] == "hello override"
    assert config.raw["message"] == "hello {agent.name}"


def test_empty_yaml_file_is_empty_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hawi.yaml").write_text("", encoding="utf-8")

    result = load_config_from_directory_chain([config_dir], "hawi.yaml").data

    assert result == {}


def test_non_mapping_yaml_file_raises_value_error(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_yaml(
        config_dir / "hawi.yaml",
        """
        - a
        - b
        """,
    )

    with pytest.raises(ConfigLoaderError, match="mapping"):
        load_config_from_directory_chain([config_dir], "hawi.yaml")


def test_load_supports_toml_by_suffix(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hawi.toml").write_text(
        textwrap.dedent(
            """
            title = "hello {agent.name}"

            [agent]
            name = "alice"
            """
        ).strip() + "\n",
        encoding="utf-8",
    )

    config = load_config_from_directory_chain([config_dir], "hawi.toml")

    assert config.raw["title"] == "hello {agent.name}"
    assert config.data["title"] == "hello alice"


def test_load_config_file_uses_suffix_and_rejects_unknown_suffix(tmp_path: Path) -> None:
    config_file = tmp_path / "hawi.toml"
    config_file.write_text("name = \"alice\"\n", encoding="utf-8")

    assert load_config_file(config_file) == {"name": "alice"}

    bad = tmp_path / "hawi.conf"
    bad.write_text("name: alice\n", encoding="utf-8")
    with pytest.raises(ConfigLoaderError, match="Unsupported"):
        load_config_file(bad)


def test_config_keeps_raw_and_resubstitutes_after_raw_update() -> None:
    config = Config({"agent": {"name": "alice"}, "message": "hello {agent.name}"})

    assert config.raw["message"] == "hello {agent.name}"
    assert config.data["message"] == "hello alice"

    config.set_raw("agent.name", "bob")

    assert config.raw["agent"]["name"] == "bob"
    assert config.data["message"] == "hello bob"


def test_config_save_writes_raw_or_substituted_config(tmp_path: Path) -> None:
    config = Config({"agent": {"name": "alice"}, "message": "hello {agent.name}"})
    raw_path = tmp_path / "raw.json"
    substituted_path = tmp_path / "substituted.yaml"
    toml_path = tmp_path / "raw.toml"

    config.save(raw_path)
    config.save(substituted_path, substituted=True)
    config.save(toml_path)

    assert json.loads(raw_path.read_text(encoding="utf-8"))["message"] == "hello {agent.name}"
    assert load_config_file(substituted_path)["message"] == "hello alice"
    assert load_config_file(toml_path)["agent"]["name"] == "alice"


def test_find_files_supports_multiple_filenames_in_each_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    project = tmp_path / "project"
    root.mkdir()
    project.mkdir()
    _write_json(root / "a.json", {"name": "root-a"})
    _write_json(project / "b.json", {"name": "project-b"})

    loader = ConfigLoader(["a.json", "b.json"])

    assert loader.find_files([root, project]) == [
        root / "a.json",
        project / "b.json",
    ]


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")
