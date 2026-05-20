from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hawi.engine.inspect import build_inspect_payload
from hawi.engine.plugin_registry import KNOWN_PLUGINS


def test_inspect_payload_contains_models_and_plugin_catalog() -> None:
    payload = build_inspect_payload()

    assert payload["version"] == "hawi.core.v1"
    assert isinstance(payload["models"], list)
    assert payload["default_system_prompt"]
    catalog = payload["plugin_catalog"]
    assert {item["key"] for item in catalog} == set(KNOWN_PLUGINS)
    assert {item["name"] for item in catalog} == set(KNOWN_PLUGINS)
    assert all("display_name" in item for item in catalog)
    assert all("description" in item for item in catalog)
    assert all("dependencies" in item for item in catalog)
    assert all("schema" in item and "defaults" in item for item in catalog)


def test_hawi_core_inspect_cli_outputs_json(tmp_path: Path) -> None:
    config_dir = tmp_path / ".hawi"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        """
providers:
  - name: test
    adapter: OpenAIModel
    model_ids:
      - fake-model
    properties:
      base_url: http://localhost:1234/v1
      api_key: test
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(Path.cwd()),
    }
    result = subprocess.run(
        [sys.executable, "-m", "hawi.engine", "--inspect"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["models"] == ["test/fake-model"]
    assert payload["plugin_catalog"]
