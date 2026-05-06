from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hawi_core_cli.inspect import build_inspect_payload


def test_inspect_payload_contains_models_and_plugin_catalog() -> None:
    payload = build_inspect_payload()

    assert payload["version"] == "hawi.core.v1"
    assert isinstance(payload["models"], list)
    assert payload["default_system_prompt"]
    catalog = payload["plugin_catalog"]
    assert {item["key"] for item in catalog} >= {
        "filesystem",
        "shell",
        "web",
        "skills",
        "python_interpreter",
        "mcp",
    }
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
        [sys.executable, "-m", "hawi_core_cli", "--inspect"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["models"] == ["test/fake-model"]
    assert payload["plugin_catalog"]
