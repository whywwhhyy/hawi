from __future__ import annotations

import pytest

import hawi_plugins.mcp_plugin.plugin as mcp_plugin_module
from hawi_plugins.mcp_plugin import MCPPlugin


class _FakePool:
    def __init__(self):
        self.clients: list[tuple[str, object]] = []
        self.all_tools = []
        self.all_resources = []

    def add_client(self, name: str, client: object):
        self.clients.append((name, client))

    async def connect_all(self):
        return None

    async def disconnect_all(self):
        return None


class _FakeClient:
    @staticmethod
    def from_stdio(command: str, args: list[str], env: dict[str, str] | None = None):
        return {"mode": "stdio", "command": command, "args": args, "env": env}

    @staticmethod
    def from_sse(url: str, headers: dict[str, str] | None = None):
        return {"mode": "sse", "url": url, "headers": headers}


@pytest.mark.asyncio
async def test_connect_loads_servers_from_config_path(tmp_path, monkeypatch):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        """{
  "mcpServers": {
    "fs": {
      "command": "python",
      "args": ["-m", "mock_server"]
    }
  }
}""",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_plugin_module, "MCPClientPool", _FakePool)
    monkeypatch.setattr(mcp_plugin_module, "MCPClient", _FakeClient)

    plugin = MCPPlugin(config_path=str(cfg))
    await plugin.connect()

    assert plugin._connected is True
    assert len(plugin._server_configs) == 1
    assert plugin._server_configs[0]["name"] == "fs"


@pytest.mark.asyncio
async def test_connect_raises_when_config_missing(tmp_path):
    plugin = MCPPlugin(config_path=str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        await plugin.connect()


@pytest.mark.asyncio
async def test_connect_raises_when_config_invalid_json(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text("{not json", encoding="utf-8")
    plugin = MCPPlugin(config_path=str(cfg))
    with pytest.raises(ValueError, match="Invalid MCP config JSON"):
        await plugin.connect()


@pytest.mark.asyncio
async def test_connect_raises_when_mcp_servers_empty(tmp_path):
    cfg = tmp_path / "empty.json"
    cfg.write_text("{}", encoding="utf-8")
    plugin = MCPPlugin(config_path=str(cfg))
    with pytest.raises(ValueError, match="No MCP servers found"):
        await plugin.connect()
