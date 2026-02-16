"""
MCP 插件测试

运行测试:
    pytest hawi_plugins/mcp/test_mcp_plugin.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hawi_plugins.mcp_plugin import MCPPlugin, MCPClient, MCPClientPool, MCPTool, MCPResource


class TestMCPTool:
    """测试 MCPTool 类"""

    def test_creation(self):
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_to_dict(self):
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
        )
        result = tool.to_dict()
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        assert result["inputSchema"] == {"type": "object"}


class TestMCPResource:
    """测试 MCPResource 类"""

    def test_creation(self):
        resource = MCPResource(
            uri="file:///test.txt",
            name="test.txt",
            description="A test file",
            mime_type="text/plain",
            size=100,
        )
        assert resource.uri == "file:///test.txt"
        assert resource.name == "test.txt"
        assert resource.description == "A test file"
        assert resource.mime_type == "text/plain"
        assert resource.size == 100

    def test_to_dict(self):
        resource = MCPResource(
            uri="file:///test.txt",
            name="test.txt",
            description="A test file",
            mime_type="text/plain",
        )
        result = resource.to_dict()
        assert result["uri"] == "file:///test.txt"
        assert result["name"] == "test.txt"
        assert result["description"] == "A test file"
        assert result["mimeType"] == "text/plain"

    def test_to_dict_optional_fields(self):
        resource = MCPResource(
            uri="file:///test.txt",
            name="test.txt",
        )
        result = resource.to_dict()
        assert result["uri"] == "file:///test.txt"
        assert result["name"] == "test.txt"
        assert "description" not in result
        assert "mimeType" not in result
        assert "size" not in result


class TestMCPPlugin:
    """测试 MCPPlugin 类"""

    def test_creation(self):
        plugin = MCPPlugin()
        assert plugin is not None
        assert not plugin._connected

    def test_add_stdio_server(self):
        plugin = MCPPlugin()
        plugin.add_stdio_server("test", "python", ["-m", "server"])
        
        assert len(plugin._server_configs) == 1
        config = plugin._server_configs[0]
        assert config["type"] == "stdio"
        assert config["name"] == "test"
        assert config["command"] == "python"
        assert config["args"] == ["-m", "server"]

    def test_add_sse_server(self):
        plugin = MCPPlugin()
        plugin.add_sse_server("test", "http://localhost:3000/sse")
        
        assert len(plugin._server_configs) == 1
        config = plugin._server_configs[0]
        assert config["type"] == "sse"
        assert config["name"] == "test"
        assert config["url"] == "http://localhost:3000/sse"

    def test_add_server_from_command_stdio(self):
        plugin = MCPPlugin()
        plugin.add_server_from_command("test", "python -m server arg1")
        
        assert len(plugin._server_configs) == 1
        config = plugin._server_configs[0]
        assert config["type"] == "stdio"
        assert config["name"] == "test"
        assert config["command"] == "python"
        assert config["args"] == ["-m", "server", "arg1"]

    def test_add_server_from_command_sse(self):
        plugin = MCPPlugin()
        plugin.add_server_from_command("test", "http://localhost:3000/sse")
        
        assert len(plugin._server_configs) == 1
        config = plugin._server_configs[0]
        assert config["type"] == "sse"
        assert config["name"] == "test"
        assert config["url"] == "http://localhost:3000/sse"

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """测试连接和断开"""
        plugin = MCPPlugin()
        plugin.add_stdio_server("test", "echo", ["hello"])
        
        # 模拟客户端
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.tools = []
        mock_client.resources = []
        
        with patch("hawi_plugins.mcp_plugin.plugin.MCPClient") as MockClient:
            MockClient.from_stdio.return_value = mock_client
            
            await plugin.connect()
            assert plugin._connected
            
            await plugin.disconnect()
            assert not plugin._connected

    def test_list_mcp_resources_empty(self):
        """测试列出资源（无资源时）"""
        plugin = MCPPlugin()
        plugin._connected = True
        plugin._resource_wrappers = {}
        
        result = plugin.list_mcp_resources()
        assert "暂无" in result or "暂无可用的 MCP 资源" in result

    def test_list_mcp_resources_with_items(self):
        """测试列出资源（有资源时）"""
        plugin = MCPPlugin()
        plugin._connected = True
        
        # 创建模拟资源
        mock_resource = MagicMock()
        mock_resource.uri = "file:///test.txt"
        mock_resource.name = "test.txt"
        mock_resource.description = "A test file"
        mock_resource.mime_type = "text/plain"
        
        mock_wrapper = MagicMock()
        mock_wrapper._mcp_resource = mock_resource
        plugin._resource_wrappers = {"file:///test.txt": mock_wrapper}
        
        result = plugin.list_mcp_resources()
        assert "file:///test.txt" in result
        assert "test.txt" in result
        assert "A test file" in result

    def test_get_mcp_resource_not_found(self):
        """测试获取不存在的资源"""
        plugin = MCPPlugin()
        plugin._connected = True
        plugin._resource_wrappers = {}
        
        result = plugin.get_mcp_resource("file:///nonexistent.txt")
        assert "错误" in result or "未找到" in result


class TestMCPClient:
    """测试 MCPClient 类"""

    def test_from_stdio(self):
        client = MCPClient.from_stdio("python", ["-m", "server"])
        assert client._connection_type == "stdio"
        assert client._server_params.command == "python"
        assert client._server_params.args == ["-m", "server"]

    def test_from_sse(self):
        client = MCPClient.from_sse("http://localhost:3000/sse", {"Authorization": "token"})
        assert client._connection_type == "sse"
        assert client._sse_url == "http://localhost:3000/sse"
        assert client._sse_headers == {"Authorization": "token"}

    def test_from_command_stdio(self):
        client = MCPClient.from_command("python -m server arg1")
        assert client._connection_type == "stdio"
        assert client._server_params.command == "python"
        assert client._server_params.args == ["-m", "server", "arg1"]

    def test_from_command_sse(self):
        client = MCPClient.from_command("http://localhost:3000/sse")
        assert client._connection_type == "sse"
        assert client._sse_url == "http://localhost:3000/sse"

    def test_not_connected_initially(self):
        client = MCPClient.from_stdio("echo", ["hello"])
        assert not client.is_connected


class TestMCPClientPool:
    """测试 MCPClientPool 类"""

    def test_creation(self):
        pool = MCPClientPool()
        assert pool is not None
        assert pool.all_tools == []
        assert pool.all_resources == []

    def test_add_remove_client(self):
        pool = MCPClientPool()
        mock_client = MagicMock()
        
        pool.add_client("test", mock_client)
        # 内部实现细节，这里主要测试不抛出异常

    @pytest.mark.asyncio
    async def test_connect_all(self):
        pool = MCPClientPool()
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.tools = []
        mock_client.resources = []
        
        pool.add_client("test", mock_client)
        await pool.connect_all()
        
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        pool = MCPClientPool()
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.tools = []
        mock_client.resources = []
        
        pool.add_client("test", mock_client)
        await pool.connect_all()
        await pool.disconnect_all()
        
        mock_client.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
