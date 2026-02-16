"""
MCP (Model Context Protocol) 客户端实现

支持通过 stdio 和 SSE 连接到 MCP 服务器，发现和调用工具、资源。
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any
from pydantic import AnyUrl

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

class MCPTool:
    """MCP 工具封装"""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MCPResource:
    """MCP 资源封装"""

    def __init__(
        self,
        uri: str,
        name: str,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
    ):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.size = size

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.size is not None:
            result["size"] = self.size
        return result


class MCPClient:
    """
    MCP 客户端

    用于连接到 MCP 服务器，发现和调用工具、资源。

    使用示例:
        # stdio 连接
        client = MCPClient.from_stdio("python", ["-m", "mcp_server_example"])
        
        # SSE 连接
        client = MCPClient.from_sse("http://localhost:3000/sse")
        
        async with client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
    """

    # 连接类型: "stdio" 或 "sse"
    _connection_type: str
    # stdio 连接参数
    _server_params: StdioServerParameters
    # SSE 连接参数
    _sse_url: str
    _sse_headers: dict[str, str] | None
    # 会话相关
    _session: ClientSession | None
    _exit_stack: AsyncExitStack | None
    _connected: bool
    # 工具和资源
    _tools: list[MCPTool]
    _resources: list[MCPResource]

    def __init__(self):
        self._session = None
        self._exit_stack = None
        self._tools = []
        self._resources = []
        self._connected = False
        # 以下属性在 from_stdio/from_sse 中设置
        self._connection_type = ""
        self._server_params = None  # type: ignore
        self._sse_url = ""
        self._sse_headers = None

    @classmethod
    def from_stdio(cls, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> MCPClient:
        """
        创建 stdio 连接的 MCP 客户端

        Args:
            command: 命令（如 "python", "npx" 等）
            args: 命令参数
            env: 环境变量

        Returns:
            MCPClient 实例
        """
        instance = cls()
        instance._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        instance._connection_type = "stdio"
        return instance

    @classmethod
    def from_sse(cls, url: str, headers: dict[str, str] | None = None) -> MCPClient:
        """
        创建 SSE 连接的 MCP 客户端

        Args:
            url: SSE 服务器 URL
            headers: 请求头

        Returns:
            MCPClient 实例
        """
        instance = cls()
        instance._sse_url = url
        instance._sse_headers = headers
        instance._connection_type = "sse"
        return instance

    @classmethod
    def from_command(cls, command: str) -> MCPClient:
        """
        从命令字符串创建 MCP 客户端

        支持格式:
            - "python -m mcp_server"
            - "npx -y @modelcontextprotocol/server-filesystem /tmp"
            - "http://localhost:3000/sse" (自动识别为 SSE)

        Args:
            command: 命令字符串

        Returns:
            MCPClient 实例
        """
        command = command.strip()
        
        # 检测是否为 URL
        if command.startswith(("http://", "https://")):
            return cls.from_sse(command)
        
        # 解析命令
        parts = command.split()
        if not parts:
            raise ValueError("Empty command")
        
        return cls.from_stdio(parts[0], parts[1:])

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """连接到 MCP 服务器"""
        if self._connected:
            return

        self._exit_stack = AsyncExitStack()

        try:
            if self._connection_type == "stdio":
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(self._server_params)
                )
                read_stream, write_stream = stdio_transport
            else:  # sse
                sse_transport = await self._exit_stack.enter_async_context(
                    sse_client(self._sse_url, headers=self._sse_headers)
                )
                read_stream, write_stream = sse_transport

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # 初始化会话
            await self._session.initialize()
            self._connected = True

            # 预加载工具和资源配置
            await self._refresh_capabilities()

        except Exception:
            await self._exit_stack.aclose()
            raise

    async def disconnect(self) -> None:
        """断开与 MCP 服务器的连接"""
        if not self._connected:
            return

        if self._exit_stack:
            await self._exit_stack.aclose()

        self._session = None
        self._exit_stack = None
        self._connected = False
        self._tools = []
        self._resources = []

    async def _refresh_capabilities(self) -> None:
        """刷新服务器能力（工具和资源）"""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        # 获取工具列表
        try:
            tools_response = await self._session.list_tools()
            self._tools = [
                MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
                for tool in tools_response.tools
            ]
        except Exception:
            self._tools = []

        # 获取资源列表
        try:
            resources_response = await self._session.list_resources()
            self._resources = [
                MCPResource(
                    uri=str(resource.uri),
                    name=resource.name,
                    description=resource.description,
                    mime_type=resource.mimeType,
                    size=resource.size,
                )
                for resource in resources_response.resources
            ]
        except Exception:
            self._resources = []

    @property
    def is_connected(self) -> bool:
        """是否已连接到服务器"""
        return self._connected

    @property
    def tools(self) -> list[MCPTool]:
        """获取已发现的工具列表"""
        return self._tools.copy()

    @property
    def resources(self) -> list[MCPResource]:
        """获取已发现的资源列表"""
        return self._resources.copy()

    async def list_tools(self) -> list[MCPTool]:
        """
        刷新并返回工具列表

        Returns:
            工具列表
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        await self._refresh_capabilities()
        return self._tools

    async def list_resources(self) -> list[MCPResource]:
        """
        刷新并返回资源列表

        Returns:
            资源列表
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        await self._refresh_capabilities()
        return self._resources

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        调用工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具调用结果
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        result = await self._session.call_tool(name, arguments)
        
        # 转换为可序列化的格式
        content_items = []
        for item in result.content:
            if isinstance(item, types.TextContent):
                content_items.append({"type": "text", "text": item.text})
            elif isinstance(item, types.ImageContent):
                content_items.append({
                    "type": "image",
                    "data": item.data,
                    "mimeType": item.mimeType,
                })
            elif isinstance(item, types.EmbeddedResource):
                resource_dict: dict[str, Any] = {"type": "resource"}
                if isinstance(item.resource, types.TextResourceContents):
                    resource_dict["resource"] = {
                        "uri": item.resource.uri,
                        "text": item.resource.text,
                    }
                    if item.resource.mimeType:
                        resource_dict["resource"]["mimeType"] = item.resource.mimeType
                elif isinstance(item.resource, types.BlobResourceContents):
                    resource_dict["resource"] = {
                        "uri": item.resource.uri,
                        "blob": item.resource.blob,
                    }
                    if item.resource.mimeType:
                        resource_dict["resource"]["mimeType"] = item.resource.mimeType
                content_items.append(resource_dict)

        return {
            "content": content_items,
            "isError": result.isError,
        }

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        读取资源内容

        Args:
            uri: 资源 URI

        Returns:
            资源内容
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        result = await self._session.read_resource(AnyUrl(uri))
        
        contents = []
        for item in result.contents:
            if isinstance(item, types.TextResourceContents):
                content = {
                    "uri": item.uri,
                    "text": item.text,
                }
                if item.mimeType:
                    content["mimeType"] = item.mimeType
                contents.append(content)
            elif isinstance(item, types.BlobResourceContents):
                content = {
                    "uri": item.uri,
                    "blob": item.blob,
                }
                if item.mimeType:
                    content["mimeType"] = item.mimeType
                contents.append(content)

        return {"contents": contents}


class MCPClientPool:
    """
    MCP 客户端池

    管理多个 MCP 客户端连接，聚合所有服务器的工具和资源。
    """

    _clients: dict[str, MCPClient]
    _tool_to_client: dict[str, str]
    _resource_to_client: dict[str, str]

    def __init__(self):
        self._clients = {}
        self._tool_to_client = {}
        self._resource_to_client = {}

    def add_client(self, name: str, client: MCPClient) -> None:
        """
        添加客户端到池

        Args:
            name: 客户端名称（唯一标识）
            client: MCPClient 实例
        """
        self._clients[name] = client

    def remove_client(self, name: str) -> None:
        """移除客户端"""
        if name in self._clients:
            del self._clients[name]
        # 清理映射关系
        self._tool_to_client = {
            k: v for k, v in self._tool_to_client.items() if v != name
        }
        self._resource_to_client = {
            k: v for k, v in self._resource_to_client.items() if v != name
        }

    async def connect_all(self) -> None:
        """连接所有客户端"""
        for name, client in self._clients.items():
            try:
                await client.connect()
                # 建立工具到客户端的映射
                for tool in client.tools:
                    self._tool_to_client[tool.name] = name
                # 建立资源到客户端的映射
                for resource in client.resources:
                    self._resource_to_client[resource.uri] = name
            except Exception as e:
                print(f"Failed to connect to MCP server '{name}': {e}")

    async def disconnect_all(self) -> None:
        """断开所有客户端"""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:
                pass
        self._tool_to_client.clear()
        self._resource_to_client.clear()

    @property
    def all_tools(self) -> list[MCPTool]:
        """获取所有客户端的工具"""
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools

    @property
    def all_resources(self) -> list[MCPResource]:
        """获取所有客户端的资源"""
        resources = []
        for client in self._clients.values():
            resources.extend(client.resources)
        return resources

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        调用工具（自动路由到正确的客户端）

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具调用结果
        """
        client_name = self._tool_to_client.get(name)
        if not client_name:
            raise ValueError(f"Unknown tool: {name}")

        client = self._clients.get(client_name)
        if not client:
            raise RuntimeError(f"Client '{client_name}' not available")

        return await client.call_tool(name, arguments)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        读取资源（自动路由到正确的客户端）

        Args:
            uri: 资源 URI

        Returns:
            资源内容
        """
        client_name = self._resource_to_client.get(uri)
        if not client_name:
            raise ValueError(f"Unknown resource: {uri}")

        client = self._clients.get(client_name)
        if not client:
            raise RuntimeError(f"Client '{client_name}' not available")

        return await client.read_resource(uri)
