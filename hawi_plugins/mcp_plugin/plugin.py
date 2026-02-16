"""
MCP (Model Context Protocol) Hawi 插件

此插件允许 Hawi Agent 连接到 MCP 服务器，使用 MCP 提供的工具和资源。

使用示例:
    # 创建插件并连接到 MCP 服务器
    mcp_plugin = MCPPlugin()
    
    # 添加 stdio 服务器
    mcp_plugin.add_stdio_server("filesystem", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    
    # 添加 SSE 服务器
    mcp_plugin.add_sse_server("remote", "http://localhost:3000/sse")
    
    # 添加到 agent
    agent = HawiAgent(
        model=model,
        plugins=[mcp_plugin],
    )
"""

from __future__ import annotations

import asyncio
from typing import Any

from hawi.tool import AgentTool
from hawi.plugin import HawiPlugin
import hawi.plugin as plugin
from hawi.tool import ToolResult
from hawi.plugin.resource import HawiResource, ResourceContent
from hawi.plugin.resource.implementations import HawiDynamicResource

from .client import MCPClient, MCPClientPool


class MCPPlugin(HawiPlugin):
    """
    MCP (Model Context Protocol) 插件

    将 MCP 服务器的工具和资源集成到 Hawi Agent 中。

    特性:
        - 支持多个 MCP 服务器连接（stdio 和 SSE）
        - 自动发现和使用 MCP 工具
        - 自动发现和使用 MCP 资源
        - 工具调用结果自动转换

    使用示例:
        mcp_plugin = MCPPlugin()
        
        # 添加文件系统 MCP 服务器
        mcp_plugin.add_stdio_server(
            "fs",
            "npx",
            ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
        )
        
        # 连接到所有服务器
        await mcp_plugin.connect()
        
        # 现在 agent 可以使用 MCP 工具和资源了
    """

    def __init__(self):
        super().__init__()
        self._pool = MCPClientPool()
        self._server_configs: list[dict[str, Any]] = []
        self._connected = False
        self._tool_wrappers: dict[str, _MCPToolWrapper] = {}
        self._resource_wrappers: dict[str, _MCPResourceWrapper] = {}

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """
        添加 stdio 类型的 MCP 服务器

        Args:
            name: 服务器名称（唯一标识）
            command: 命令（如 "python", "npx", "uvx" 等）
            args: 命令参数列表
            env: 环境变量字典

        示例:
            # npx 方式
            plugin.add_stdio_server(
                "filesystem",
                "npx",
                ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
            )
            
            # uvx 方式
            plugin.add_stdio_server(
                "sqlite",
                "uvx",
                ["mcp-server-sqlite", "--db-path", "/path/to/db.sqlite"]
            )
            
            # Python 模块方式
            plugin.add_stdio_server(
                "my_server",
                "python",
                ["-m", "my_mcp_server"]
            )
        """
        self._server_configs.append({
            "type": "stdio",
            "name": name,
            "command": command,
            "args": args or [],
            "env": env,
        })

    def add_sse_server(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        添加 SSE 类型的 MCP 服务器

        Args:
            name: 服务器名称（唯一标识）
            url: SSE 服务器 URL
            headers: 请求头字典

        示例:
            plugin.add_sse_server(
                "remote",
                "http://localhost:3000/sse",
                {"Authorization": "Bearer token"}
            )
        """
        self._server_configs.append({
            "type": "sse",
            "name": name,
            "url": url,
            "headers": headers,
        })

    def add_server_from_command(self, name: str, command: str) -> None:
        """
        从命令字符串添加 MCP 服务器

        支持格式:
            - "python -m mcp_server"
            - "npx -y @modelcontextprotocol/server-filesystem /tmp"
            - "uvx mcp-server-sqlite --db-path /path/to/db.sqlite"
            - "http://localhost:3000/sse" (自动识别为 SSE)

        Args:
            name: 服务器名称
            command: 命令字符串
        """
        command = command.strip()
        
        # 检测是否为 URL
        if command.startswith(("http://", "https://")):
            self.add_sse_server(name, command)
        else:
            parts = command.split()
            if parts:
                self.add_stdio_server(name, parts[0], parts[1:])

    async def connect(self) -> None:
        """
        连接到所有配置的 MCP 服务器

        会自动发现所有服务器的工具和资源。
        """
        if self._connected:
            return

        # 创建客户端
        for config in self._server_configs:
            if config["type"] == "stdio":
                client = MCPClient.from_stdio(
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env"),
                )
            else:  # sse
                client = MCPClient.from_sse(
                    url=config["url"],
                    headers=config.get("headers"),
                )
            self._pool.add_client(config["name"], client)

        # 连接所有客户端
        await self._pool.connect_all()
        self._connected = True

        # 创建工具包装器
        for tool in self._pool.all_tools:
            wrapper = _MCPToolWrapper(self._pool, tool)
            self._tool_wrappers[tool.name] = wrapper

        # 创建资源包装器
        for resource in self._pool.all_resources:
            wrapper = _MCPResourceWrapper(self._pool, resource)
            self._resource_wrappers[resource.uri] = wrapper

    async def disconnect(self) -> None:
        """断开所有 MCP 服务器连接"""
        await self._pool.disconnect_all()
        self._connected = False
        self._tool_wrappers.clear()
        self._resource_wrappers.clear()

    def _ensure_connected(self) -> None:
        """确保已连接（同步上下文中使用）"""
        if not self._connected:
            try:
                asyncio.get_running_loop()
                # 在异步上下文中，应该使用 async connect()
                raise RuntimeError(
                    "MCP plugin not connected. "
                    "Please call 'await mcp_plugin.connect()' before using."
                )
            except RuntimeError as e:
                if "no running event loop" in str(e).lower():
                    # 没有事件循环，创建一个临时连接
                    asyncio.run(self.connect())
                else:
                    raise

    @property
    def hooks(self) -> plugin.PluginHooks:
        """生命周期钩子"""
        return {
            "before_session": self._on_before_session,
            "after_session": self._on_after_session,
        }

    def _on_before_session(self, agent) -> None:
        """会话开始前连接 MCP 服务器"""
        try:
            asyncio.get_running_loop()
            # 异步上下文，应该已经手动连接
            pass
        except RuntimeError:
            # 同步上下文，自动连接
            if not self._connected:
                asyncio.run(self.connect())

    def _on_after_session(self, agent) -> None:
        """会话结束后断开连接"""
        if self._connected:
            try:
                asyncio.get_running_loop()
                # 异步上下文，由用户控制
                pass
            except RuntimeError:
                # 同步上下文，自动断开
                asyncio.run(self.disconnect())

    @property
    def tools(self) -> list[AgentTool]:
        """获取所有 MCP 工具（已包装为 Hawi 工具格式）"""
        self._ensure_connected()
        return [wrapper.get_tool() for wrapper in self._tool_wrappers.values()]

    @property
    def resources(self):
        """获取所有 MCP 资源（已包装为 Hawi 资源格式）"""
        self._ensure_connected()
        return [wrapper.get_resource() for wrapper in self._resource_wrappers.values()]

    def get_tool_names(self) -> list[str]:
        """获取所有可用工具名称"""
        self._ensure_connected()
        return list(self._tool_wrappers.keys())

    def get_resource_uris(self) -> list[str]:
        """获取所有可用资源 URI"""
        self._ensure_connected()
        return list(self._resource_wrappers.keys())

    @plugin.tool
    def list_mcp_resources(self) -> str:
        """
        列出所有可用的 MCP 资源

        返回一个格式化的资源列表，包含资源的 URI、名称和描述。
        当你需要了解有哪些 MCP 资源可用时调用此工具。

        Returns:
            格式化的资源列表字符串
        """
        self._ensure_connected()
        
        if not self._resource_wrappers:
            return "暂无可用的 MCP 资源。"
        
        lines = ["可用的 MCP 资源列表:", ""]
        for uri, wrapper in self._resource_wrappers.items():
            resource = wrapper._mcp_resource
            lines.append(f"URI: {resource.uri}")
            lines.append(f"  名称: {resource.name}")
            if resource.description:
                lines.append(f"  描述: {resource.description}")
            if resource.mime_type:
                lines.append(f"  MIME 类型: {resource.mime_type}")
            lines.append("")
        
        return "\n".join(lines)

    @plugin.tool
    def get_mcp_resource(self, uri: str) -> str:
        """
        获取指定 MCP 资源的内容

        通过资源的 URI 读取资源内容。你可以先调用 list_mcp_resources 获取可用的资源列表。

        Args:
            uri: 资源的 URI（如 "file:///path/to/file.txt"）

        Returns:
            资源的文本内容
        """
        self._ensure_connected()
        
        if uri not in self._resource_wrappers:
            available = list(self._resource_wrappers.keys())
            return f"错误: 未找到资源 '{uri}'。\n可用资源: {', '.join(available) if available else '无'}"
        
        try:
            wrapper = self._resource_wrappers[uri]
            resource = wrapper.get_resource()
            content = resource.read()
            
            if content.is_text:
                text = content.get_text()
                # 如果内容太长，提供摘要
                max_length = 10000
                if len(text) > max_length:
                    return text[:max_length] + f"\n\n... (内容已截断，共 {len(text)} 字符)"
                return text
            else:
                return f"[二进制资源，MIME 类型: {content.mime_type or 'unknown'}]"
        except Exception as e:
            return f"读取资源失败: {e}"


class _MCPToolWrapper:
    """MCP 工具包装器 - 将 MCP 工具转换为 Hawi 工具"""

    def __init__(self, pool: MCPClientPool, mcp_tool):
        self._pool = pool
        self._mcp_tool = mcp_tool
        self._hawi_tool = None

    def get_tool(self):
        """获取或创建 Hawi 工具"""
        if self._hawi_tool is None:
            from hawi.tool import tool as create_tool

            async def async_tool_impl(**kwargs):
                return await self._pool.call_tool(self._mcp_tool.name, kwargs)

            def sync_tool_impl(**kwargs):
                try:
                    loop = asyncio.get_running_loop()
                    # 在异步上下文中
                    raise RuntimeError(
                        "MCP tools can only be called synchronously. "
                        "Please use the async interface."
                    )
                except RuntimeError as e:
                    if "no running event loop" in str(e).lower():
                        # 同步上下文
                        return asyncio.run(self._pool.call_tool(self._mcp_tool.name, kwargs))
                    raise

            # 创建工具函数
            tool_func = sync_tool_impl
            tool_func.__name__ = self._mcp_tool.name
            tool_func.__doc__ = self._mcp_tool.description

            # 使用 Hawi 的 tool 装饰器
            self._hawi_tool = create_tool(
                name=self._mcp_tool.name,
                description=self._mcp_tool.description,
                parameters_schema=self._mcp_tool.input_schema,
            )(self._create_tool_wrapper())

        return self._hawi_tool

    def _create_tool_wrapper(self):
        """创建实际的工具包装函数"""
        pool = self._pool
        tool_name = self._mcp_tool.name

        def tool_wrapper(**kwargs) -> ToolResult:
            """MCP 工具包装函数"""
            try:
                # 在异步上下文中运行
                result = asyncio.run(pool.call_tool(tool_name, kwargs))
                
                # 解析结果
                content = result.get("content", [])
                is_error = result.get("isError", False)
                
                # 构建输出文本
                output_parts = []
                for item in content:
                    if item.get("type") == "text":
                        output_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        output_parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
                    elif item.get("type") == "resource":
                        resource = item.get("resource", {})
                        output_parts.append(f"[Resource: {resource.get('uri', 'unknown')}]")
                
                return ToolResult(
                    output="\n".join(output_parts) if output_parts else "",
                    error="Tool execution failed" if is_error else "",
                    success=not is_error,
                )
            except Exception as e:
                return ToolResult(
                    output="",
                    error=str(e),
                    success=False,
                )

        # 设置函数元数据
        tool_wrapper.__name__ = self._mcp_tool.name
        tool_wrapper.__doc__ = self._mcp_tool.description

        return tool_wrapper


class _MCPResourceWrapper:
    """MCP 资源包装器 - 将 MCP 资源转换为 Hawi 资源"""

    def __init__(self, pool: MCPClientPool, mcp_resource):
        self._pool = pool
        self._mcp_resource = mcp_resource
        self._hawi_resource = None

    def get_resource(self) -> HawiResource:
        """获取或创建 Hawi 资源"""
        if self._hawi_resource is None:
            self._hawi_resource = HawiDynamicResource(
                uri=self._mcp_resource.uri,
                name=self._mcp_resource.name,
                description=self._mcp_resource.description or "",
                generator=self._create_generator(),
                mime_type=self._mcp_resource.mime_type,
            )
        return self._hawi_resource

    def _create_generator(self):
        """创建资源内容生成器"""
        pool = self._pool
        uri = self._mcp_resource.uri
        mime_type = self._mcp_resource.mime_type

        def generator() -> ResourceContent:
            """生成资源内容"""
            try:
                result = asyncio.run(pool.read_resource(uri))
                contents = result.get("contents", [])
                
                if not contents:
                    return ResourceContent(
                        uri=uri,
                        text="",
                        mime_type=mime_type,
                    )
                
                # 取第一个内容
                content = contents[0]
                
                if "text" in content:
                    return ResourceContent(
                        uri=uri,
                        text=content["text"],
                        mime_type=content.get("mimeType", mime_type),
                    )
                else:
                    import base64
                    blob = base64.b64decode(content.get("blob", ""))
                    return ResourceContent(
                        uri=uri,
                        blob=blob,
                        mime_type=content.get("mimeType", mime_type),
                    )
            except Exception as e:
                return ResourceContent(
                    uri=uri,
                    text=f"Error reading resource: {e}",
                    mime_type="text/plain",
                )

        return generator
