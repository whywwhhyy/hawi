"""
MCP (Model Context Protocol) Hawi 插件

此插件允许 Hawi Agent 连接到 MCP 服务器，使用 MCP 提供的工具和资源。

基本用法:
    from hawi.agent import HawiAgent
    from hawi.agent.models.kimi import KimiModel
    from hawi_plugins.mcp_plugin import MCPPlugin

    # 创建插件
    mcp_plugin = MCPPlugin()
    
    # 添加 MCP 服务器（stdio 类型）
    mcp_plugin.add_stdio_server(
        "filesystem",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
    )
    
    # 添加 MCP 服务器（SSE 类型）
    mcp_plugin.add_sse_server(
        "remote",
        "http://localhost:3000/sse"
    )
    
    # 连接到所有服务器
    await mcp_plugin.connect()
    
    # 创建 Agent
    agent = HawiAgent(
        model=KimiModel(),
        plugins=[mcp_plugin],
    )
    
    # 使用 Agent（自动使用 MCP 工具和资源）
    result = await agent.run("List files in /home/user/docs")

高级用法 - 使用客户端池:
    from hawi_plugins.mcp import MCPClient, MCPClientPool
    
    # 创建客户端
    client = MCPClient.from_stdio("python", ["-m", "my_mcp_server"])
    
    # 或使用命令字符串
    client = MCPClient.from_command("npx -y @modelcontextprotocol/server-filesystem /tmp")
    
    async with client:
        tools = await client.list_tools()
        result = await client.call_tool("tool_name", {"arg": "value"})
"""

from .client import MCPClient, MCPClientPool, MCPResource, MCPTool
from .plugin import MCPPlugin

__all__ = [
    # 主要插件类
    "MCPPlugin",
    # 客户端类
    "MCPClient",
    "MCPClientPool",
    # 数据类
    "MCPTool",
    "MCPResource",
]
