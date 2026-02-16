"""
MCP 插件使用示例

运行示例:
    # 安装 MCP 服务器依赖
    npm install -g @modelcontextprotocol/server-filesystem
    
    # 运行示例
    python -m hawi_plugins.mcp_plugin.examples
"""

from __future__ import annotations

import asyncio


async def example_basic_usage():
    """基础使用示例"""
    from hawi_plugins.mcp_plugin import MCPPlugin

    # 创建插件
    plugin = MCPPlugin()

    # 添加文件系统 MCP 服务器
    # 这会安装并使用官方的 MCP 文件系统服务器
    plugin.add_stdio_server(
        "filesystem",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    # 连接到所有服务器
    await plugin.connect()

    try:
        # 查看可用工具
        print("Available tools:", plugin.get_tool_names())

        # 查看可用资源
        print("Available resources:", plugin.get_resource_uris())

        # 获取工具实例
        tools = plugin.tools
        print(f"Loaded {len(tools)} tools")

        # 获取资源实例
        resources = plugin.resources
        print(f"Loaded {len(resources)} resources")

    finally:
        # 断开连接
        await plugin.disconnect()


async def example_with_agent():
    """与 Hawi Agent 集成示例"""
    from hawi.agent import HawiAgent
    from hawi.agent.models.kimi import KimiModel
    from hawi_plugins.mcp_plugin import MCPPlugin

    # 创建 MCP 插件
    mcp_plugin = MCPPlugin()

    # 添加文件系统服务器
    mcp_plugin.add_stdio_server(
        "fs",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    # 连接到服务器
    await mcp_plugin.connect()

    try:
        # 创建 Agent
        agent = HawiAgent(
            model=KimiModel(),
            plugins=[mcp_plugin],
        )

        # 使用 Agent（会自动使用 MCP 工具）
        result = await agent.arun("请列出 /tmp 目录下的文件")
        print(result)

    finally:
        await mcp_plugin.disconnect()


async def example_multiple_servers():
    """多服务器连接示例"""
    from hawi_plugins.mcp_plugin import MCPPlugin

    plugin = MCPPlugin()

    # 添加多个服务器
    plugin.add_stdio_server(
        "filesystem",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    )

    plugin.add_stdio_server(
        "sqlite",
        "uvx",
        ["mcp-server-sqlite", "--db-path", "/path/to/db.sqlite"]
    )

    plugin.add_sse_server(
        "remote",
        "http://localhost:3000/sse"
    )

    await plugin.connect()

    try:
        print("All tools:", plugin.get_tool_names())
        print("All resources:", plugin.get_resource_uris())
    finally:
        await plugin.disconnect()


async def example_direct_client():
    """直接使用 MCPClient 示例"""
    from hawi_plugins.mcp_plugin import MCPClient

    # 创建客户端
    client = MCPClient.from_stdio(
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    async with client:
        # 列出工具
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # 列出资源
        resources = await client.list_resources()
        print(f"\nFound {len(resources)} resources:")
        for resource in resources:
            print(f"  - {resource.name}: {resource.uri}")

        # 调用工具（示例：列出目录）
        # 注意：实际工具名称和参数取决于服务器实现
        try:
            result = await client.call_tool("list_directory", {
                "path": "/tmp"
            })
            print("\nTool result:", result)
        except Exception as e:
            print(f"\nTool call error: {e}")


async def example_client_pool():
    """使用 MCPClientPool 管理多个连接"""
    from hawi_plugins.mcp_plugin import MCPClient, MCPClientPool

    pool = MCPClientPool()

    # 创建并添加客户端
    fs_client = MCPClient.from_stdio(
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    pool.add_client("filesystem", fs_client)

    # 连接所有客户端
    await pool.connect_all()

    try:
        # 获取所有工具
        all_tools = pool.all_tools
        print(f"Total tools: {len(all_tools)}")

        # 获取所有资源
        all_resources = pool.all_resources
        print(f"Total resources: {len(all_resources)}")

        # 调用工具（自动路由到正确的客户端）
        # result = await pool.call_tool("some_tool", {"arg": "value"})

    finally:
        await pool.disconnect_all()


async def example_resource_tools():
    """使用资源查询工具示例"""
    from hawi_plugins.mcp_plugin import MCPPlugin

    plugin = MCPPlugin()

    # 添加文件系统服务器
    plugin.add_stdio_server(
        "fs",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    await plugin.connect()

    try:
        # 使用 list_mcp_resources 工具列出资源
        print("=== 列出 MCP 资源 ===")
        resources_list = plugin.list_mcp_resources()
        print(resources_list)

        # 获取资源 URI 列表
        uris = plugin.get_resource_uris()
        if uris:
            # 使用 get_mcp_resource 工具获取第一个资源的内容
            print(f"\n=== 获取资源内容: {uris[0]} ===")
            content = plugin.get_mcp_resource(uris[0])
            print(content[:500] if len(content) > 500 else content)

    finally:
        await plugin.disconnect()


def main():
    """运行示例"""
    print("=" * 50)
    print("MCP Plugin Examples")
    print("=" * 50)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Direct Client", example_direct_client),
        ("Client Pool", example_client_pool),
        ("Multiple Servers", example_multiple_servers),
        ("Resource Tools", example_resource_tools),
        # ("With Agent", example_with_agent),  # 需要 API key
    ]

    for name, example_func in examples:
        print(f"\n{'=' * 50}")
        print(f"Example: {name}")
        print("=" * 50)
        try:
            asyncio.run(example_func())
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
