# MCP (Model Context Protocol) Hawi 插件

此插件允许 Hawi Agent 连接到 [MCP (Model Context Protocol)](https://modelcontextprotocol.io) 服务器，使用 MCP 提供的工具和资源。

## 特性

- 🔌 **多种连接方式**：支持 stdio 和 SSE 两种 MCP 服务器连接方式
- 🛠️ **工具集成**：自动将 MCP 工具转换为 Hawi 工具
- 📁 **资源集成**：自动将 MCP 资源转换为 Hawi 资源
- 🔗 **多服务器支持**：可同时连接多个 MCP 服务器
- 🔄 **自动发现**：自动发现服务器提供的工具和资源

## 安装

确保项目已安装 `mcp` 依赖（已在项目的 `pyproject.toml` 中配置）：

```bash
uv sync
```

## 快速开始

### 基础用法

```python
from hawi.agent import HawiAgent
from hawi.models.kimi import KimiModel
from hawi_plugins.mcp_plugin import MCPPlugin

# 创建插件
mcp_plugin = MCPPlugin()

# 添加 MCP 服务器（stdio 类型）
mcp_plugin.add_stdio_server(
    "filesystem",
    "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
)

# 连接到所有服务器
await mcp_plugin.connect()

# 创建 Agent
agent = HawiAgent(
    model=KimiModel(),
    plugins=[mcp_plugin],
)

# 使用 Agent（自动使用 MCP 工具）
result = await agent.run("请列出 /home/user/docs 目录下的文件")
```

### 使用命令字符串添加服务器

```python
# 自动识别为 stdio 或 sse
mcp_plugin.add_server_from_command("fs", "npx -y @modelcontextprotocol/server-filesystem /tmp")
mcp_plugin.add_server_from_command("remote", "http://localhost:3000/sse")
```

## API 参考

### MCPPlugin

主插件类，继承自 `HawiPlugin`。

#### 方法

##### `add_stdio_server(name: str, command: str, args: list[str] | None = None, env: dict[str, str] | None = None)`

添加 stdio 类型的 MCP 服务器。

- `name`: 服务器名称（唯一标识）
- `command`: 命令（如 "python", "npx", "uvx" 等）
- `args`: 命令参数列表
- `env`: 环境变量字典

示例：
```python
# npx 方式
mcp_plugin.add_stdio_server(
    "filesystem",
    "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
)

# uvx 方式
mcp_plugin.add_stdio_server(
    "sqlite",
    "uvx",
    ["mcp-server-sqlite", "--db-path", "/path/to/db.sqlite"]
)
```

##### `add_sse_server(name: str, url: str, headers: dict[str, str] | None = None)`

添加 SSE 类型的 MCP 服务器。

- `name`: 服务器名称
- `url`: SSE 服务器 URL
- `headers`: 请求头字典

示例：
```python
mcp_plugin.add_sse_server(
    "remote",
    "http://localhost:3000/sse",
    {"Authorization": "Bearer token"}
)
```

##### `async connect()`

连接到所有配置的 MCP 服务器，自动发现工具和资源。

##### `async disconnect()`

断开所有 MCP 服务器连接。

##### `get_tool_names() -> list[str]`

获取所有可用工具名称。

##### `get_resource_uris() -> list[str]`

获取所有可用资源 URI。

#### 资源查询工具

MCP 资源默认情况下不会自动被 Agent 使用，插件提供了以下工具供 Agent 查询和获取资源：

##### `list_mcp_resources() -> str`

列出所有可用的 MCP 资源。返回格式化的资源列表，包含 URI、名称、描述等信息。

使用示例：
```python
# Agent 可以调用此工具来发现可用资源
result = agent.run("请列出所有可用的 MCP 资源")
```

##### `get_mcp_resource(uri: str) -> str`

获取指定 MCP 资源的内容。

- `uri`: 资源的 URI（如 "file:///path/to/file.txt"）

使用示例：
```python
# Agent 可以先列出资源，然后获取特定资源的内容
result = agent.run("请读取 file:///home/user/docs/readme.md 的内容")
```

### MCPClient

底层 MCP 客户端，可直接使用。

#### 类方法

##### `from_stdio(command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> MCPClient`

创建 stdio 连接的客户端。

##### `from_sse(url: str, headers: dict[str, str] | None = None) -> MCPClient`

创建 SSE 连接的客户端。

##### `from_command(command: str) -> MCPClient`

从命令字符串创建客户端。

#### 方法

##### `async connect()`

连接到 MCP 服务器。

##### `async disconnect()`

断开连接。

##### `async list_tools() -> list[MCPTool]`

获取工具列表。

##### `async list_resources() -> list[MCPResource]`

获取资源列表。

##### `async call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]`

调用工具。

##### `async read_resource(uri: str) -> dict[str, Any]`

读取资源内容。

### MCPClientPool

管理多个 MCP 客户端连接，自动路由工具和资源请求。

## 常见 MCP 服务器

### 文件系统服务器

```bash
npx -y @modelcontextprotocol/server-filesystem <path>
```

### SQLite 服务器

```bash
uvx mcp-server-sqlite --db-path <path>
```

### PostgreSQL 服务器

```bash
npx -y @modelcontextprotocol/server-postgres postgresql://localhost/mydb
```

### Git 服务器

```bash
uvx mcp-server-git
```

## 完整示例

查看 [examples.py](examples.py) 获取更多使用示例。

运行示例：

```bash
# 安装文件系统 MCP 服务器
npm install -g @modelcontextprotocol/server-filesystem

# 运行示例
python -m hawi_plugins.mcp_plugin.examples
```

## 测试

```bash
pytest hawi_plugins/mcp/test_mcp_plugin.py -v
```

## 参考资料

- [MCP 官方文档](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)
