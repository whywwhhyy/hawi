# MCP 兼容资源系统

Hawi 提供 MCP（Model Context Protocol）兼容的资源系统，允许 Agent 访问外部数据源。

## 概述

资源是 Agent 可以读取的数据单元，通过 URI 唯一标识。支持文本和二进制内容。

## 核心协议

### HawiResource 协议

```python
from hawi.plugin.resource import HawiResource, ResourceContent

# 检查对象是否实现了资源协议
if isinstance(obj, HawiResource):
    content = obj.read()
```

**必需属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `uri` | `str` | 唯一标识符（如 `file:///prompts/system.txt`） |
| `name` | `str` | 显示名称 |
| `description` | `str \| None` | 可选描述 |
| `mime_type` | `str \| None` | MIME 类型 |
| `size` | `int \| None` | 大小（字节） |

**必需方法：**

```python
def read(self) -> ResourceContent:
    """读取并返回资源内容"""
```

## 资源内容

### ResourceContent 类

```python
from hawi.plugin.resource import ResourceContent

# 文本内容
content = ResourceContent(
    uri="file:///example.txt",
    text="Hello, World!",
    mime_type="text/plain"
)

# 二进制内容
content = ResourceContent(
    uri="file:///image.png",
    blob=b"\x89PNG...",
    mime_type="image/png"
)
```

**属性和方法：**

| 属性/方法 | 说明 |
|-----------|------|
| `is_text` | 是否为文本内容 |
| `is_binary` | 是否为二进制内容 |
| `get_text()` | 获取文本内容（二进制会抛异常） |
| `get_bytes()` | 获取字节内容 |
| `to_dict()` | 转换为 MCP 兼容字典 |

```python
# 转换为 MCP 格式
mcp_dict = content.to_dict()
# {'uri': 'file:///example.txt', 'mimeType': 'text/plain', 'text': 'Hello, World!'}
```

## 内置实现

### HawiLiteralResource

内存中的文本资源：

```python
from hawi.plugin.resource import HawiLiteralResource

resource = HawiLiteralResource(
    uri="literal://greeting",
    name="Greeting",
    text="Hello, Agent!",
    description="简单的问候语"
)
```

### HawiFileResource

文件系统资源，支持懒加载：

```python
from hawi.plugin.resource import HawiFileResource

resource = HawiFileResource(
    uri="file:///prompts/system.txt",
    filepath="/path/to/system.txt",
    description="系统提示词文件"
)

# 读取内容
content = resource.read()
print(content.get_text())
```

**特性：**
- 懒加载：文件内容只在 `read()` 时读取
- 自动 MIME 类型检测

### HawiDynamicResource

动态生成的内容：

```python
from hawi.plugin.resource import HawiDynamicResource

def generate_content():
    return ResourceContent(
        uri="dynamic://current_time",
        text=f"Current time: {datetime.now()}",
        mime_type="text/plain"
    )

resource = HawiDynamicResource(
    uri="dynamic://current_time",
    name="Current Time",
    description="动态生成的时间",
    content_generator=generate_content
)
```

## 在插件中使用

```python
from hawi.plugin import HawiPlugin
from hawi.plugin.resource import HawiLiteralResource

class MyPlugin(HawiPlugin):
    @property
    def resources(self):
        return [
            HawiLiteralResource(
                uri="config://settings",
                name="Settings",
                text="...",
            )
        ]
```

## URI 方案约定

| 方案 | 说明 | 示例 |
|------|------|------|
| `file://` | 本地文件 | `file:///path/to/file.txt` |
| `literal://` | 内联文本 | `literal://greeting` |
| `dynamic://` | 动态生成 | `dynamic://current_time` |
| `resource://` | 资源服务器 | `resource://api/data` |
