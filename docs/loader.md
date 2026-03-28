# Module Loader

`ModuleLoader` 是一个通用的动态模块加载器，以目录为单位扫描 Python 模块，通过可组合的条件（Predicate）过滤，再通过提取器（Extractor）拿到你真正需要的对象。

## 支持的模块形式

加载器会识别目录下两种形式的模块：

**包（Package）**：子目录包含 `__init__.py`
```
plugins/
└── my_tool/
    ├── __init__.py   ← 入口
    └── helpers.py
```

**单文件模块**：以 `.py` 结尾、不以 `_` 开头的文件
```
plugins/
└── web_search.py   ← 直接作为模块加载
```

两种形式在使用上完全一致，加载器统一处理。

---

## 核心概念

### ModulePredicate

```python
ModulePredicate = Callable[[ModuleType], bool]
```

接收一个模块，返回是否通过。多个 predicate 之间是 **AND** 关系，全部通过才会进入提取阶段。

### Extractor

```python
Extractor = Callable[[ModuleType], T | None]
```

从通过过滤的模块中提取你想要的对象。返回 `None` 表示该模块无有效内容，会被自动跳过。

---

## API

### `ModuleLoader`

```python
loader = ModuleLoader(directory: str | Path)

loader.load(
    predicates: list[ModulePredicate] | None = None,
    extractor: Extractor[T] | None = None,
) -> list[T]
```

- `predicates` 为 `None` 时接受所有模块
- `extractor` 为 `None` 时直接返回模块对象本身

### 内置 Predicate 工厂

| 函数 | 说明 |
|------|------|
| `has_subclass(base_class)` | 模块中存在 `base_class` 的具体子类（非抽象） |
| `has_function(name)` | 模块中存在名为 `name` 的可调用对象 |
| `has_attribute(name)` | 模块中存在名为 `name` 的属性 |

### 内置 Extractor 工厂

| 函数 | 说明 |
|------|------|
| `extract_subclass(base_class)` | 提取第一个具体子类（返回类本身，不实例化） |
| `extract_all_subclasses(base_class)` | 提取所有具体子类，返回列表 |
| `extract_function(name)` | 提取指定名称的函数 |

---

## 使用示例

### 加载 HawiPlugin 子类

```python
from hawi.utils.loader import ModuleLoader, has_subclass, extract_subclass
from hawi.plugin import HawiPlugin

loader = ModuleLoader("./plugins")
plugin_classes = loader.load(
    predicates=[has_subclass(HawiPlugin)],
    extractor=extract_subclass(HawiPlugin),
)
plugins = [cls() for cls in plugin_classes]
```

目录结构示例：
```
plugins/
├── my_plugin/
│   └── __init__.py    # 定义 class MyPlugin(HawiPlugin): ...
└── another.py         # 定义 class AnotherPlugin(HawiPlugin): ...
```

### 加载工厂函数

```python
from hawi.utils.loader import ModuleLoader, has_function, extract_function

loader = ModuleLoader("./plugins")
factories = loader.load(
    predicates=[has_function("create_plugin")],
    extractor=extract_function("create_plugin"),
)
plugins = [f() for f in factories]
```

### 多条件组合（AND）

```python
# 要求模块既有 HawiPlugin 子类，又有 setup 函数
plugins = loader.load(
    predicates=[
        has_subclass(HawiPlugin),
        has_function("setup"),
    ],
    extractor=extract_subclass(HawiPlugin),
)
```

### OR 条件

内置 predicate 之间是 AND，OR 逻辑用 lambda 自行组合：

```python
from hawi.utils.loader import has_subclass, has_function

either = lambda m: has_subclass(HawiPlugin)(m) or has_function("create_plugin")(m)

loader.load(predicates=[either], extractor=my_extractor)
```

### 自定义 Predicate

```python
def requires_version(min_version: str):
    """要求模块声明了 __version__ 且不低于指定版本。"""
    from packaging.version import Version
    def check(module) -> bool:
        v = getattr(module, "__version__", None)
        return v is not None and Version(v) >= Version(min_version)
    return check

loader.load(
    predicates=[has_subclass(HawiPlugin), requires_version("1.0")],
    extractor=extract_subclass(HawiPlugin),
)
```

### 自定义 Extractor

```python
def extract_plugin_with_meta(module) -> dict | None:
    """提取插件类及其元数据。"""
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, HawiPlugin) and obj is not HawiPlugin:
            return {
                "cls": obj,
                "version": getattr(module, "__version__", "unknown"),
                "author": getattr(module, "__author__", "unknown"),
            }
    return None

results = loader.load(
    predicates=[has_subclass(HawiPlugin)],
    extractor=extract_plugin_with_meta,
)
```

---

## 行为说明

**加载失败不崩溃**：模块 import 出错时只发出 `UserWarning`，不影响其他模块的加载。

**模块缓存**：已加载的模块会注册到 `sys.modules`，同一路径不会重复 import。缓存 key 格式为 `_hawi_loader_{dir_name}_{module_name}`。

**扫描顺序**：按文件名字母序扫描，保证跨平台一致性。

**跳过私有文件**：以 `_` 开头的 `.py` 文件（如 `_utils.py`）不会被扫描。包目录的 `__init__.py` 本身不受此限制，它是包的入口。

---

## 实现位置

```
hawi/utils/loader.py
```

对外导出路径：

```python
from hawi.utils import (
    ModuleLoader,
    has_subclass,
    has_function,
    has_attribute,
    extract_subclass,
    extract_all_subclasses,
    extract_function,
)
```
