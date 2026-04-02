# 配置系统

Hawi 提供灵活的配置系统，支持 YAML 配置文件、环境变量和程序化配置。

## 概述

配置系统基于 `ModelRegistry` 单例，支持：

- **YAML 配置文件**：定义模型工厂和 API Key
- **环境变量**：敏感信息注入
- **自动加载**：自动查找并加载配置文件
- **配置继承**：工厂配置可以继承其他配置

## 配置文件格式

### 基本结构

```yaml
# 属性模板定义
templates:
  deepseek-apikey:
    api_key: sk-xxxxxxxxxxxxxxxxxxxxxxxx
  
  deepseek-base:
    parent: deepseek-apikey
    class: DeepSeekOpenAIModel
    timeout: 60
    max_retries: 3

# 模型工厂定义
factories:
  # 继承基础配置
  deepseek-chat:
    parent: deepseek-base
    model_id: deepseek-chat
  
  deepseek-reasoner:
    parent: deepseek-base
    model_id: deepseek-reasoner
```

### 占位符语法

配置文件支持环境变量占位符：

```yaml
factories:
  mymodel:
    class: OpenAIModel
    api_key: ${OPENAI_API_KEY}           # 直接引用环境变量
    base_url: ${CUSTOM_URL:http://localhost:8080/v1}  # 带默认值
```

## 配置文件位置

Hawi 会按以下顺序自动加载配置：

1. **用户级配置**：`~/.hawi/models.yaml`
2. **项目级配置**：`./.hawi/models.yaml`

```
~
└── .hawi/
    └── models.yaml          # 用户级配置（全局生效）

./
├── .hawi/
│   └── models.yaml          # 项目级配置（覆盖用户级）
└── your_script.py
```

### 禁用自动加载

```bash
# 通过环境变量禁用
export HAWI_NO_AUTO_LOAD=1
```

```python
# 或在代码中控制
import os
os.environ["HAWI_NO_AUTO_LOAD"] = "1"

from hawi.models import model_registry
# 此时不会自动加载配置
```

## 使用配置

### 自动加载

只需导入即可自动加载配置：

```python
from hawi import HawiAgent
from hawi.models import model_registry

# 如果存在配置文件，会自动加载
model = model_registry.create_model("deepseek-chat")
agent = HawiAgent(model=model)
```

### 手动加载

```python
from hawi.models import model_registry

# 加载特定配置文件
model_registry.load_config("/path/to/custom/models.yaml")

# 现在可以使用其中定义的工厂
model = model_registry.create_model("my-custom-model")
```

### 查看已加载的配置

```python
from hawi.models import model_registry

# 列出所有工厂
factories = model_registry.list_factories()
print(factories)

# 列出所有 Model 类
classes = model_registry.list_model_adapters()
print(classes)

# 检查工厂是否存在
if model_registry.has_factory("deepseek-chat"):
    print("deepseek-chat 配置已加载")
```

## 完整配置示例

### apikey.yaml（传统格式）

```yaml
- name: deepseek
  apikey: sk-xxxxxxxxxxxxxxxxxxxxxxxx

- name: kimi-openai
  apikey: sk-yyyyyyyyyyyyyyyyyyyyyyyy

- name: openai
  apikey: sk-zzzzzzzzzzzzzzzzzzzzzzzz
```

### models.yaml（新格式）

```yaml
templates:
  # ========== DeepSeek ==========
  deepseek-apikey:
    api_key: sk-xxxxxxxxxxxxxxxxxxxxxxxx
  
  deepseek-base:
    parent: deepseek-apikey
    class: DeepSeekOpenAIModel
    timeout: 60
  
  # ========== Kimi ==========
  kimi-config:
    class: KimiOpenAIModel
    api_key: sk-yyyyyyyyyyyyyyyyyyyyyyyy
    temperature: 0.7
    max_tokens: 8192
  
  # ========== OpenAI ==========
  openai-config:
    class: OpenAIModel
    api_key: sk-zzzzzzzzzzzzzzzzzzzzzzzz
    temperature: 0.7

factories:
  deepseek-chat:
    parent: deepseek-base
    model_id: deepseek-chat
  
  deepseek-reasoner:
    parent: deepseek-base
    model_id: deepseek-reasoner
  
  kimi-k2:
    parent: kimi-config
    model_id: kimi-k2-5
  
  gpt-4o:
    parent: openai-config
    model_id: gpt-4o
  
  gpt-4o-mini:
    parent: openai-config
    model_id: gpt-4o-mini
```

## 程序化配置

除了配置文件，也可以直接在代码中注册：

```python
from hawi.models import model_registry, OpenAIModel

# 注册 Model 类（通常已内置，无需手动注册）
model_registry.register_adapter("OpenAIModel", OpenAIModel)

# 注册工厂
model_registry.register_factory(
    name="my-gpt-4",
    class_name="OpenAIModel",
    arguments={
        "model_id": "gpt-4",
        "api_key": "sk-...",
        "temperature": 0.5
    }
)

# 使用
model = model_registry.create_model("my-gpt-4")
```

## 配置继承

工厂配置支持继承，子配置可以覆盖父配置的参数。支持继承 template 或 factory：

```yaml
templates:
  # 基础配置（可被多个 factory 继承）
  openai-base:
    class: OpenAIModel
    api_key: ${OPENAI_API_KEY}
    timeout: 60
    temperature: 0.7

factories:
  # 继承并覆盖
  gpt-4-creative:
    parent: openai-base
    model_id: gpt-4
    temperature: 1.0  # 覆盖父配置的 temperature
  
  gpt-4-conservative:
    parent: openai-base
    model_id: gpt-4
    temperature: 0.2  # 覆盖父配置的 temperature
```

**注意**：
- `class` 和 `arguments` 都可以继承和覆盖
- 子配置的参数完全覆盖父配置的同名参数（不是合并）
- Factory 可以继承 template 或其他 factory
- Template 只能继承其他 template
- 支持多级继承，但请避免循环依赖

## 与 Agent 集成

### 通过名称创建 Agent

```python
from hawi import HawiAgent
from hawi.models import model_registry

# 直接从 registry 创建模型
model = model_registry.create_model("deepseek-chat")
agent = HawiAgent(model=model)

# 或者传入字符串，Agent 会自动解析
agent = HawiAgent(model="deepseek-chat")
```

### 运行时覆盖参数

```python
from hawi.models import model_registry

# 创建模型时覆盖配置参数
model = model_registry.create_model(
    "deepseek-chat",
    temperature=0.9,
    max_tokens=4096,
)
```

## 环境变量控制

| 环境变量 | 说明 |
|----------|------|
| `HAWI_NO_AUTO_LOAD` | 设置为 `1` 禁用自动加载 |
| `HAWI_AUTO_CONFIG` | 设置为 `0` 禁用 `hawi/__init__.py` 中的自动配置 |

## 最佳实践

1. **分离敏感信息**
   ```yaml
   # 将 API Key 放在用户级 templates
   # ~/.hawi/models.yaml
   templates:
     openai-apikey:
       api_key: sk-...
   
   # 项目配置继承并定义工厂
   # ./.hawi/models.yaml
   templates:
     openai-config:
       parent: openai-apikey  # 从用户级配置继承 api_key
       class: OpenAIModel
   
   factories:
     gpt-4o:
       parent: openai-config
       model_id: gpt-4o
   ```

2. **使用继承减少重复**
   ```yaml
   factories:
     base:
       class: OpenAIModel
       timeout: 60
     
     gpt-4o:
       parent: base
       model_id: gpt-4o
     
     gpt-4o-mini:
       parent: base
       model_id: gpt-4o-mini
   ```

3. **版本控制**
   - 将 `./.hawi/models.yaml` 加入版本控制（不含敏感信息）
   - 将 `~/.hawi/models.yaml` 加入 `.gitignore`
