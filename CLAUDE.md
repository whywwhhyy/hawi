# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hawi** is an AI Agent framework with model compatibility layers for multiple LLM providers (DeepSeek, Kimi/Moonshot). It provides persistent Python interpreter execution, a tool registry with two-layer architecture, and workflow orchestration.

## Development Commands

**Package Management (UV):**
```bash
uv sync                    # Install dependencies
uv add <package>           # Add dependency
uv add --dev <package>     # Add dev dependency
uv run <command>           # Run command in virtualenv
```

**Testing:**
```bash
pytest                     # Run all tests
pytest test/unit/          # Unit tests only
pytest test/integration/   # Integration tests only
pytest test/unit/test_python_interpreter.py::TestPythonInterpreter::test_execute_simple_expression  # Single test
```

**Running the Application:**
```bash
uv run python main.py [provider] [prompt]
uv run python main.py deepseek
uv run python main.py kimi-oai
```

## Architecture

### Layer Structure (Dependency Direction: Top → Bottom)

```
builder        # Configuration-based agent/workflow building (user interface)
    ↓
workflow       # Orchestration layer - flow control and node execution
    ↓
agent          # Execution layer - LLM interaction and model adapters
    ↓
tool           # Tool layer - core abstractions and registries
    ↓
utils          # Infrastructure layer - context, lifecycle, terminal UI
```

**Key Principle:** Single-direction dependencies. No cycles allowed.

### Core Components

#### Model Adapters (`hawi/agent/models/`)

Unified interface for multiple LLM providers with factory pattern:

```
Model (ABC)
├── OpenAIModel
│   ├── DeepSeekOpenAIModel      # DeepSeek via OpenAI API
│   └── KimiOpenAIModel          # Kimi via OpenAI API
├── AnthropicModel
│   ├── DeepSeekAnthropicModel   # DeepSeek via Anthropic API
│   └── KimiAnthropicModel       # Kimi via Anthropic API
└── StrandsModel                 # Adapter for Strands framework
```

**Usage:**
```python
from hawi.agent.models import DeepSeekModel, KimiModel

# Auto-detect API type based on URL
deepseek = DeepSeekModel(model_id="deepseek-chat", api_key="...")

# Force specific API format
kimi = KimiModel(model_id="kimi-k2-5", api="openai")  # or "anthropic"

# Use with agent
agent = HawiAgent(model=deepseek, ...)
```

**DeepSeek API Compatibility:**

| Feature | OpenAI API | Anthropic API |
|---------|-----------|---------------|
| Endpoint | `https://api.deepseek.com` | `https://api.deepseek.com/anthropic` |
| Models | `deepseek-chat`, `deepseek-reasoner` | `deepseek-chat`, `deepseek-reasoner` |
| Images/Documents | Not supported | Not supported |
| `top_k` parameter | Not supported | Not supported |
| `temperature`/`top_p` (reasoner) | Ignored | Ignored |
| `thinking.budget_tokens` | Ignored | Ignored |
| Parallel tool use | Supported | Ignored |

References:
- [DeepSeek OpenAI API](https://api-docs.deepseek.com/guides/openai_api)
- [DeepSeek Anthropic API](https://api-docs.deepseek.com/guides/anthropic_api)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)

**Kimi API Compatibility:**

| Feature | OpenAI API | Anthropic API |
|---------|-----------|---------------|
| Endpoint | `https://api.moonshot.cn/v1` | `https://api.kimi.com/coding` |
| Thinking Mode | `temperature=1.0` | Supported |
| K2.5 Fixed Params | `top_p=0.95, n=1` | - |
| Citations | - | Returned in response |

References:
- [Strands Model Providers](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/)
- [Kimi OpenAI Compatibility](https://platform.moonshot.cn/docs/guide/migrating-from-openai-to-kimi)

#### Python Interpreter (`hawi_plugins/python_interpreter/`)

Persistent subprocess Python execution with state management.

```python
from hawi_plugins.python_interpreter import PythonInterpreter

# As a plugin
interpreter = PythonInterpreter()
agent = HawiAgent(plugins=[interpreter], ...)

# Direct usage
result = interpreter.execute("x = 5\nprint(x * 2)")
print(result.output)  # 10
```

**Features:**
- Length-prefix protocol for subprocess communication
- State persistence between executions
- Pre-imported modules: math, os, sys, json, datetime, time, random, re, collections, itertools, functools, pathlib, typing

#### Tool System (`hawi/tool/`)

Two-layer architecture for tool registration:

```
┌─────────────────────────────────────┐
│         Agent Tool Layer            │
│  (Instance-based, LLM-facing)       │
│  - FunctionAgentTool instances      │
│  - Direct tool execution            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         Base Access Layer           │
│  (Factory-based, infrastructure)    │
│  - Jira, Confluence clients         │
│  - Config-based creation            │
└─────────────────────────────────────┘
```

**Creating a Tool:**
```python
from hawi.tool import AgentTool, ToolResult

class MyTool(AgentTool):
    name = "my_tool"
    description = "Does something useful"
    parameters_schema = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"]
    }

    def run(self, input: str) -> ToolResult:
        return ToolResult(success=True, output={"result": input.upper()})
```

**Function-based Tool:**
```python
from hawi.tool import tool

@tool()
def my_function(input: str) -> dict:
    """Description of what this does."""
    return {"result": input.upper()}
```

#### Context System (`hawi/utils/context.py`)

Thread-safe context with reference counting for cross-thread/coroutine sharing.

```python
from hawi.utils import ContextManager

# Create context
ctx_id = ContextManager.create({"user_id": "123"})

# Attach to current task
ContextManager.attach(ctx_id)

# Access anywhere in the same task
user_id = ContextManager.get("user_id")

# Fork for child tasks
child_id = ContextManager.fork(ctx_id, {"extra": "data"})

# Cleanup
ContextManager.detach(ctx_id)
```

#### Lifecycle Management (`hawi/utils/lifecycle.py`)

Multi-layer cleanup guarantee.

```python
from hawi.utils import ExitHandler

def cleanup():
    print("Cleaning up resources...")

# Register with priority (lower = earlier)
ExitHandler.register(cleanup, priority=10, name="my_cleanup")
```

#### Event System (`hawi/agent/events.py`)

Non-blocking, read-only event streaming for agent execution observability.

```python
from hawi.agent import EventBus

# Subscribe to events
def on_tool_call(event):
    print(f"Tool called: {event.metadata['tool_name']}")

EventBus.subscribe("agent.tool_call", on_tool_call)
```

**Event vs Hook:**
- **Events**: Non-blocking, read-only, multi-consumer (for logging, UI updates)
- **Hooks**: Blocking, mutable, single-consumer (for intervention)

#### Agent Context (`hawi/agent/context.py`)

Conversation state management.

```python
from hawi.agent import AgentContext

context = AgentContext(
    messages=[],
    tools=[my_tool],
    system_prompt=[TextPart(text="You are a helpful assistant.")]
)

# Prepare request for model
request = context.prepare_request()
```

### Project Structure

```
hawi/
├── agent/              # Execution layer
│   ├── agent.py        # HawiAgent main class
│   ├── context.py      # AgentContext
│   ├── events.py       # Event system
│   ├── messages.py     # Message types
│   ├── model.py        # Model ABC
│   ├── result.py       # AgentRunResult
│   └── models/         # Model implementations
│       ├── deepseek/   # DeepSeek adapters
│       ├── kimi/       # Kimi adapters
│       ├── openai/     # OpenAI base
│       └── anthropic/  # Anthropic base
├── tool/               # Tool system
│   ├── types.py        # AgentTool, ToolResult
│   ├── registry.py     # ToolRegistry
│   └── function_tool.py
├── plugin/             # Plugin system
│   ├── plugin.py       # HawiPlugin base
│   ├── types.py        # Hook types
│   └── decorators.py   # @tool, @before_session, etc.
├── resources/          # MCP-compatible resources
├── utils/              # Infrastructure
│   ├── context.py      # ContextManager
│   ├── lifecycle.py    # ExitHandler
│   └── terminal.py     # Terminal UI
└── workflow/           # Workflow orchestration (planned)

hawi_plugins/           # Plugin implementations
└── python_interpreter/
    ├── python_interpreter.py
    └── multi_python_interpreter.py

test/
├── unit/               # Unit tests
└── integration/        # Integration tests
```

## Quick Start

**Basic Agent:**
```python
from hawi.agent import HawiAgent
from hawi.agent.models import DeepSeekModel

model = DeepSeekModel(model_id="deepseek-chat")
agent = HawiAgent(model=model)

result = agent.run("Hello, what can you do?")
print(result.messages[-1]["content"][0]["text"])
```

**With Tools:**
```python
from hawi.tool import tool

@tool()
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

agent = HawiAgent(model=model, plugins=[calculator.to_plugin()])
result = agent.run("What is 15 * 23?")
```

**Streaming:**
```python
for event in agent.run("Tell me a story", stream=True):
    if event.type == "model.content_block_delta":
        print(event.metadata["delta"], end="", flush=True)
```

## References

- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [Kimi Platform](https://platform.moonshot.cn/)
- [Strands Agents](https://strandsagents.com/)
- [Anthropic API](https://docs.anthropic.com/)
- [OpenAI API](https://platform.openai.com/docs/)

### Import Conventions

**Within a package** - use relative imports:
```python
from .tool import AgentTool
from .types import ToolResult
```

**Across packages** - use absolute imports:
```python
from hawi.tool import AgentTool, ToolRegistry
from hawi.utils import ContextManager
```

### Testing Patterns

- Tests use pytest fixtures for setup/teardown
- Context isolation via `ContextManager` for test isolation
- Integration tests require API keys (configured via `apikey.yaml`, gitignored)

**Example:**
```python
import pytest
from hawi.utils import ContextManager

@pytest.fixture
def isolated_context():
    ctx_id = ContextManager.create({"test": True})
    ContextManager.attach(ctx_id)
    yield ctx_id
    ContextManager.detach(ctx_id)

def test_something(isolated_context):
    # Test with isolated context
    pass
```
