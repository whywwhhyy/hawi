# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

**Hawi** is an AI Agent framework with model compatibility layers for multiple LLM providers. It provides persistent Python interpreter execution, a plugin-based tool system, a unified event system, and an agent runner for queue-driven execution.

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
```

**Running the Application:**
```bash
uv run python main.py [provider] [prompt]
```

## Architecture

### Layer Structure (Dependency Direction: Top → Bottom)

```
agent          # Execution layer - LLM interaction, tool loops, plugin hooks
    ↓
plugin         # Plugin system - HawiPlugin base, hooks, decorators, resources
    ↓
tool           # Tool layer - core abstractions and registries
    ↓
models         # Model layer - unified LLM adapter interface
    ↓
events         # Event system - EventBus, Model/Agent/AgentRunner events
    ↓
utils          # Infrastructure layer - lifecycle, loader, terminal UI
```

**Key Principle:** Single-direction dependencies. No cycles allowed.

### Core Components

#### Model Adapters (`hawi/models/`)

Unified interface for multiple LLM providers:

- **`Model`** (ABC): Base class defining `invoke()`, `stream()`, `ainvoke()`, `astream()`
- **`OpenAIModel`**, **`AnthropicModel`**, **`StrandsModel`**: Provider-specific protocol adapters
- **`DeepSeekModel`**, **`KimiModel`**, **`MiniMaxModel`**: High-level model entries resolved via `model_registry` and `models.yaml`

**Usage:**
```python
from hawi.models import model_registry

# Create from registry (requires models.yaml)
model = model_registry.create_model("deepseek-chat")

# Or instantiate directly
from hawi.models import DeepSeekModel
model = DeepSeekModel(model_id="deepseek-chat", api_key="...")
```

#### Python Interpreter (`hawi/builtin_plugins/python_interpreter/`)

Persistent subprocess Python execution with state management.

```python
from hawi.builtin_plugins.python_interpreter import PythonInterpreterPlugin

interpreter = PythonInterpreterPlugin()
agent = HawiAgent(model=model, plugins=[interpreter])
```

#### Tool System (`hawi/tool/`)

- **`AgentTool`** (ABC): Base class for tools. Subclasses implement `name`, `description`, `parameters_schema`, and `run()` / `arun()`
- **`@tool`**: Function-based decorator to create `AgentTool` instances from plain functions
- **`ToolRegistry`**: Singleton registry with two layers (base factory layer + agent tool layer)

**Class-based Tool:**
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

#### Plugin System (`hawi/plugin/`)

Plugins provide tools, hooks, and MCP-compatible resources.

- **`HawiPlugin`**: Base class. Auto-discovers methods decorated with `@tool`, `@before_session`, `@after_session`, etc.
- **`PluginManager`**: Manages plugin lifecycle, tool definitions, and hook dispatch
- **Resources**: `HawiResource`, `HawiLiteralResource`, `HawiFileResource`, `HawiDynamicResource`

#### Event System (`hawi/events/`)

Read-only, non-blocking event streaming for observability.

```python
from hawi.events import EventBus

# Subscribe to events
EventBus().subscribe(callback, event_types=["agent.tool_call"])
```

Event types: `Model*Event`, `Agent*Event`, `AgentRunner*Event`.

**Event vs Hook:**
- **Events**: Non-blocking, read-only, multi-consumer (for logging, UI updates)
- **Hooks**: Blocking, mutable, single-consumer (for intervention). Defined via plugin decorators.

#### Agent Context (`hawi/agent/context.py`)

Conversation state management.

```python
from hawi.agent import AgentContext

context = AgentContext(
    messages=[],
    tool_definitions=defs,
    system_prompt=[{"type": "text", "text": "You are a helpful assistant."}]
)
```

#### Agent Execution (`hawi/agent/`)

`HawiAgent` keeps the public API and the main model/tool loop in `agent.py`,
while operational concerns live in explicit components instead of mixins:

- **`AgentRuntime`** (`runtime.py`): interrupt state, steer inputs, runtime snapshots, and context-length retry helpers.
- **`AgentCompactor`** (`compaction.py`): explicit and automatic context compaction.
- **`AgentEvents`** (`eventing.py`): event subscription and fan-out to the agent bus / override bus.
- **`HookDispatcher`** (`hook_dispatcher.py`): plugin hook invocation while preserving the owning agent as the first hook argument.
- **`ToolExecutor`** (`tool_executor.py`): tool argument preparation, runtime injection, audits, and execution.

`HawiAgent` exposes compatibility facade methods such as `_emit_event()`,
`_invoke_*()`, `snapshot_runtime()`, and `steer()`, so tests and integrations can
keep using the established entry points while the implementation remains
componentized.

#### SubAgents (`hawi/agent/subagent/`)

Core sub-agent orchestration lives in a package:

- **`manager.py`**: `SubAgentManager` lifecycle, runner wiring, event forwarding.
- **`types.py`**: `SubAgentSpec`, `SubAgentHandle`, `SubAgentStatus`, limits, plugin policy.
- **`prompts.py`**: built-in role system prompts.
- **`utils.py`**: fork-context cleanup and event/content preview helpers.

`from hawi.agent.subagent import SubAgentManager, SubAgentSpec` remains the
public import style.

#### Agent Runner (`hawi/agent/runner/`)

Message queue management and agent orchestration.

- **`AgentRunner`**: Single-agent queue runner
- **`AgentExecutor`**: Executes agents with queue support
- **`EventInterceptor`**: Intercepts and routes events

### Project Structure

```
hawi/
├── agent/              # Execution layer
│   ├── agent.py        # HawiAgent public API + main model/tool loop
│   ├── runtime.py      # interrupt, steer, runtime snapshots
│   ├── compaction.py   # explicit/automatic context compaction
│   ├── eventing.py     # EventBus facade and event fan-out
│   ├── hook_dispatcher.py # plugin hook invocation
│   ├── tool_executor.py # tool preparation, audit, execution
│   ├── config.py       # model error policy, auto-compact config
│   ├── state.py        # execution/steer runtime dataclasses
│   ├── content_utils.py # content rendering helpers
│   ├── context_retry.py # context-length retry truncation helpers
│   ├── context.py      # AgentContext, ToolCallContext
│   ├── result.py       # AgentRunResult, ToolCallRecord
│   ├── subagent/       # SubAgentManager, specs, status, role prompts
│   ├── runner/      # AgentRunner, AgentExecutor, queues
│   ├── printers/       # Output printers (plain, rich)
│   └── stream_accumulator.py
├── models/             # Model layer
│   ├── model.py        # Model ABC
│   ├── message.py      # Message types
│   ├── registry.py     # ModelRegistry
│   ├── openai/         # OpenAI adapter
│   ├── anthropic/      # Anthropic adapter
│   ├── deepseek/       # DeepSeek adapter
│   ├── kimi/           # Kimi adapter
│   ├── minimax/        # MiniMax adapter
│   └── strands/        # Strands adapter
├── plugin/             # Plugin system
│   ├── plugin.py       # HawiPlugin base
│   ├── manager.py      # PluginManager
│   ├── decorators.py   # @tool, @before_session, etc.
│   ├── hook_context.py # HookContext, HookResult
│   ├── resource/       # MCP-compatible resources
│   └── types.py        # Hook types
├── tool/               # Tool system
│   ├── types.py        # AgentTool, ToolResult
│   ├── registry.py     # ToolRegistry
│   └── function_tool.py
├── events/             # Event system
│   ├── event.py
│   ├── event_bus.py
│   ├── agent_events.py
│   ├── model_events.py
│   └── runner_events.py
├── errors/             # Error types
│   ├── agent_errors.py
│   ├── model_errors.py
│   └── error.py
└── utils/              # Infrastructure
    ├── lifecycle.py    # ExitHandler
    ├── loader.py       # ModuleLoader
    ├── markdown_streaming_parser.py
    └── terminal.py

hawi/builtin_plugins/           # Plugin implementations
├── filesystem_plugin/   # File operations (read/write/edit/glob/grep)
├── shell_plugin/       # Shell command execution
├── python_interpreter/ # Persistent Python interpreter
├── mcp_plugin/         # MCP server integration
├── skills_plugin/      # Claude Skills architecture
└── web/                # Web fetch capabilities

test/
├── unit/               # Unit tests
└── integration/        # Integration tests
```

## Quick Start

**Basic Agent:**
```python
from hawi.agent import HawiAgent
from hawi.models import model_registry

model = model_registry.create_model("deepseek-chat")
agent = HawiAgent(model=model)

result = agent.run("Hello, what can you do?")
print(result.text)
```

**With Tools:**
```python
from hawi.tool import tool
from hawi.plugin import HawiPlugin

@tool()
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

class CalculatorPlugin(HawiPlugin):
    def __init__(self):
        self._calc = calculator

agent = HawiAgent(model=model, plugins=[CalculatorPlugin()])
result = agent.run("What is 15 * 23?")
```

**Streaming:**
```python
for event in agent.run("Tell me a story", stream=True):
    if event.type == "model.content_block_delta":
        print(event.metadata["delta"], end="", flush=True)
```

## Import Conventions

**Within a package** - use relative imports:
```python
from .tool import AgentTool
from .types import ToolResult
```

**Across packages** - use absolute imports:
```python
from hawi.tool import AgentTool, ToolRegistry
from hawi.events import EventBus
```

## Testing Patterns

- Tests use pytest fixtures for setup/teardown
- Integration tests require API keys (configured via `apikey.yaml`, gitignored)
