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

### Key Components

**Model Adapters** (`hawi/agent/models/`):
- `DeepSeekOpenAIModel` - DeepSeek via OpenAI compatibility API (`https://api.deepseek.com`)
- `DeepSeekAnthropicModel` - DeepSeek via Anthropic compatibility API (`https://api.deepseek.com/anthropic`)
- `KimiOpenAIModel` - Kimi via OpenAI compatibility API
- `KimiAnthropicModel` - Kimi via Anthropic compatibility API

**DeepSeek API Compatibility Notes:**

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
- [Strands Model Providers](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/)
- [Kimi OpenAI Compatibility](https://platform.moonshot.cn/docs/guide/migrating-from-openai-to-kimi)

**Python Interpreter** (`hawi/agent_tools/`):
- `PythonInterpreter` - Persistent subprocess Python execution with state management
- `MultiPythonInterpreter` - Multi-instance Python execution manager
- Uses length-prefix protocol for subprocess communication

**Tool System** (`hawi/tool/`):
- `Tool` / `BaseTool` - Abstract base class for tools
- `ToolRegistry` - Global tool registration (singleton)
- Two-layer architecture: base access tools (Jira, Confluence, etc.) + Agent-facing tools

**Context System** (`hawi/utils/context.py`):
- `ContextManager` - Thread-safe context with reference counting
- Uses `contextvars` for cross-thread/coroutine context sharing
- Supports `create()`, `attach()`, `detach()`, `fork()` for context lifecycle

**Lifecycle Management** (`hawi/utils/lifecycle.py`):
- `ExitHandler` - Multi-layer cleanup guarantee (singleton)
- Use `ExitHandler.register(cleanup_fn, priority)` for resource cleanup

### Import Conventions

**Within a package** - use relative imports:
```python
from .tool import Tool
from .executors import PythonInterpreter
```

**Across packages** - use absolute imports:
```python
from hawi.tool import Tool, ToolRegistry
from hawi.utils import ContextManager
```

### Testing Patterns

- Tests use pytest fixtures for setup/teardown
- Context isolation via `ContextManager` for test isolation
- Integration tests require API keys (configured via `apikey.yaml`, gitignored)
