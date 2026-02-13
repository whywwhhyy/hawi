"""Hawi Agent - Main entry point

使用 Hawi 框架实现的 Agent 主程序
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any
import yaml

# Interactive REPL
import readline

# 过滤 Pydantic 警告
warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)


def _supports_color() -> bool:
    """检测当前终端是否支持 ANSI 颜色。"""
    # 显式禁用颜色
    if os.environ.get("NO_COLOR"):
        return False

    # 不是终端（管道/重定向）
    if not sys.stdout.isatty():
        return False

    # TERM=dumb 表示不支持转义序列
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False

    # Windows 检测
    if sys.platform == "win32":
        # Windows 10+ 支持 ANSI，但需要启用
        # 简化处理：如果没有 FORCE_COLOR，假设不支持
        if not os.environ.get("FORCE_COLOR"):
            return False

    return True

from hawi.agent import HawiAgent
from hawi.agent.events import PlainPrinter, RichStreamingPrinter
from hawi.agent.model import Model
from hawi.agent.models.deepseek import DeepSeekModel
from hawi.agent.models.kimi import KimiModel
from hawi.plugin import HawiPlugin
from hawi.utils.terminal import user_select

from hawi_plugins.python_interpreter import PythonInterpreter, MultiPythonInterpreter


def load_apikey_yaml() -> list[dict[str, Any]]:
    """Load apikey.yaml from project root if it exists."""
    project_root = Path(__file__).parent
    apikey_path = project_root / "apikey.yaml"

    if apikey_path.exists():
        with open(apikey_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    return []

def create_model(argv:list[str]):
    def take_item(name:str, items):
        def select_from_argv_or_user(keys):
            for key in keys:
                if str(key) in argv:
                    argv.remove(key)
                    return key
            key = user_select(keys, f"Select {name}:")
            if key is None:
                print("")
                exit()
            return key

        if not isinstance(items, list):
            return items
        if len(items) == 1:
            return items[0]
        if all(not isinstance(item, dict) for item in items):
            return select_from_argv_or_user(items)
        items_dict = {item["key"]:item for item in items}
        item_key = select_from_argv_or_user(list(items_dict.keys()))
        return items_dict[item_key]
    provider_config = take_item("provider", load_apikey_yaml())

    get_by_key = lambda config, key: take_item(key, config[key])
    apikey = get_by_key(provider_config, 'apikey')
    model_config = get_by_key(provider_config, 'model')
    adapter = get_by_key(model_config, 'adapter')
    base_url = get_by_key(model_config, 'base_url')
    api = get_by_key(model_config, 'api')
    model_id = get_by_key(model_config, 'model_id')
    if adapter == "DeepSeekModel":
        model = DeepSeekModel(
            base_url=base_url,
            api_key=apikey,
            model_id=model_id,
            api=api,
        )
    elif adapter == "KimiModel":
        model = KimiModel(
            base_url=base_url,
            api_key=apikey,
            model_id=model_id,
            api=api,
        )
    else:
        raise Exception("unknown model adapter")
    return provider_config['key'], model

def create_agent(model: Model) -> HawiAgent:
    """Create a HawiAgent with the specified provider."""
    plugin = PythonInterpreter(print_execution=False)
    # print(model.get_balance())

    return HawiAgent(
        model=model,
        plugins=[plugin],
        system_prompt="""You are a helpful AI assistant with Python execution capabilities.

You have access to a persistent Python interpreter through the following tools:
- execute: Run Python code (variables persist between calls)
- install_dependency: Install Python packages
- restart_server: Clear interpreter state
- save_script: Save code to a file
- execute_script: Run a saved script
- list_scripts: See available scripts

Use these tools to help users with coding tasks, data analysis, calculations, etc.
Always explain what you're doing before executing code.
""",
        max_iterations=10,
        enable_streaming=True,   # Enable streaming for real-time output
    )


def main():
    argv = sys.argv[1:]

    # Parse arguments
    printer_type = "auto"  # auto, rich, text

    # Parse printer type
    if "--printer" in argv:
        idx = argv.index("--printer")
        argv.pop(idx)
        if idx < len(argv):
            printer_type = argv.pop(idx)

    # Create agent
    llm_provider, model = create_model(argv)

    # Determine actual printer to use
    use_rich = printer_type == "rich" or (printer_type == "auto" and _supports_color())
    actual_printer = "rich" if use_rich else "plain"

    agent = create_agent(model)
    print(f"Using provider: {llm_provider}")
    print(f"Model: {model.model_id}")
    print(f"Printer: {actual_printer}" + (" (auto-detected)" if printer_type == "auto" else ""))
    print("Type 'exit', 'quit', or 'q' to exit\n")

    # Create printer based on mode
    def create_printer():
        if use_rich:
            return RichStreamingPrinter(
                show_reasoning=True,
                show_tools=True,
                text_style="green",
            )
        else:
            return PlainPrinter(
                show_reasoning=True,
                show_tools=True,
            )

    # Execute prompt if provided
    if argv:
        # Use streaming mode with StreamingPrinter
        import asyncio
        printer = create_printer()

        async def process_events():
            async for event in agent.arun(argv[0], stream=True):
                await printer.handle(event)

        asyncio.run(process_events())
        print()
        return

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue
            if prompt.lower() in ['exit', 'quit', 'q']:
                break

            # Use streaming mode with StreamingPrinter
            import asyncio
            printer = create_printer()

            async def process_events():
                async for event in agent.arun(prompt, stream=True):
                    await printer.handle(event)

            asyncio.run(process_events())
            print()

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
