"""Hawi Agent - Main entry point

使用 Hawi 框架实现的 Agent 主程序
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any,Literal,cast
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
from hawi.agent.printers import create_printer
from hawi.models import Model
from hawi.models import get_model_class
from hawi.utils.terminal import user_select

from hawi_plugins.python_interpreter import PythonInterpreterPlugin
from hawi_plugins.skills_plugin import SkillsPlugin


def load_apikey_yaml() -> list[dict[str, Any]]:
    """Load apikey.yaml from project root if it exists."""
    project_root = Path(__file__).parent
    apikey_path = project_root / "apikey.yaml"

    if apikey_path.exists():
        with open(apikey_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    return []

QUICK_ARGUMENTS = []
def create_model(argv:list[str]):
    def take_item(items, name:str):
        def select_from_argv_or_user(keys):
            for key in keys:
                if str(key) in argv:
                    argv.remove(key)
                    break
            else:
                key = user_select(keys, f"Select {name}:")
                if key is None:
                    print("")
                    exit()
            QUICK_ARGUMENTS.append(key)
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

    provider_config = take_item(load_apikey_yaml(), "provider")

    apikey = take_item(provider_config['apikey'], 'apikey')
    model_config = take_item(provider_config['model'], 'model')
    adapter = take_item(model_config['adapter'], 'adapter')
    model_params = {'api_key': apikey}
    for key in model_config.keys():
        if key in ('key', 'adapter'):
            continue
        model_params[key] = take_item(model_config[key], key)
    model_class = get_model_class(adapter)
    if not model_class:
        raise Exception(f"unknown model adapter {adapter}")
    
    print(f"quick arguments: {' '.join(QUICK_ARGUMENTS)}")
    return provider_config['key'], model_class(**model_params)

def create_agent(model: Model, event_dump_file: str | None = None, streaming: bool = True) -> HawiAgent:
    """Create a HawiAgent with the specified provider."""
    # print(model.get_balance())

    return HawiAgent(
        model=model,
        plugins=[
            PythonInterpreterPlugin(work_dir=".python_vm", print_execution=False),
            SkillsPlugin(skills_dir=".skills"),
        ],
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
        max_iterations=None,
        event_dump_file=event_dump_file,
        streaming=streaming,
    )


def main():
    argv = sys.argv[1:]

    # Parse arguments
    printer_type = "auto"  # auto, rich, text
    loop = False
    event_dump_file = None
    streaming = True  # Default to streaming mode

    # Parse printer type
    if "--printer" in argv:
        idx = argv.index("--printer")
        argv.pop(idx)
        if idx < len(argv):
            printer_type = argv.pop(idx)

    if "--continue" in argv:
        argv.remove("--continue")
        loop = True

    # Parse streaming flag
    if "--no-streaming" in argv:
        argv.remove("--no-streaming")
        streaming = False
    elif "--streaming" in argv:
        argv.remove("--streaming")
        streaming = True

    # Parse event dump file
    if "--dump-events" in argv:
        idx = argv.index("--dump-events")
        argv.pop(idx)
        if idx < len(argv):
            event_dump_file = argv.pop(idx)
        else:
            # Default dump file with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_dump_file = f".dumps/events_{timestamp}.jsonl"

    # Create agent
    llm_provider, model = create_model(argv)

    agent = create_agent(model, event_dump_file=event_dump_file, streaming=streaming)
    print(f"Using provider: {llm_provider}")
    print(f"Model: {model.model_id}")
    if event_dump_file:
        print(f"Event dump: {event_dump_file}")
    print("Type 'exit', 'quit', or 'q' to exit\n")

    # Setup printer for output display
    # Printer handles events from both streaming and non-streaming modes
    # Note: create_printer automatically selects appropriate printer based on environment
    assert printer_type in ['auto','rich','block','plain']
    printer = create_printer(cast(Literal['auto','rich','block','plain'], printer_type), streaming=streaming)
    print(f"Printer: {type(printer).__name__}", file=sys.stderr)

    agent.subscribe(printer.handle)

    # Execute prompt if provided
    def execute_prompt(prompt:str):
        import asyncio

        try:
            asyncio.run(agent.arun(prompt))
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if event_dump_file:
                print(f"📄 Event dump available at: {event_dump_file}")
            raise

    if argv:
        # Use streaming mode with StreamingPrinter
        for prompt in argv:
            execute_prompt(prompt)
        if not loop:
            return

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue
            if prompt.startswith('/'):
                commands = {
                    'model': (
                        "show model arguments",
                        lambda: print(' '.join(QUICK_ARGUMENTS))
                    ),
                    'clear': (
                        "clear conversation",
                        lambda: agent.context.clear()
                    ),
                    'save_markdown': (
                        "save conversation as markdown",
                        lambda: agent.context.save("history.md", format='markdown')
                    ),
                    'save': (
                        "save conversation",
                        lambda: agent.context.save("session.md", format='json')
                    ),
                    'load': (
                        "load conversation",
                        lambda: agent.context.load("session.md")
                    ),
                    'exit': (
                        "quit agent cli",
                        lambda: exit()
                    ),
                    'quit': 'exit',
                    'q': 'exit',
                }
                def print_help():
                    aliases = {}
                    for k,v in commands.items():
                        if isinstance(v,str):
                            aliases.setdefault(v, []).append(k)
                        else:
                            aliases.setdefault(k, []).append(k)
                    for k,v in aliases.items():
                        desc,_ = commands[k]
                        k = ','.join(v)
                        print(f"{k}: {desc}")
                cmd = prompt.lower().lstrip('/')
                if cmd in ('help','h'):
                    print_help()
                    continue
                command = commands.get(cmd)
                if command:
                    if isinstance(command,str):
                        command = commands[command]
                    _,func = command
                    func()
                    continue
                print("unknown '/' command")
                print_help()
                continue
            if prompt.lower() in ['exit', 'quit', 'q']:
                break

            try:
                execute_prompt(prompt)
            except Exception as e:
                import traceback
                traceback.print_exception(e)
                pass

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
