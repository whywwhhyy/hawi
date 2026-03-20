"""Hawi Agent - Main entry point

使用 Hawi 框架实现的 Agent 主程序
"""

import os
import sys
import warnings
from typing import Literal, cast

# Interactive REPL
import readline

# 过滤 Pydantic 警告
warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

from hawi.agent import HawiAgent
from hawi.agent.printers import create_printer
from hawi.models import Model, model_registry
from utils.terminal import user_select


QUICK_ARGUMENTS: list[str] = []

def create_model_from_argv(argv: list[str]) -> tuple[str, Model]:
    """
    从命令行参数或交互式选择创建模型。

    流程：
    1. 从 ~/.hawi/models.yaml 和 ./.hawi/models.yaml 自动加载配置
    2. 选择 factory（从 argv 或交互式）
    3. 使用 factory 创建模型实例

    配置格式示例（models.yaml）：
        factories:
          deepseek-chat:
            class: DeepSeekOpenAIModel
            model_id: deepseek-chat
            base_url: https://api.deepseek.com
            api_key: ${DEEPSEEK_API_KEY}
    """
    # 确保配置已加载（首次调用会自动加载）
    # 注意：后加载的配置会覆盖先加载的，所以项目级配置最后加载
    from pathlib import Path
    user_config = Path.home() / ".hawi" / "models.yaml"
    project_config = Path.cwd() / "models.yaml"

    # 先加载用户级配置
    if user_config.exists():
        model_registry.load_config(user_config, quiet=True)
    # 再加载项目级配置（覆盖用户级）
    # 注意：load_config 会自动设置 _initialized=True，禁用自动加载
    if project_config.exists():
        model_registry.load_config(project_config, quiet=True)

    # 调试：显示已加载的 factories
    if os.environ.get("HAWI_DEBUG"):
        print(f"[DEBUG] Loaded factories: {model_registry.list_factories()}")
        config = model_registry.get_factory("deepseek-chat")
        if config:
            api_key_preview = config.arguments.get("api_key", "N/A")
            if api_key_preview and len(str(api_key_preview)) > 10:
                api_key_preview = str(api_key_preview)[:10] + "..."
            print(f"[DEBUG] deepseek-chat api_key: {api_key_preview}")

    available_factories = model_registry.list_factories()
    if not available_factories:
        raise RuntimeError(
            "No model factories available.\n"
            "Please create ~/.hawi/models.yaml or ./.hawi/models.yaml with factory definitions.\n"
            "Example:\n"
            "  factories:\n"
            "    deepseek-chat:\n"
            "      class: DeepSeekOpenAIModel\n"
            "      model_id: deepseek-chat\n"
            "      base_url: https://api.deepseek.com\n"
            "      api_key: ${DEEPSEEK_API_KEY}"
        )

    # 从 argv 或交互式选择 factory
    factory_name = None
    for arg in argv[:]:
        if arg in available_factories:
            argv.remove(arg)
            factory_name = arg
            break

    if factory_name is None:
        # 交互式选择
        factory_name = user_select(available_factories, "Select model factory:")
        if factory_name is None:
            print("")
            exit()

    QUICK_ARGUMENTS.append(factory_name)

    # 使用 factory 创建模型
    try:
        model = model_registry.create_model(factory_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create model from factory '{factory_name}': {e}\n"
            f"Please check your models.yaml configuration."
        )

    print(f"quick arguments: {' '.join(QUICK_ARGUMENTS)}")
    return factory_name, model

def create_agent(model: Model, event_dump_file: str | None = None, streaming: bool = True) -> HawiAgent:
    """Create a HawiAgent with the specified provider."""
    # print(model.get_balance())
    from hawi_plugins.skills_plugin import SkillsPlugin
    from hawi_plugins.web import WebPlugin

    return HawiAgent(
        model=model,
        plugins=[
            SkillsPlugin(skills_dir=".skills"),
            WebPlugin(),
        ],
        system_prompt="""You are a helpful AI assistant with Skills""",
        max_iterations=None,
        event_dump_file=event_dump_file,
        streaming=streaming,
    )


def main():
    argv = sys.argv[1:]

    # Parse arguments
    printer_type = "auto"  # auto, rich, plain
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

    # Create agent using new config system
    factory_name, model = create_model_from_argv(argv)
    agent = create_agent(model, event_dump_file=event_dump_file, streaming=streaming)
    print(f"Using factory: {factory_name}")
    print(f"Model: {model.model_id}")
    if event_dump_file:
        print(f"Event dump: {event_dump_file}")
    print("Type 'exit', 'quit', or 'q' to exit\n")

    # Setup printer for output display
    # Printer handles events from both streaming and non-streaming modes
    # Note: create_printer automatically selects appropriate printer based on environment
    assert printer_type in ['auto', 'rich', 'plain']
    printer = create_printer(cast(Literal['auto', 'rich', 'plain'], printer_type), streaming=streaming)
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
