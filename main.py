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
from hawi.models import Model, model_registry
from hawi.config import (
    select_model_id,
    get_template_providers_with_configs,
)
from hawi.utils.terminal import user_select


QUICK_ARGUMENTS: list[str] = []


def create_model(argv: list[str]) -> tuple[str, str, Model]:
    """
    从命令行参数或交互式选择创建模型。

    流程：
    1. 选择模板（从 argv 或交互式）
    2. 选择 provider（如果该模板有多个 provider）
    3. 选择 model_id（从 argv 或交互式）
    4. 验证 api_key 并创建模型实例

    注意：配置已在导入 hawi 时自动加载
    """
    # 1. 选择模板
    available_templates = list(model_registry.list_aliases().keys())
    if not available_templates:
        raise RuntimeError("No model templates available. Check models.yaml configuration.")

    template_name = _select_template(argv, available_templates)
    QUICK_ARGUMENTS.append(template_name)

    # 2. 选择 provider（如果该模板有多个）
    provider_name, api_key = _select_provider(template_name, argv)
    if provider_name:
        QUICK_ARGUMENTS.append(provider_name)

    # 3. 选择 model_id
    model_id = select_model_id(template_name, argv)
    if model_id:
        QUICK_ARGUMENTS.append(model_id)

    # 4. 验证 api_key 并创建模型
    if not api_key:
        # 尝试从环境变量获取
        model_class = model_registry.get_class(template_name)
        class_name = model_class.__name__ if model_class else None
        api_key = _get_api_key_from_env(template_name, class_name)
        if not api_key:
            raise RuntimeError(
                f"No API key found for template '{template_name}'.\n"
                f"Please either:\n"
                f"  1. Add it to apikey.yaml\n"
                f"  2. Set the appropriate environment variable (e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY)"
            )

    params = {"model_id": model_id, "api_key": api_key} if model_id else {"api_key": api_key}
    model = model_registry.create(template_name, params)

    print(f"quick arguments: {' '.join(QUICK_ARGUMENTS)}")
    return template_name, provider_name or "default", model


def _get_api_key_from_env(template_name: str, class_name: str | None) -> str | None:
    """尝试从环境变量获取 API key。"""
    import os

    # 模板名特定的环境变量
    template_env_map = {
        "openai-official": "OPENAI_API_KEY",
        "anthropic-official": "ANTHROPIC_API_KEY",
        "deepseek-openai": "DEEPSEEK_API_KEY",
        "deepseek-anthropic": "DEEPSEEK_API_KEY",
        "kimi-openai": "MOONSHOT_API_KEY",
        "kimi-anthropic": "KIMI_API_KEY",
        "minimax-openai": "MINIMAX_API_KEY",
        "minimax-anthropic": "MINIMAX_API_KEY",
        "glm-openai": "GLM_API_KEY",
        "glm-anthropic": "GLM_API_KEY",
        "stepfun-openai": "STEPFUN_API_KEY",
        "siliconflow-deepseek": "SILICONFLOW_API_KEY",
        "siliconflow-kimi": "SILICONFLOW_API_KEY",
        "siliconflow-openai": "SILICONFLOW_API_KEY",
        "ali-openai": "DASHSCOPE_API_KEY",
    }

    # 先尝试模板名映射
    if template_name in template_env_map:
        env_var = template_env_map[template_name]
        return os.environ.get(env_var)

    # 回退到类名映射
    class_env_map = {
        "OpenAIModel": "OPENAI_API_KEY",
        "AnthropicModel": "ANTHROPIC_API_KEY",
        "DeepSeekModel": "DEEPSEEK_API_KEY",
        "KimiModel": "MOONSHOT_API_KEY",
        "MiniMaxModel": "MINIMAX_API_KEY",
    }

    if class_name in class_env_map:
        return os.environ.get(class_env_map[class_name])

    return None


def _select_template(argv: list[str], templates: list[str]) -> str:
    """从 argv 或交互式选择模板。"""
    # 检查 argv
    for arg in argv[:]:
        if arg in templates:
            argv.remove(arg)
            return arg

    # 交互式选择
    selected = user_select(templates, "Select model template:")
    if selected is None:
        print("")
        exit()
    return selected


def _select_provider(template_name: str, argv: list[str]) -> tuple[str | None, str | None]:
    """
    选择 provider 并返回对应的 api_key。

    如果该 template 只有一个 provider，直接返回。
    如果有多个，让用户选择或从 argv 中匹配。

    Returns:
        (provider_name, api_key) 元组，如果没有可用 provider 返回 (None, None)
    """
    providers_with_configs = get_template_providers_with_configs(template_name)

    if not providers_with_configs:
        # 没有配置 provider，返回 None，让上层尝试环境变量
        return None, None

    if len(providers_with_configs) == 1:
        # 只有一个 provider，直接使用
        provider_name, config = providers_with_configs[0]
        return provider_name, config.get("apikey")

    # 多个 provider，先检查 argv 是否匹配
    for arg in argv[:]:
        for provider_name, config in providers_with_configs:
            if arg == provider_name:
                argv.remove(arg)
                return provider_name, config.get("apikey")

    # 交互式选择
    provider_names = [name for name, _ in providers_with_configs]
    selected = user_select(provider_names, f"Multiple providers for {template_name}, select one:")
    if selected is None:
        print("")
        exit()

    # 找到选中的 provider 配置
    for name, config in providers_with_configs:
        if name == selected:
            return name, config.get("apikey")

    return None, None

def create_agent(model: Model, event_dump_file: str | None = None, streaming: bool = True) -> HawiAgent:
    """Create a HawiAgent with the specified provider."""
    # print(model.get_balance())
    from hawi_plugins.python_interpreter import PythonInterpreterPlugin
    from hawi_plugins.skills_plugin import SkillsPlugin
    from hawi_plugins.web import WebPlugin

    return HawiAgent(
        model=model,
        plugins=[
            #PythonInterpreterPlugin(work_dir=".python_vm", print_execution=False),
            SkillsPlugin(skills_dir=".skills"),
            WebPlugin(),
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

    # Create agent using new config system
    template_name, provider_name, model = create_model(argv)
    agent = create_agent(model, event_dump_file=event_dump_file, streaming=streaming)
    print(f"Using template: {template_name}")
    print(f"Using provider: {provider_name}")
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
