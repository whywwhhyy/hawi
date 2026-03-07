import sys

from typing import Literal

from .base import BasePrinter as BasePrinter
from .plain import PlainPrinter as PlainPrinter
from .block import BlockPrinter as BlockPrinter
from .rich import RichPrinter as RichPrinter
from .markdown import StreamMarkdownPrinter as StreamMarkdownPrinter


def create_printer(
    printer_type: Literal['auto','rich','block','plain'] = "auto",
    *,
    streaming: bool = False,
    show_reasoning: bool = True,
    show_tools: bool = True,
    show_errors: bool = True,
    show_error_stack: bool = True,
    max_arg_length: int = 80,
    max_result_length: int = 200,
    show_full_tool_content: bool = True,
) -> BasePrinter:
    """
    创建打印机实例。

    自动检测环境并选择合适的打印机：
    - auto模式：
      - 终端环境：使用 rich printer
      - 非终端 + streaming：使用 plain printer（逐行输出）
      - 非终端 + 非 streaming：使用 block printer（完整块输出）
    - 显式指定 printer 时：
      - rich + 非终端：回退到 block printer（不支持 Live 更新）
      - 其他：按指定类型创建

    Args:
        printer_type: 打印机类型 ("auto", "rich", "block", "plain", "markdown")
        streaming: 是否为流式模式
        **kwargs: 传递给打印机构造函数的参数

    Returns:
        打印机实例
    """
    is_tty = sys.stdout.isatty()

    # auto 模式：根据环境自动选择
    if printer_type == "auto":
        if is_tty:
            actual_printer = "rich"
        elif streaming:
            # 非终端 + streaming：使用 plain printer 逐行输出
            actual_printer = "plain"
        else:
            # 非终端 + 非 streaming：使用 block printer 完整块输出
            actual_printer = "block"
    else:
        actual_printer = printer_type
        # 显式指定 rich 但非终端：回退到 block（Live 不支持管道）
        if actual_printer == "rich" and not is_tty:
            actual_printer = "block"

    common_args = {
        "show_reasoning": show_reasoning,
        "show_tools": show_tools,
        "show_errors": show_errors,
        "show_error_stack": show_error_stack,
        "max_arg_length": max_arg_length,
        "max_result_length": max_result_length,
        "show_full_tool_content": show_full_tool_content,
    }

    if actual_printer == "rich":
        return RichPrinter(**common_args)
    elif actual_printer == "block":
        return BlockPrinter(**common_args)
    elif actual_printer == "plain":
        return PlainPrinter(**common_args)
    elif actual_printer == "markdown":
        return StreamMarkdownPrinter(**common_args)
    else:
        raise ValueError(f"Unknown printer type: {printer_type}")
