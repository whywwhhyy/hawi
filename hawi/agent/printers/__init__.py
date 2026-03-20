"""
Hawi Printer System

提供统一的事件驱动输出接口，支持多种终端环境。

主要组件：
- RichPrinter: 智能流式 Markdown 渲染器（默认）
- PlainPrinter: 纯文本输出（用于非终端环境）
- BasePrinter: 打印机基类（用于自定义实现）

使用方式：
    from hawi.agent.printers import create_printer
    
    # 自动检测最佳打印机
    printer = create_printer()
    
    # 强制指定模式
    printer = create_printer(streaming=True)   # 强制 streaming 模式
    printer = create_printer(streaming=False)  # 强制 non-streaming 模式

环境变量：
    HAWI_STREAMING=0    # 强制 non-streaming 模式
    HAWI_STREAMING=1    # 强制 streaming 模式
"""

import sys
from typing import Literal

from .base import BasePrinter as BasePrinter
from .plain import PlainPrinter as PlainPrinter
from .rich import RichPrinter as RichPrinter


def create_printer(
    printer_type: Literal['auto', 'rich', 'plain'] = "auto",
    *,
    streaming: bool | None = None,
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
    - auto 模式：
      - 终端环境：使用 RichPrinter（自动选择 streaming/non-streaming）
      - 非终端环境：使用 PlainPrinter（纯文本输出）
    
    Streaming 模式控制：
    - 参数 `streaming` 优先级最高（True/False）
    - 环境变量 `HAWI_STREAMING` 次之（0/1）
    - 自动检测终端能力（默认）
    
    Args:
        printer_type: 打印机类型 ("auto", "rich", "plain")
        streaming: 强制指定 streaming 模式（None=自动, True=streaming, False=non-streaming）
        show_reasoning: 是否显示推理内容
        show_tools: 是否显示工具调用
        show_errors: 是否显示错误
        show_error_stack: 是否显示错误堆栈
        max_arg_length: 工具参数最大显示长度
        max_result_length: 工具结果最大显示长度
        show_full_tool_content: 是否显示完整工具内容
    
    Returns:
        打印机实例
    
    Example:
        # 自动检测
        printer = create_printer()
        
        # 强制 rich，自动检测 streaming
        printer = create_printer("rich")
        
        # 强制 rich，强制 streaming 模式
        printer = create_printer("rich", streaming=True)
        
        # 强制 rich，强制 non-streaming 模式
        printer = create_printer("rich", streaming=False)
        
        # 纯文本模式（非终端环境推荐）
        printer = create_printer("plain")
    """
    is_tty = sys.stdout.isatty()

    # 确定实际使用的打印机类型
    if printer_type == "auto":
        actual_printer = "rich" if is_tty else "plain"
    else:
        actual_printer = printer_type

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
        # RichPrinter 内部处理 streaming 模式
        if streaming is not None:
            common_args["streaming"] = streaming
        return RichPrinter(**common_args)
    elif actual_printer == "plain":
        return PlainPrinter(**common_args)
    else:
        raise ValueError(f"Unknown printer type: {printer_type}")


__all__ = [
    "BasePrinter",
    "PlainPrinter", 
    "RichPrinter",
    "create_printer",
]
