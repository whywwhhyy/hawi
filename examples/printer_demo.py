#!/usr/bin/env python3
"""RichPrinter 演示 - 展示代码块样式自定义"""

import asyncio
import sys
from hawi.agent.printers import create_printer, RichPrinter
from hawi.agent.events import (
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
)


def make_delta(text: str, delta_type: str = "text"):
    part = {
        "type": "text_delta" if delta_type == "text" else "thinking_delta",
        "index": 0,
        "delta": text,
        "is_start": False,
        "is_end": False,
    }
    return ModelContentBlockDeltaEvent.create("demo-1", part)


async def demo_code_style():
    """演示代码块样式"""
    print("=" * 60)
    print("代码块样式演示")
    print("=" * 60)
    print()
    
    # 使用自定义代码块样式
    printer = RichPrinter(
        code_theme="dracula",
        code_border_style="blue",
        code_background="#1e1e1e",
        code_show_language=True,
    )
    
    await printer.handle(ModelContentBlockStartEvent.create("demo-1", 0, "text"))
    
    chunks = [
        "# 代码块样式演示\n\n",
        "## Python\n\n",
        "```python\n",
        "def fibonacci(n):\n",
        "    if n <= 1:\n",
        "        return n\n",
        "    return fibonacci(n-1) + fibonacci(n-2)\n",
        "```\n\n",
        "完成！",
    ]
    
    for chunk in chunks:
        await printer.handle(make_delta(chunk))
    
    await printer.handle(ModelContentBlockStopEvent.create("demo-1", 0, [{"type": "text", "text": ""}]))


async def main():
    await demo_code_style()


if __name__ == "__main__":
    asyncio.run(main())
