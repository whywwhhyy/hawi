#!/usr/bin/env python3
"""RichPrinter 演示"""

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
        "type": f"{delta_type}_delta",
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
    
    printer = RichPrinter(
        code_theme="dracula",
        code_border_style="blue",
        code_background="#1e1e1e",
        code_show_language=True,
    )
    
    await printer.handle(ModelContentBlockStartEvent.create("demo-1", 0, "text"))
    
    chunks = [
        "# 代码块演示\n\n",
        "```python\n",
        "def hello():\n",
        "    print('Hello')\n",
        "```\n\n",
        "完成！",
    ]
    
    for chunk in chunks:
        await printer.handle(make_delta(chunk, "text"))
    
    await printer.handle(ModelContentBlockStopEvent.create("demo-1", 0, []))


async def main():
    await demo_code_style()


if __name__ == "__main__":
    asyncio.run(main())
