"""HawiScheduler Demo - Interactive CLI

展示 HawiScheduler 的消息队列、优先级调度和中断功能。

用法:
    uv run python scheduler_demo.py [model_factory] [--demo MODE]

示例:
    uv run python scheduler_demo.py deepseek-chat
    uv run python scheduler_demo.py --demo auto          # 自动演示模式
    uv run python scheduler_demo.py --demo interactive   # 交互式命令模式
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from datetime import datetime
from typing import Literal

# 过滤 Pydantic 警告
warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

from hawi.agent import HawiAgent, HawiScheduler, QueueType, SchedulerState
from hawi.agent.printers import create_printer
from hawi.events import Event
from hawi.models import Model, model_registry
from utils.terminal import user_select


DEMO_QUICK_ARGUMENTS: list[str] = []


def create_model_from_argv(argv: list[str]) -> tuple[str, Model]:
    """从命令行参数或交互式选择创建模型。"""
    from pathlib import Path
    user_config = Path.home() / ".hawi" / "models.yaml"
    project_config = Path.cwd() / "models.yaml"

    if user_config.exists():
        model_registry.load_config(user_config, quiet=True)
    if project_config.exists():
        model_registry.load_config(project_config, quiet=True)

    available_factories = model_registry.list_factories()
    if not available_factories:
        raise RuntimeError(
            "No model factories available.\n"
            "Please create ~/.hawi/models.yaml or ./models.yaml with factory definitions."
        )

    factory_name = None
    for arg in argv[:]:
        if arg in available_factories:
            argv.remove(arg)
            factory_name = arg
            break

    if factory_name is None:
        factory_name = user_select(available_factories, "Select model factory:")
        if factory_name is None:
            print("")
            exit()

    DEMO_QUICK_ARGUMENTS.append(factory_name)
    model = model_registry.create_model(factory_name)
    return factory_name, model


def create_scheduler(model: Model, streaming: bool = True) -> HawiScheduler:
    """创建带 Scheduler 的 Agent。"""
    from hawi_plugins.skills_plugin import SkillsPlugin
    from hawi_plugins.web import WebPlugin

    agent = HawiAgent(
        model=model,
        plugins=[
            SkillsPlugin(skills_dir=".skills"),
            WebPlugin(),
        ],
        system_prompt="""You are a helpful AI assistant with Skills.

When handling multiple tasks:
1. Process each task according to its priority
2. For urgent interruptions, acknowledge the context switch
3. Maintain context across related tasks""",
        max_iterations=None,
        streaming=streaming,
    )

    return HawiScheduler(agent)


class SchedulerDemo:
    """HawiScheduler 交互式演示。"""

    def __init__(self, scheduler: HawiScheduler, demo_mode: str = "interactive"):
        self.scheduler = scheduler
        self.demo_mode = demo_mode
        self.running = False
        self.message_counter = 0
        self.event_log: list[str] = []
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()

        # 订阅事件以显示调度器活动
        self.scheduler.subscribe(self._on_scheduler_event)
        self.scheduler.agent.subscribe(self._on_agent_event)

    def _on_scheduler_event(self, event: Event) -> None:
        """处理调度器事件。"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if event.type.startswith("scheduler."):
            self.event_log.append(f"[{timestamp}] {event.type}")
            # 打印重要事件
            if event.type == "scheduler.enqueue":
                print(f"  📥 消息入队: {event.message_id} ({event.queue_type})")
            elif event.type == "scheduler.interrupt":
                print(f"  ⚡ 执行中断: {event.reason}")

    def _on_agent_event(self, event: Event) -> None:
        """处理 Agent 事件。"""
        if event.type == "agent.run_start":
            print(f"  ▶️  Agent Run 开始: {event.run_id}")
        elif event.type == "agent.run_stop":
            print(f"  ⏹️  Agent Run 结束: {event.stop_reason}")

    def show_status(self) -> None:
        """显示当前队列状态。"""
        lengths = self.scheduler.get_queue_lengths()
        state = self.scheduler.state
        exec_state = self.scheduler._executor.state
        print(f"\n  📊 调度器: {state.name} | 执行器: {exec_state.name}")
        print(f"     队列: 紧急={lengths['urgent']} 高优={lengths['high_prio']} 普通={lengths['normal']}")
        print()

    def enqueue_message(self, content: str, queue: Literal["normal", "high_prio", "urgent"], show_confirm: bool = True) -> str | None:
        """入队一条消息。"""
        try:
            msg_id = self.scheduler.enqueue(content, queue)
            self.message_counter += 1
            if show_confirm:
                emoji = {"normal": "📨", "high_prio": "🔼", "urgent": "🔴"}
                preview = content[:50] + ('...' if len(content) > 50 else '')
                print(f"  {emoji[queue]} [{queue}] {preview}")
            return msg_id
        except Exception as e:
            print(f"  ❌ 入队失败: {e}")
            return None

    async def run_scheduler(self) -> None:
        """在后台运行调度器循环。"""
        print("  🚀 调度器已启动")
        try:
            await self.scheduler.run_forever(poll_interval=0.5)
        except asyncio.CancelledError:
            pass
        finally:
            print("  🛑 调度器已停止")

    def show_help(self) -> None:
        """显示帮助信息。"""
        print("""
  📖 可用命令:

  队列操作:
    n <消息>      - 入队普通消息 (normal)
    h <消息>      - 入队高优先级消息 (high_prio)  
    u <消息>      - 入队紧急消息 (urgent) - 会打断当前执行
    status        - 显示队列状态
    clear [queue] - 清空队列 (normal/high_prio/urgent/all)

  信息展示:
    events        - 显示最近的事件日志
    help          - 显示此帮助
    exit/quit/q   - 退出

  快捷命令:
    /save         - 保存对话历史
    /clear        - 清空对话上下文
    /model        - 显示当前模型
""")

    async def read_input(self) -> None:
        """在后台读取用户输入。"""
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # 使用 run_in_executor 让 input 不会阻塞事件循环
                is_idle = self.scheduler._executor.is_idle
                prompt_str = "scheduler> " if is_idle else "(running)> "
                user_input = await loop.run_in_executor(
                    None, lambda: input(prompt_str)
                )
                await self._input_queue.put(user_input)
            except EOFError:
                await self._input_queue.put("exit")
                break
            except KeyboardInterrupt:
                await self._input_queue.put("exit")
                break
            except Exception as e:
                # 其他错误，打印并继续
                print(f"\n  输入错误: {e}")
                await asyncio.sleep(0.1)

    async def process_commands(self) -> None:
        """处理用户命令。"""
        while self.running:
            try:
                # 等待输入（带超时，以便定期更新状态）
                try:
                    user_input = await asyncio.wait_for(
                        self._input_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if not user_input.strip():
                    continue

                # 退出命令
                if user_input.lower() in ["exit", "quit", "q"]:
                    break

                # 帮助
                if user_input.lower() in ["help", "h", "?"]:
                    self.show_help()
                    continue

                # 队列操作
                if user_input.startswith("n "):
                    self.enqueue_message(user_input[2:], "normal")
                    continue

                if user_input.startswith("h "):
                    self.enqueue_message(user_input[2:], "high_prio")
                    continue

                if user_input.startswith("u "):
                    self.enqueue_message(user_input[2:], "urgent")
                    continue

                if user_input == "status":
                    self.show_status()
                    continue

                if user_input.startswith("clear"):
                    parts = user_input.split()
                    if len(parts) == 1:
                        result = self.scheduler.clear_all_queues()
                        print(f"  🧹 已清空所有队列: {result}")
                    else:
                        queue = parts[1]
                        if queue in ["normal", "high_prio", "urgent"]:
                            count = self.scheduler.clear_queue(queue)
                            print(f"  🧹 已清空 {queue} 队列: {count} 条消息")
                        else:
                            print(f"  ❌ 未知队列类型: {queue}")
                    continue

                # 信息展示
                if user_input == "events":
                    print("\n  📜 最近事件:")
                    for event in self.event_log[-10:]:
                        print(f"     {event}")
                    print()
                    continue

                # 内置命令
                if user_input == "/clear":
                    self.scheduler.agent.context.clear()
                    print("  🧹 对话上下文已清空")
                    continue

                if user_input == "/save":
                    filename = f"scheduler_demo_{datetime.now():%Y%m%d_%H%M%S}.md"
                    self.scheduler.agent.context.save(filename, format="markdown")
                    print(f"  💾 对话已保存: {filename}")
                    continue

                if user_input == "/model":
                    print(f"  🤖 模型: {' '.join(DEMO_QUICK_ARGUMENTS)}")
                    continue

                # 默认：作为普通消息入队
                self.enqueue_message(user_input, "normal")

            except Exception as e:
                print(f"  ❌ 错误: {e}")

    async def interactive_loop(self) -> None:
        """交互式命令循环。"""
        print("\n" + "=" * 50)
        print("  🎯 HawiScheduler 交互式演示")
        print("=" * 50)
        print("\n调度器在后台自动运行，直接输入消息即可入队")
        print("输入 'help' 查看可用命令\n")

        # 先设置 running 标志
        self.running = True
        
        # 启动调度器（后台任务）
        scheduler_task = asyncio.create_task(self.run_scheduler())
        
        # 启动输入读取（后台任务）
        input_task = asyncio.create_task(self.read_input())
        
        # 处理命令（主循环）
        try:
            await self.process_commands()
        finally:
            # 清理
            self.running = False
            self.scheduler.stop()
            scheduler_task.cancel()
            input_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass
            try:
                await input_task
            except asyncio.CancelledError:
                pass


async def auto_demo(scheduler: HawiScheduler) -> None:
    """自动演示模式 - 展示调度器的核心功能。"""
    demo = SchedulerDemo(scheduler, demo_mode="auto")

    print("\n" + "=" * 60)
    print("  🤖 HawiScheduler 自动演示")
    print("=" * 60)
    print("\n这个演示将展示:")
    print("  1. 三层消息队列 (NORMAL/HIGH_PRIO/URGENT)")
    print("  2. 紧急消息中断功能")
    print("  3. 队列优先级调度")
    print()

    # 启动调度器
    scheduler_task = asyncio.create_task(demo.run_scheduler())
    await asyncio.sleep(0.5)

    # 演示 1: 普通消息入队
    print("\n📌 演示 1: 入队普通消息")
    demo.enqueue_message("请介绍一下 Python 编程语言", "normal")
    demo.enqueue_message("Python 的列表和元组有什么区别", "normal")
    demo.show_status()
    await asyncio.sleep(5)

    # 演示 2: 高优先级消息（如果正在执行，会在工具调用后插入）
    print("\n📌 演示 2: 高优先级消息插队")
    demo.enqueue_message("[重要] 请先回答这个问题: 什么是装饰器?", "high_prio")
    demo.show_status()
    await asyncio.sleep(5)

    # 演示 3: 紧急消息（打断）
    print("\n📌 演示 3: 紧急消息打断当前执行")
    print("   (如果有正在执行的普通任务会被中断)")
    demo.enqueue_message("[紧急] 停止当前任务，回答这个紧急问题: 2+2=?", "urgent")
    demo.show_status()
    await asyncio.sleep(5)

    # 演示 4: 批量入队
    print("\n📌 演示 4: 批量入队展示优先级处理")
    demo.enqueue_message("解释什么是异步编程", "normal")
    demo.enqueue_message("[紧急] 快速回答: Python 的版本号是多少?", "urgent")
    demo.enqueue_message("[高优] 简短解释什么是 GIL", "high_prio")
    demo.enqueue_message("列举 Python 的常用数据结构", "normal")
    demo.show_status()

    print("\n⏳ 等待队列处理...")
    await asyncio.sleep(15)

    # 结束
    print("\n📌 演示结束")
    demo.show_status()
    demo.scheduler.stop()
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    print("\n✅ 自动演示完成!")
    print(f"   总共处理了 {demo.message_counter} 条消息")


def main():
    argv = sys.argv[1:]

    # 解析参数
    demo_mode = "interactive"  # auto, interactive
    streaming = True

    if "--demo" in argv:
        idx = argv.index("--demo")
        argv.pop(idx)
        if idx < len(argv):
            demo_mode = argv.pop(idx)

    if "--no-streaming" in argv:
        argv.remove("--no-streaming")
        streaming = False

    # 创建模型和调度器
    factory_name, model = create_model_from_argv(argv)
    scheduler = create_scheduler(model, streaming=streaming)

    print(f"\n🚀 HawiScheduler Demo")
    print(f"   模型: {factory_name}")
    print(f"   模式: {demo_mode}")
    if streaming:
        print(f"   流式输出: 已启用")
    print()

    # 设置事件打印机（实时输出）
    printer = create_printer("auto", streaming=streaming)
    scheduler.agent.subscribe(printer.handle)

    # 运行演示
    try:
        if demo_mode == "auto":
            asyncio.run(auto_demo(scheduler))
        else:
            demo = SchedulerDemo(scheduler, demo_mode="interactive")
            asyncio.run(demo.interactive_loop())
    except KeyboardInterrupt:
        print("\n\n👋 再见!")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
