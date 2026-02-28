"""
DumpManager - 事件转储管理器

用于调试和日志记录，将所有事件序列化保存到本地文件。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from hawi.events import Event


class DumpManager:
    """管理事件转储到文件的类。

    特性：
    - 自动创建目录
    - JSON 格式输出，便于分析
    - 增量写入，不影响性能
    - 错误静默处理，不影响主流程

    Example:
        dump_manager = DumpManager("./dumps/events.json")
        dump_manager.dump(event)
    """

    def __init__(self, dump_file: str | Path | None = None):
        """初始化 DumpManager。

        Args:
            dump_file: 转储文件路径，None 表示不转储
        """
        self._dump_file: Path | None = Path(dump_file) if dump_file else None
        self._initialized: bool = False

        if self._dump_file:
            self._initialize_file()

    def _initialize_file(self) -> None:
        """初始化转储文件。"""
        if not self._dump_file:
            return

        try:
            # 创建目录
            self._dump_file.parent.mkdir(parents=True, exist_ok=True)

            # 写入初始结构
            initial_data = {
                "session_start": time.time(),
                "session_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "events": [],
            }
            with open(self._dump_file, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)

            self._initialized = True
        except Exception:
            # 初始化失败则禁用转储
            self._dump_file = None
            self._initialized = False

    def dump(self, event: Event) -> None:
        """转储单个事件到文件。

        Args:
            event: 要转储的事件
        """
        if not self._dump_file or not self._initialized:
            return

        try:
            # 读取现有数据
            with open(self._dump_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 转换事件为字典
            event_data = event.model_dump(mode="json")

            # 添加事件记录
            event_record = {
                "timestamp": time.time(),
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "type": event.type,
                "source": event.source,
                "data": event_data,
            }

            data["events"].append(event_record)

            # 写回文件
            with open(self._dump_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        except Exception:
            # 静默忽略错误，不影响主流程
            pass

    def dump_raw(self, data: dict[str, Any]) -> None:
        """转储原始数据（用于调试特定场景）。

        Args:
            data: 要转储的字典数据
        """
        if not self._dump_file or not self._initialized:
            return

        try:
            with open(self._dump_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            if "raw_dumps" not in file_data:
                file_data["raw_dumps"] = []

            file_data["raw_dumps"].append({
                "timestamp": time.time(),
                "data": data,
            })

            with open(self._dump_file, "w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False, default=str)

        except Exception:
            pass

    def is_enabled(self) -> bool:
        """检查转储功能是否启用。"""
        return self._initialized and self._dump_file is not None

    def get_dump_path(self) -> Path | None:
        """获取转储文件路径。"""
        return self._dump_file
