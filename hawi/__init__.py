"""Hawi - AI Agent framework with model compatibility layers."""

import os

from .agent import HawiAgent
from .agent.context import AgentContext
from .agent.result import AgentRunResult, ToolCallRecord

__all__ = [
    "HawiAgent",
    "AgentContext",
    "AgentRunResult",
    "ToolCallRecord",
]

# 自动加载配置（如果存在配置文件且未禁用）
# 可通过设置 HAWI_AUTO_CONFIG=0 禁用自动加载
def _auto_load_config():
    """自动加载模型配置和 API 密钥（如果存在）。"""
    if os.environ.get("HAWI_AUTO_CONFIG", "1") == "0":
        return

    try:
        from pathlib import Path

        # 检查是否存在配置文件
        apikey_path = Path.cwd() / "apikey.yaml"
        models_path = Path.cwd() / "models.yaml"

        # 至少存在一个配置文件才自动加载
        if apikey_path.exists() or models_path.exists():
            from .config import setup_registry
            from .models import model_registry

            setup_registry(registry=model_registry)
    except Exception:
        # 自动加载失败不阻止导入，让用户手动处理
        pass


_auto_load_config()
del _auto_load_config
