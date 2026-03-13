"""
Web 抓取插件

提供网页内容抓取功能，支持渐进式降级策略：
1. Cloudflare Markdown for Agents
2. markdownify 本地转换
3. 正则清理后的裸 HTML

示例:
    from hawi_plugins.web import WebPlugin

    plugin = WebPlugin()
    agent = HawiAgent(model=model, plugins=[plugin])
"""

from .plugin import WebPlugin

__all__ = ["WebPlugin"]
