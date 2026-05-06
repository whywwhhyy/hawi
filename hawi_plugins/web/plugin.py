from __future__ import annotations

from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult
from .fetch import Fetcher


class WebPlugin(HawiPlugin):
    """
    Web 抓取插件，支持渐进式降级获取网页 Markdown 内容

    使用三层策略：
    1. Cloudflare Markdown for Agents (Accept: text/markdown)
    2. markdownify 本地转换
    3. 正则清理后的裸 HTML

    示例:
        plugin = WebPlugin()
        agent = HawiAgent(model=model, plugins=[plugin])
        # Agent 可以调用 fetch 工具抓取网页
    """

    def __init__(self):
        super().__init__()
        self._fetcher = Fetcher()

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    def clone(self) -> 'WebPlugin':
        """
        创建副本用于 Agent fork/clone

        WebPlugin 是无状态的，返回 self 即可
        """
        return self

    @tool()
    def fetch(
        self,
        url: str,
        max_length: int = 5000,
        start_index: int = 0,
        raw: bool = False,
        clean: bool = True,
    ) -> ToolResult:
        """
        抓取网页并转换为 Markdown

        首先尝试 Cloudflare 的 Markdown for Agents 服务，
        如果不可用则使用 markdownify 本地转换，
        如果失败则返回正则清理后的裸 HTML。

        Args:
            url: 要抓取的 URL（必需）
            max_length: 最大返回字符数（默认 5000，最大 100000）
            start_index: 开始读取的字符位置（用于分页，默认 0）
            raw: 是否返回原始 HTML（默认 false，返回 Markdown）
            clean: 是否启用智能内容过滤（默认 true，过滤导航栏等低信息内容）

        Returns:
            网页内容（Markdown 或 HTML）

        示例:
            fetch(url="https://example.com/article")
            fetch(url="https://example.com/long", max_length=10000)
            fetch(url="https://example.com/long", start_index=5000, max_length=5000)
            fetch(url="https://example.com", clean=False)  # 不过滤内容
        """
        try:
            result = self._fetcher.fetch(
                url=url,
                max_length=max_length,
                start_index=start_index,
                raw=raw,
                clean=clean,
            )

            # 构建输出，包含元数据信息
            header_lines = [
                f"[WebFetch] 来源: {result.source}",
                f"[WebFetch] 类型: {result.content_type}",
                f"[WebFetch] 总长度: {result.total_length} 字符",
            ]
            if result.truncated:
                header_lines.append(f"[WebFetch] 已截断，显示 {len(result.content)} 字符")
            header = "\n".join(header_lines) + "\n" + "=" * 60 + "\n\n"

            # 如果被截断，添加提示
            content = result.content
            if result.truncated:
                content += f"\n\n... (内容已截断，共 {result.total_length} 字符，使用 start_index={start_index + max_length} 继续读取)"

            if not result.success:
                return ToolResult(
                    output=header + content,
                    error=content or "抓取失败",
                    success=False,
                )

            return ToolResult(
                output=header + content,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                output="",
                error=f"抓取失败: {str(e)}",
                success=False,
            )
