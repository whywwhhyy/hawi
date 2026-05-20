import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from markdownify import markdownify as md


@dataclass
class FetchResult:
    content: str
    truncated: bool
    total_length: int
    content_type: str
    source: str  # 'cloudflare', 'markdownify', 'cleaned_html', 'raw'
    success: bool = True


class Fetcher:
    """网页抓取器，支持渐进式降级"""

    USER_AGENT = "HawiAgent/0.1.0 WebPlugin"
    TIMEOUT = 30.0
    MAX_REDIRECTS = 5

    def __init__(self):
        self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def close(self):
        """关闭 HTTP 客户端"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.TIMEOUT,
                follow_redirects=True,
                max_redirects=self.MAX_REDIRECTS,
            )
        return self._client

    def _is_markdown_response(self, content_type: str) -> bool:
        """检查响应是否为 Markdown"""
        return "text/markdown" in content_type.lower()

    def _validate_url(self, url: str) -> str:
        """验证并规范化 URL"""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        # Ensure netloc looks like a valid domain (contains a dot for TLD)
        if "." not in parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        return url

    def _fetch_html(self, url: str, accept_markdown: bool = False) -> tuple[str, str, str]:
        """获取网页内容（带重试）"""
        url = self._validate_url(url)

        headers = {"User-Agent": self.USER_AGENT}
        if accept_markdown:
            headers["Accept"] = "text/markdown, text/html"

        client = self._get_client()

        try:
            response = client.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
        except httpx.TimeoutException:
            raise RuntimeError(f"Request timeout after {self.TIMEOUT}s")
        except httpx.RequestError as e:
            raise RuntimeError(f"Request failed: {e}")

        content_type = response.headers.get("content-type", "").lower()
        return content_type, response.text, str(response.url)

    def _markdownify_convert(self, html: str) -> str:
        """使用 markdownify 将 HTML 转换为 Markdown"""
        try:
            result = md(html, heading_style="ATX")
            # 清理多余空行
            lines = result.split("\n")
            cleaned = []
            prev_empty = False
            for line in lines:
                is_empty = not line.strip()
                if is_empty and prev_empty:
                    continue
                cleaned.append(line)
                prev_empty = is_empty
            return "\n".join(cleaned).strip()
        except Exception as e:
            raise RuntimeError(f"markdownify conversion failed: {e}")

    def _clean_html_regex(self, html: str) -> str:
        """
        使用正则表达式清理 HTML 中的无用标签
        这是最后的降级方案
        """
        # 删除的标签列表（及其内容）
        tags_to_remove = [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "svg",
            "canvas",
            "noscript",
            "template",
        ]

        # 删除标签及其内容
        for tag in tags_to_remove:
            pattern = re.compile(
                rf"<{tag}[^>]*>.*?</{tag}>",
                re.DOTALL | re.IGNORECASE
            )
            html = pattern.sub("", html)

        # 删除 HTML 注释
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # 删除 class 和 style 属性（简化 HTML）
        html = re.sub(r'\s+class="[^"]*"', "", html)
        html = re.sub(r"\s+class='[^']*'", "", html)
        html = re.sub(r'\s+style="[^"]*"', "", html)
        html = re.sub(r"\s+style='[^']*'", "", html)

        # 压缩多余空白
        html = re.sub(r"\n\s*\n+", "\n\n", html)

        return html.strip()

    def _extract_main_content(self, markdown: str) -> str:
        """
        从 Markdown 中提取主要内容，基于文本密度和长度启发式算法

        通用策略（不依赖特定网站）：
        1. 计算每行的信息密度（文本长度/链接数量）
        2. 识别并折叠低密度区域（导航栏、页脚等）
        3. 保留高密度区域（正文内容）
        4. 被折叠部分显示简短提示
        """
        lines = markdown.split('\n')
        result_lines = []
        folded_count = 0

        def calc_line_score(line: str) -> float:
            """计算行的内容得分，越高越可能是正文"""
            stripped = line.strip()
            if not stripped:
                return 0

            # 文本长度得分
            text_length = len(stripped)

            # 链接数量惩罚
            link_pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
            links = link_pattern.findall(stripped)
            link_count = len(links)

            # 计算链接密度
            link_text_length = sum(len(l) for l in links)
            non_link_text = text_length - link_text_length - link_count * 4  # 减去 []() 符号

            # 如果整行都是链接，得分很低
            if link_count > 0 and non_link_text < 10:
                return 0.1

            # 短行得分低（可能是导航）
            if text_length < 15:
                return 0.2

            # 标题行得分高
            if stripped.startswith('# '):
                return 3.0
            if stripped.startswith('## '):
                return 2.0
            if stripped.startswith('### '):
                return 1.5

            # 列表项得分中等
            if stripped.startswith(('- ', '* ')):
                # 长列表项更可能是正文
                if text_length > 30:
                    return 1.0
                return 0.3

            # 表格行得分中等
            if stripped.startswith('|'):
                return 0.8

            # 代码块得分高
            if stripped.startswith('```'):
                return 2.0

            # 普通文本按长度计分
            return min(1.0, text_length / 50)

        # 导航模式关键词
        nav_patterns = ['Platform', 'Solutions', 'Resources', 'Enterprise',
                       'Pricing', 'Documentation', 'Support', 'Contact',
                       'About', 'Features', 'Products', 'Services']

        def is_likely_noise(line: str) -> bool:
            """判断行是否可能是噪声（导航、模板等）"""
            stripped = line.strip()

            # 模板占位符
            if '{{' in stripped and '}}' in stripped:
                return True

            # 纯 UI 文本
            ui_texts = ['Toggle navigation', 'Menu', 'Search', 'Clear', 'Cancel',
                       'Submit', 'Dismiss', 'Reload to refresh', '{{ message }}',
                       'Search or jump to...', 'Appearance settings', 'Skip to content']
            if stripped in ui_texts:
                return True

            # 仅包含链接的短行
            link_pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
            links = link_pattern.findall(stripped)
            if len(links) >= 2:
                text_without_links = link_pattern.sub('', stripped)
                text_without_links = re.sub(r'[\[\]\(\)]', '', text_without_links).strip()
                if len(text_without_links) < 15:
                    return True

            return False

        def is_likely_nav_item(line: str) -> bool:
            """判断行是否可能是导航项（基于常见导航模式）"""
            stripped = line.strip()

            # 单独的导航关键词
            if stripped in nav_patterns:
                return True

            # 列表项且包含导航关键词
            if stripped.startswith(('- ', '* ')):
                content = stripped[2:].strip()
                # 短列表项包含导航关键词
                if len(content) < 30:
                    for pattern in nav_patterns:
                        if pattern in content:
                            return True
                # 列表项但主要是链接
                link_pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
                links = link_pattern.findall(content)
                if len(links) >= 1:
                    text_without_links = link_pattern.sub('', content)
                    text_without_links = re.sub(r'[\[\]\(\)]', '', text_without_links).strip()
                    # 如果链接文字包含营销性词汇
                    marketing_words = ['Automate', 'Instant', 'Secure', 'Better', 'Manage',
                                      'Build', 'Deploy', 'Write', 'Find', 'Stop', 'Plan',
                                      'Track', 'Review', 'Modernization', 'Healthcare',
                                      'Financial', 'Manufacturing', 'Government']
                    for word in marketing_words:
                        if word in text_without_links:
                            return True

            return False

        # 第一遍：计算每行得分并标记噪声
        line_scores = []
        for line in lines:
            if is_likely_noise(line) or is_likely_nav_item(line):
                line_scores.append((line, -1))  # 标记为噪声
            else:
                score = calc_line_score(line)
                line_scores.append((line, score))

        def is_list_item(line: str) -> bool:
            """判断是否为列表项"""
            stripped = line.strip()
            return bool(re.match(r'^[\s]*[-\*\+]\s', stripped))

        def is_nested_list_block(start_idx: int) -> tuple[bool, int]:
            """
            检测从 start_idx 开始是否是连续的嵌套列表块（导航菜单）
            返回: (是否是导航块, 块结束索引)
            """
            i = start_idx
            list_item_count = 0
            max_depth = 0

            while i < len(line_scores):
                line, score = line_scores[i]
                stripped = line.strip()

                if not stripped:
                    i += 1
                    continue

                if is_list_item(line):
                    list_item_count += 1
                    # 计算缩进深度
                    leading_space = len(line) - len(line.lstrip())
                    depth = leading_space // 2
                    max_depth = max(max_depth, depth)
                    i += 1
                elif score < 0:
                    # 噪声行可能是导航的一部分
                    i += 1
                elif score >= 0.3:
                    # 高分行，可能是正文，停止
                    break
                else:
                    # 低分行，继续检查
                    i += 1

            # 如果有很多列表项且有多层嵌套，认为是导航菜单
            is_nav = list_item_count >= 5 and max_depth >= 2
            return is_nav, i

        # 第二遍：识别连续低分区域并折叠
        i = 0
        while i < len(line_scores):
            line, score = line_scores[i]

            # 噪声行直接跳过
            if score < 0:
                folded_count += 1
                i += 1
                continue

            # 检测列表项块（可能是导航菜单）
            if is_list_item(line):
                is_nav_block, block_end = is_nested_list_block(i)
                if is_nav_block:
                    block_size = block_end - i
                    if folded_count > 0:
                        result_lines.append(f"\n[...折叠 {folded_count} 行低信息内容...]\n")
                        folded_count = 0
                    result_lines.append(f"[...折叠 {block_size} 行导航菜单...]")
                    i = block_end
                    continue

            # 低分行（可能是导航），检查是否是连续区域
            if score < 0.3 and line.strip():
                # 向前看，找连续低分行
                low_score_start = i
                while i < len(line_scores) and line_scores[i][1] < 0.3 and line_scores[i][1] >= 0:
                    i += 1
                low_score_end = i
                low_score_count = low_score_end - low_score_start

                # 如果连续低分行超过 3 行，折叠它们
                if low_score_count > 3:
                    if folded_count > 0:
                        result_lines.append(f"\n[...折叠 {folded_count} 行低信息内容...]\n")
                        folded_count = 0
                    result_lines.append(f"[...折叠 {low_score_count} 行导航/页脚内容...]")
                else:
                    # 少量低分行保留
                    for j in range(low_score_start, low_score_end):
                        result_lines.append(line_scores[j][0])
            else:
                # 高分行直接保留
                if folded_count > 0:
                    result_lines.append(f"\n[...折叠 {folded_count} 行低信息内容...]\n")
                    folded_count = 0
                result_lines.append(line)
                i += 1

        # 处理末尾的折叠计数
        if folded_count > 0:
            result_lines.append(f"\n[...折叠 {folded_count} 行低信息内容...]")

        # 清理多余空行
        final_lines = []
        prev_empty = False
        for line in result_lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            final_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(final_lines).strip()

    def _apply_pagination(
        self,
        content: str,
        max_length: int,
        start_index: int
    ) -> tuple[str, bool, int]:
        """应用分页"""
        total = len(content)

        if start_index >= total:
            return "", False, total

        content = content[start_index:]

        if len(content) > max_length:
            return content[:max_length], True, total

        return content, False, total

    def fetch(
        self,
        url: str,
        max_length: int = 5000,
        start_index: int = 0,
        raw: bool = False,
        clean: bool = True,
    ) -> FetchResult:
        """
        抓取网页内容（渐进式降级）

        Level 1: Cloudflare Markdown (Accept: text/markdown)
        Level 2: markdownify 转换
        Level 3: 正则清理后的裸 HTML
        """
        if max_length < 1:
            max_length = 1
        if max_length > 100000:
            max_length = 100000

        try:
            # 首先尝试 Cloudflare Markdown
            content_type, content, final_url = self._fetch_html(url, accept_markdown=True)
            source = "raw"

            if raw:
                # 原始模式：直接返回（但也清理一下）
                content = self._clean_html_regex(content)
                source = "cleaned_html"
            elif self._is_markdown_response(content_type):
                # Level 1: Cloudflare Markdown
                if clean:
                    content = self._extract_main_content(content)
                source = "cloudflare"
            else:
                # Level 2: markdownify 转换
                try:
                    content = self._markdownify_convert(content)
                    if clean:
                        content = self._extract_main_content(content)
                    source = "markdownify"
                except Exception:
                    # Level 3: 正则清理
                    content = self._clean_html_regex(content)
                    source = "cleaned_html"

            # 应用分页
            paginated, truncated, total = self._apply_pagination(content, max_length, start_index)

            return FetchResult(
                content=paginated,
                truncated=truncated,
                total_length=total,
                content_type="text/markdown" if source in ("cloudflare", "markdownify") else "text/html",
                source=source,
                success=True,
            )
        except (ValueError, RuntimeError) as e:
            return FetchResult(
                content=str(e),
                truncated=False,
                total_length=0,
                content_type="text/plain",
                source="error",
                success=False,
            )
