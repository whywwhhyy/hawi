import pytest
from unittest.mock import Mock, patch
from hawi_plugins.web.fetch import Fetcher


def test_web_plugin_instance():
    from hawi_plugins.web.plugin import WebPlugin
    plugin = WebPlugin()
    assert plugin is not None
    assert hasattr(plugin, 'tools')
    assert hasattr(plugin, 'clone')


def test_fetcher_cloudflare_markdown():
    fetcher = Fetcher()
    # Mock response with text/markdown content-type
    content_type = "text/markdown; charset=utf-8"
    result = fetcher._is_markdown_response(content_type)
    assert result is True

    content_type = "text/html; charset=utf-8"
    result = fetcher._is_markdown_response(content_type)
    assert result is False


def test_fetcher_basic_fetch():
    fetcher = Fetcher()
    # Mock httpx to return simple HTML
    mock_response = Mock()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.url = "https://example.com"

    with patch.object(fetcher._get_client(), "get", return_value=mock_response):
        content_type, content, url = fetcher._fetch_html("https://example.com")
        assert "html" in content.lower()


def test_fetcher_markdownify_conversion():
    fetcher = Fetcher()
    html = "<h1>Title</h1><p>Hello <b>world</b></p>"
    result = fetcher._markdownify_convert(html)
    assert "# Title" in result
    assert "**world**" in result or "world" in result


def test_fetcher_regex_cleaning():
    fetcher = Fetcher()
    html = """
    <html>
    <head><script>alert('xss')</script><style>body{color:red}</style></head>
    <body>
    <nav>Menu</nav>
    <main>Content</main>
    <footer>Footer</footer>
    </body>
    </html>
    """
    result = fetcher._clean_html_regex(html)
    assert "<script>" not in result
    assert "<style>" not in result
    assert "<nav>" not in result
    assert "<footer>" not in result
    assert "Content" in result


def test_fetcher_fetch_with_pagination():
    fetcher = Fetcher()
    # Test with mock content
    content = "A" * 10000
    result, truncated, total = fetcher._apply_pagination(content, max_length=1000, start_index=0)
    assert len(result) == 1000
    assert truncated is True
    assert total == 10000

    # Test start_index
    result, truncated, total = fetcher._apply_pagination(content, max_length=1000, start_index=5000)
    assert result.startswith("A")
    assert truncated is True
    assert total == 10000


def test_fetcher_invalid_url():
    fetcher = Fetcher()
    with pytest.raises(ValueError):
        fetcher._validate_url("not-a-url")


def test_fetcher_http_error():
    fetcher = Fetcher()
    # This should handle 404 gracefully
    result = fetcher.fetch("https://httpbin.org/status/404")
    assert result.success is False
