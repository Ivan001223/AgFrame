import sys
import types

from app.skills.research.enhanced_search import (
    _extract_browser_preview_fields,
    _parse_arxiv_feed,
    fetch_browser_previews,
)


def test_parse_arxiv_feed_extracts_entries():
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/1234.5678</id>
    <title> Paper Title </title>
    <summary>
      This is the abstract.
    </summary>
  </entry>
</feed>
"""

    results = _parse_arxiv_feed(feed, query="test query")

    assert len(results) == 1
    assert results[0].title == "Paper Title"
    assert results[0].url == "https://arxiv.org/abs/1234.5678"
    assert results[0].snippet == "This is the abstract."
    assert results[0].provider == "arxiv"


def test_extract_browser_preview_fields_prefers_title_and_meta_description():
    html_doc = """
<html>
  <head>
    <title>Example Research Page</title>
    <meta name="description" content="Structured browser preview text." />
  </head>
  <body>
    <p>Fallback paragraph.</p>
  </body>
</html>
"""

    title, description = _extract_browser_preview_fields(html_doc)

    assert title == "Example Research Page"
    assert description == "Structured browser preview text."


def test_fetch_browser_previews_prefers_playwright_when_available(monkeypatch):
    class _Response:
        status = 200

        @staticmethod
        def header_value(name: str):
            assert name == "content-type"
            return "text/html"

    class _Locator:
        def __init__(self, query: str):
            self.query = query
            self.first = self

        def get_attribute(self, name: str):
            assert name == "content"
            if self.query == "meta[name='description']":
                return "Preview from headless browser"
            return None

        def inner_text(self, timeout: int | None = None):
            return "Fallback body text"

    class _Page:
        url = "https://example.com/final"

        @staticmethod
        def goto(url: str, wait_until: str, timeout: int):
            assert url == "https://example.com"
            assert wait_until == "domcontentloaded"
            assert timeout == 3000
            return _Response()

        @staticmethod
        def wait_for_load_state(state: str, timeout: int):
            assert state == "networkidle"
            assert timeout == 3000

        @staticmethod
        def title():
            return "Rendered Preview"

        @staticmethod
        def locator(query: str):
            return _Locator(query)

    class _Context:
        @staticmethod
        def new_page():
            return _Page()

        @staticmethod
        def close():
            return None

    class _Browser:
        @staticmethod
        def new_context(ignore_https_errors: bool):
            assert ignore_https_errors is True
            return _Context()

        @staticmethod
        def close():
            return None

    class _Chromium:
        @staticmethod
        def launch(headless: bool):
            assert headless is True
            return _Browser()

    class _PlaywrightManager:
        chromium = _Chromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    requests_called = {"value": False}

    def _unexpected_requests_get(*args, **kwargs):
        requests_called["value"] = True
        raise AssertionError("HTTP fallback should not be used when Playwright succeeds")

    monkeypatch.setattr("app.skills.research.enhanced_search.requests.get", _unexpected_requests_get)
    sync_api_module = types.SimpleNamespace(sync_playwright=lambda: _PlaywrightManager())
    monkeypatch.setitem(sys.modules, "playwright", types.SimpleNamespace(sync_api=sync_api_module))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api_module)

    previews = fetch_browser_previews(["https://example.com"], max_previews=1, timeout_seconds=3)

    assert requests_called["value"] is False
    assert len(previews) == 1
    assert previews[0].url == "https://example.com"
    assert previews[0].final_url == "https://example.com/final"
    assert previews[0].title == "Rendered Preview"
    assert previews[0].description == "Preview from headless browser"
    assert previews[0].fetch_method == "playwright"
