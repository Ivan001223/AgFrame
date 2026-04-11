import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlencode

from defusedxml import ElementTree

import redis
import requests

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    query: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    provider: str
    cached: bool = False
    total_results: int = 0
    response_time_ms: int = 0


@dataclass
class BrowserPreview:
    url: str
    final_url: str
    title: str
    description: str
    status_code: int | None = None
    content_type: str | None = None
    fetch_method: str = "http_fallback"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SearchCache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _make_key(self, query: str, provider: str) -> str:
        hash_key = hashlib.sha256(f"{provider}:{query}".encode("utf-8", errors="ignore")).hexdigest()
        return f"agframe:search:{hash_key}"

    def get(self, query: str, provider: str) -> str | None:
        key = self._make_key(query, provider)
        try:
            return self.client.get(key)
        except Exception as exc:
            logger.warning("search cache get failed provider=%s query=%s error=%s", provider, query, exc)
            return None

    def set(self, query: str, provider: str, result: str) -> None:
        key = self._make_key(query, provider)
        try:
            self.client.setex(key, self.ttl, result)
        except Exception as exc:
            logger.warning("search cache set failed provider=%s query=%s error=%s", provider, query, exc)

    def delete(self, query: str, provider: str) -> None:
        key = self._make_key(query, provider)
        try:
            self.client.delete(key)
        except Exception as exc:
            logger.warning("search cache delete failed provider=%s query=%s error=%s", provider, query, exc)


class SearchProvider:
    def __init__(self, name: str):
        self.name = name

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class ArxivProvider(SearchProvider):
    def __init__(self, max_results: int = 5):
        super().__init__("arxiv")
        self.max_results = max_results

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        try:
            return _search_arxiv(query, max_results=max_results or self.max_results)
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return []


class TavilyProvider(SearchProvider):
    def __init__(self, api_key: str, max_results: int = 5):
        super().__init__("tavily")
        self.api_key = api_key
        self.max_results = max_results

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tool = TavilySearchResults(tavily_api_key=self.api_key, max_results=max_results or self.max_results)
            results = tool.invoke({"query": query})
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", r.get("snippet", "")),
                    provider=self.name,
                    query=query,
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []


class DuckDuckGoProvider(SearchProvider):
    def __init__(self, max_results: int = 5):
        super().__init__("duckduckgo")
        self.max_results = max_results

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            tool = DuckDuckGoSearchResults(max_results=max_results or self.max_results)
            raw_results = tool.invoke(query)
            parsed = json.loads(raw_results) if isinstance(raw_results, str) else raw_results
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    provider=self.name,
                    query=query,
                )
                for r in parsed
            ]
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []


class SerpAPIProvider(SearchProvider):
    def __init__(self, api_key: str, max_results: int = 5):
        super().__init__("serpapi")
        self.api_key = api_key
        self.max_results = max_results

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        try:
            from serpapi import GoogleSearch
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": max_results or self.max_results,
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic = results.get("organic_results", [])
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    provider=self.name,
                    query=query,
                )
                for r in organic
            ]
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")
            return []


class EnhancedSearchService:
    def __init__(self):
        self.cache = SearchCache()
        self._providers: dict[str, SearchProvider] = {}
        self._init_providers()

    def _init_providers(self):
        config = settings.search
        provider = config.provider
        tavily_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")
        serpapi_key = os.getenv("SERPAPI_API_KEY")

        if tavily_key:
            self._providers["tavily"] = TavilyProvider(tavily_key)
        self._providers["duckduckgo"] = DuckDuckGoProvider()
        self._providers["arxiv"] = ArxivProvider()

        if serpapi_key:
            self._providers["serpapi"] = SerpAPIProvider(serpapi_key)

    def get_provider(self, name: str = None) -> SearchProvider:
        provider_name = name or settings.search.provider
        return self._providers.get(provider_name, self._providers.get("duckduckgo"))

    async def search(
        self,
        query: str,
        provider: str = None,
        use_cache: bool = True,
        max_results: int = 5,
    ) -> SearchResponse:
        start_time = time.perf_counter()
        prov = self.get_provider(provider)
        cache_key = f"{prov.name}:{query}"

        if use_cache:
            cached = self.cache.get(query, prov.name)
            if cached:
                parsed = json.loads(cached)
                return SearchResponse(
                    query=query,
                    results=[SearchResult(**r) for r in parsed],
                    provider=prov.name,
                    cached=True,
                    total_results=len(parsed),
                    response_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

        results = await prov.search(query, max_results)
        result_dicts = [r.__dict__ for r in results]
        self.cache.set(query, prov.name, json.dumps(result_dicts))

        return SearchResponse(
            query=query,
            results=results,
            provider=prov.name,
            cached=False,
            total_results=len(results),
            response_time_ms=int((time.perf_counter() - start_time) * 1000),
        )


search_service = EnhancedSearchService()


class _PreviewHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag.lower() == "title":
            self.in_title = True
            return
        if tag.lower() != "meta":
            return
        attr_map = {str(key).lower(): str(value or "") for key, value in attrs}
        name = attr_map.get("name", "").lower()
        prop = attr_map.get("property", "").lower()
        if name in {"description", "twitter:description"} or prop == "og:description":
            content = attr_map.get("content", "").strip()
            if content and not self.description:
                self.description = html.unescape(content)

    def handle_endtag(self, tag: str):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str):
        if self.in_title and data.strip():
            self.title_parts.append(data.strip())


def _search_arxiv(query: str, max_results: int = 5) -> list[SearchResult]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"https://export.arxiv.org/api/query?{urlencode(params)}"
    response = requests.get(
        url,
        timeout=12,
        headers={"User-Agent": "AgFrame/1.0 (research loop paper-first)"},
    )
    response.raise_for_status()
    return _parse_arxiv_feed(response.text, query=query)


def _parse_arxiv_feed(feed_xml: str, *, query: str) -> list[SearchResult]:
    root = ElementTree.fromstring(feed_xml)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[SearchResult] = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace) or "").split())
        entry_id = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
        if not title or not entry_id:
            continue
        results.append(
            SearchResult(
                title=title,
                url=entry_id,
                snippet=summary,
                provider="arxiv",
                query=query,
            )
        )
    return results


def _extract_browser_preview_fields(markup: str) -> tuple[str, str]:
    parser = _PreviewHTMLParser()
    parser.feed(markup or "")
    parser.close()
    title = " ".join(" ".join(parser.title_parts).split())
    if not parser.description:
        match = re.search(r"<p[^>]*>(.*?)</p>", markup or "", re.IGNORECASE | re.DOTALL)
        if match:
            fallback = re.sub(r"<[^>]+>", " ", match.group(1))
            parser.description = " ".join(html.unescape(fallback).split())
    return title, parser.description or ""


def fetch_browser_previews(
    urls: list[str],
    *,
    max_previews: int = 3,
    timeout_seconds: int = 8,
) -> list[BrowserPreview]:
    try:
        previews = _fetch_browser_previews_playwright(
            urls,
            max_previews=max_previews,
            timeout_seconds=timeout_seconds,
        )
        if previews:
            return previews
    except Exception as exc:
        logger.warning("playwright browser preview failed; falling back to requests error=%s", exc)

    return _fetch_browser_previews_http_fallback(
        urls,
        max_previews=max_previews,
        timeout_seconds=timeout_seconds,
    )


def _fetch_browser_previews_playwright(
    urls: list[str],
    *,
    max_previews: int = 3,
    timeout_seconds: int = 8,
) -> list[BrowserPreview]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        logger.warning("playwright import unavailable for browser previews error=%s", exc)
        return []

    previews: list[BrowserPreview] = []
    seen: set[str] = set()
    timeout_ms = max(int(timeout_seconds * 1000), 1000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()
        for url in urls:
            normalized = str(url or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            try:
                response = page.goto(normalized, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 4000))
                title = (page.title() or "").strip()
                description = page.locator("meta[name='description']").first.get_attribute("content") or ""
                if not description.strip():
                    description = page.locator("meta[property='og:description']").first.get_attribute("content") or ""
                if not description.strip():
                    body_text = page.locator("body").inner_text(timeout=min(timeout_ms, 3000))
                    description = " ".join(str(body_text or "").split())[:400]
                previews.append(
                    BrowserPreview(
                        url=normalized,
                        final_url=page.url,
                        title=title or normalized,
                        description=description.strip(),
                        status_code=response.status if response else None,
                        content_type=(response.header_value("content-type") if response else None),
                        fetch_method="playwright",
                    )
                )
            except Exception as exc:
                logger.warning("playwright preview failed url=%s error=%s", normalized, exc)
            if len(previews) >= max_previews:
                break
        context.close()
        browser.close()
    return previews


def _fetch_browser_previews_http_fallback(
    urls: list[str],
    *,
    max_previews: int = 3,
    timeout_seconds: int = 8,
) -> list[BrowserPreview]:
    previews: list[BrowserPreview] = []
    seen: set[str] = set()
    for url in urls:
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            response = requests.get(
                normalized,
                timeout=timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0 AgFrameResearchPreview/1.0"},
            )
            response.raise_for_status()
            title, description = _extract_browser_preview_fields(response.text)
            previews.append(
                BrowserPreview(
                    url=normalized,
                    final_url=str(response.url),
                    title=title or normalized,
                    description=description,
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type"),
                    fetch_method="http_fallback",
                )
            )
        except Exception as exc:
            logger.warning("browser preview failed url=%s error=%s", normalized, exc)
        if len(previews) >= max_previews:
            break
    return previews


def format_search_results(response: SearchResponse, max_length: int = 2000) -> str:
    if not response.results:
        return "No results found."

    formatted = [f"Search results from {response.provider}" + (" (cached)" if response.cached else "") + ":\n"]
    for i, result in enumerate(response.results, 1):
        snippet = result.snippet[:300] + "..." if len(result.snippet) > 300 else result.snippet
        formatted.append(f"{i}. [{result.title}]({result.url})")
        formatted.append(f"   {snippet}\n")

    output = "\n".join(formatted)
    return output[:max_length] if max_length else output


async def enhanced_web_search(
    query: str,
    provider: str = None,
    use_cache: bool = True,
    max_results: int = 5,
) -> str:
    response = await search_service.search(query, provider, use_cache, max_results)
    return format_search_results(response)


async def enhanced_search_response(
    query: str,
    provider: str = None,
    use_cache: bool = True,
    max_results: int = 5,
) -> SearchResponse:
    return await search_service.search(query, provider, use_cache, max_results)
