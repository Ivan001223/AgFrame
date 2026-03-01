import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import redis

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
        return self.client.get(key)

    def set(self, query: str, provider: str, result: str) -> None:
        key = self._make_key(query, provider)
        self.client.setex(key, self.ttl, result)

    def delete(self, query: str, provider: str) -> None:
        key = self._make_key(query, provider)
        self.client.delete(key)


class SearchProvider:
    def __init__(self, name: str):
        self.name = name

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


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
