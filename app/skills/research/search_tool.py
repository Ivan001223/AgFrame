import logging
import os

from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


@tool
def web_search_unavailable(query: str) -> str:
    """当搜索依赖未安装或未配置时的兜底搜索工具。"""
    return "搜索工具不可用：请安装 ddgs（pip install -U ddgs）或在配置中切换到 Tavily 并设置 API Key。"

class SearchToolFactory:
    @staticmethod
    def get_search_tool(return_results_obj=False):
        """
        返回配置的搜索工具提供方。
        
        参数:
            return_results_obj: 为 True 时返回 SearchResults 对象（包含元数据）。
                                为 False 时返回 SearchRun 对象（仅文本）。
        """
        config = settings.search
        provider = config.provider
        tavily_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        if provider == "tavily" or (tavily_key and provider != "duckduckgo"):
            if tavily_key:
                return TavilySearchResults(tavily_api_key=tavily_key, max_results=5)
            else:
                logger.warning("Tavily selected but API key not found, falling back to DuckDuckGo")
        
        if return_results_obj:
            try:
                return DuckDuckGoSearchResults()
            except ImportError:
                return web_search_unavailable
        else:
            try:
                return DuckDuckGoSearchRun()
            except ImportError:
                return web_search_unavailable

def get_search_tool(return_results_obj=False) -> DuckDuckGoSearchRun | DuckDuckGoSearchResults | str:
    return SearchToolFactory.get_search_tool(return_results_obj)
