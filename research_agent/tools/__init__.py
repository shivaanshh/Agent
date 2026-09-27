from __future__ import annotations

from .base import Param, Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolResult, TransientToolError
from .calculator import calculator
from .fetch import fetch_page
from .search import news_search, web_search

DEFAULT_TOOLS = [news_search, web_search, fetch_page, calculator]


def build_default_registry(settings, injector=None, emit=None, **kwargs) -> ToolRegistry:
    registry = ToolRegistry(
        injector,
        timeout=settings.tool_timeout,
        retries=settings.tool_retries,
        backoff_base=settings.backoff_base,
        **({"emit": emit} if emit else {}),
        **kwargs,
    )
    for tool in DEFAULT_TOOLS:
        registry.register(tool)
    return registry


__all__ = [
    "Param", "Tool", "ToolContext", "ToolError", "ToolOutput", "ToolRegistry", "ToolResult",
    "TransientToolError", "build_default_registry", "DEFAULT_TOOLS",
]
