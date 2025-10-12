"""
工具模組
提供各種搜尋和研究工具（使用 @tool 裝飾器）
"""
# 簡化工具（使用 @tool 裝飾器）
from .simple_tools import (
    tavily_search_tool,
    duckduckgo_search_tool,
    file_read_tool,
    file_write_tool,
    database_query_tool,
    calculator_tool,
    weather_tool,
    get_all_tools
)

__all__ = [
    # 搜尋工具
    "tavily_search_tool",
    "duckduckgo_search_tool",
    # 檔案工具
    "file_read_tool",
    "file_write_tool",
    # 資料庫工具
    "database_query_tool",
    # 計算工具
    "calculator_tool",
    # API 工具
    "weather_tool",
    # 工具管理器
    "get_all_tools"
]
