"""
使用 @tool 裝飾器的簡化工具實現
展示 LangChain 的 @tool 裝飾器用法
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


# 使用 @tool 裝飾器建立 Tavily 搜尋工具
@tool
def tavily_search_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic"
) -> List[Dict[str, Any]]:
    """
    使用 Tavily API 進行網路搜尋，獲取最新的網頁內容和資訊

    Args:
        query: 搜尋查詢字串
        max_results: 最大搜尋結果數量 (1-20)
        search_depth: 搜尋深度，basic 或 advanced

    Returns:
        搜尋結果列表，包含標題、URL、內容等資訊
    """
    try:
        # 檢查 API 金鑰
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return [{"error": "TAVILY_API_KEY 環境變數未設定"}]

        # 建立 Tavily 工具
        tavily_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth
        )

        # 執行搜尋
        results = tavily_tool.run(query)

        # 格式化結果
        formatted_results = []
        if results and isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                        "published_date": result.get("published_date", "")
                    })
                else:
                    formatted_results.append({
                        "title": str(result),
                        "url": "",
                        "content": str(result),
                        "score": 0.0,
                        "published_date": ""
                    })
        else:
            formatted_results.append({
                "title": "無搜尋結果",
                "url": "",
                "content": "未找到相關內容",
                "score": 0.0,
                "published_date": ""
            })

        return formatted_results

    except Exception as e:
        return [{"error": f"搜尋失敗: {str(e)}"}]


# 使用 @tool 裝飾器建立檔案讀取工具
@tool
def file_read_tool(
    file_path: str,
    encoding: str = "utf-8",
    max_size: int = 1024*1024
) -> Dict[str, Any]:
    """
    讀取本地檔案內容，支援文字檔案

    Args:
        file_path: 要讀取的檔案路徑
        encoding: 檔案編碼格式 (utf-8, gbk, ascii)
        max_size: 最大檔案大小（位元組），預設 1MB

    Returns:
        包含檔案內容和元數據的字典
    """
    try:
        # 安全性檢查
        if not os.path.exists(file_path):
            return {"error": f"檔案不存在: {file_path}"}

        # 檢查檔案大小
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return {"error": f"檔案過大: {file_size} bytes > {max_size} bytes"}

        # 讀取檔案
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        return {
            "file_path": file_path,
            "content": content,
            "size": file_size,
            "encoding": encoding
        }

    except Exception as e:
        return {"error": f"讀取檔案失敗: {str(e)}"}


# 使用 @tool 裝飾器建立資料庫查詢工具
@tool
def database_query_tool(
    query: str,
    database_url: str,
    limit: int = 100
) -> Dict[str, Any]:
    """
    執行資料庫查詢，支援 SQLite 和 PostgreSQL

    Args:
        query: SQL 查詢語句（只允許 SELECT）
        database_url: 資料庫連接字串
        limit: 查詢結果限制數量 (1-1000)

    Returns:
        包含查詢結果和元數據的字典
    """
    try:
        # 安全性檢查
        if not query.strip().upper().startswith('SELECT'):
            return {"error": "只允許 SELECT 查詢"}

        # 模擬資料庫查詢結果
        return {
            "query": query,
            "database_url": database_url,
            "results": [
                {"id": 1, "name": "範例用戶", "age": 25},
                {"id": 2, "name": "測試用戶", "age": 30}
            ],
            "row_count": 2,
            "limit": limit
        }

    except Exception as e:
        return {"error": f"資料庫查詢失敗: {str(e)}"}


# 使用 @tool 裝飾器建立計算工具
@tool
def calculator_tool(
    expression: str
) -> Dict[str, Any]:
    """
    安全的數學計算工具

    Args:
        expression: 數學表達式，例如 "2 + 3 * 4"

    Returns:
        包含計算結果的字典
    """
    try:
        # 只允許安全的數學運算
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "表達式包含不安全的字符"}

        # 執行計算
        result = eval(expression)

        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }

    except Exception as e:
        return {"error": f"計算失敗: {str(e)}"}


# 工具集合
def get_all_tools():
    """獲取所有工具"""
    return [
        tavily_search_tool,
        file_read_tool,
        database_query_tool,
        calculator_tool
    ]


# 測試函數
def test_tools():
    """測試所有工具"""
    print("🧪 測試 @tool 裝飾器工具")
    print("=" * 40)

    # 測試 Tavily 搜尋
    print("1. 測試 Tavily 搜尋工具...")
    try:
        result = tavily_search_tool.invoke({
            "query": "人工智慧最新發展",
            "max_results": 3
        })
        print(f"   搜尋結果數量: {len(result)}")
        if result and not any("error" in r for r in result):
            print(f"   第一個結果: {result[0].get('title', '無標題')}")
    except Exception as e:
        print(f"   搜尋測試失敗: {e}")

    # 測試檔案讀取
    print("\n2. 測試檔案讀取工具...")
    try:
        result = file_read_tool.invoke({
            "file_path": "test.txt",
            "max_size": 1024
        })
        if "error" in result:
            print(f"   檔案讀取結果: {result['error']}")
        else:
            print(f"   檔案大小: {result['size']} bytes")
    except Exception as e:
        print(f"   檔案讀取測試失敗: {e}")

    # 測試資料庫查詢
    print("\n3. 測試資料庫查詢工具...")
    try:
        result = database_query_tool.invoke({
            "query": "SELECT * FROM users",
            "database_url": "sqlite:///test.db"
        })
        print(f"   查詢結果行數: {result.get('row_count', 0)}")
    except Exception as e:
        print(f"   資料庫查詢測試失敗: {e}")

    # 測試計算工具
    print("\n4. 測試計算工具...")
    try:
        result = calculator_tool.invoke({
            "expression": "2 + 3 * 4"
        })
        if "error" in result:
            print(f"   計算結果: {result['error']}")
        else:
            print(f"   計算結果: {result['result']}")
    except Exception as e:
        print(f"   計算測試失敗: {e}")


if __name__ == "__main__":
    test_tools()
