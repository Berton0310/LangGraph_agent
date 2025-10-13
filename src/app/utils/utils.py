"""深度研究代理的實用函數和輔助工具。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from ..config import Configuration, SearchAPI
from ..prompt import summarize_webpage_prompt
from ..state.research_state import ResearchComplete, Summary

##########################
# Tavily 搜尋工具實用函數
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "針對全面、準確和可信結果優化的搜尋引擎。 "
    "當您需要回答有關當前事件的問題時很有用。"
)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news",
                             "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """從 Tavily 搜尋 API 獲取並總結搜尋結果。

    Args:
        queries: 要執行的搜尋查詢列表
        max_results: 每個查詢返回的最大結果數
        topic: 搜尋結果的主題過濾器（general、news 或 finance）
        config: 用於 API 金鑰和模型設定的運行時配置

    Returns:
        包含總結搜尋結果的格式化字串
    """
    # 步驟 1：異步執行搜尋查詢
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # 步驟 2：按 URL 去重結果，避免多次處理相同內容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # 步驟 3：使用配置設置總結模型
    configurable = Configuration.from_runnable_config(config)

    # 字符限制以保持在模型 token 限制內（可配置）
    max_char_to_include = configurable.max_content_length

    # 使用重試邏輯初始化總結模型
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # 步驟 4：創建總結任務（跳過空內容）
    async def noop():
        """沒有原始內容結果的無操作函數。"""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # 步驟 5：並行執行所有總結任務
    summaries = await asyncio.gather(*summarization_tasks)

    # 步驟 6：將結果與其總結結合
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # 步驟 7：格式化最終輸出
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """異步執行多個 Tavily 搜尋查詢。

    Args:
        search_queries: 要執行的搜尋查詢字串列表
        max_results: 每個查詢的最大結果數
        topic: 用於過濾結果的主題類別
        include_raw_content: 是否包含完整網頁內容
        config: 用於 API 金鑰訪問的運行時配置

    Returns:
        來自 Tavily API 的搜尋結果字典列表
    """
    # 使用配置中的 API 金鑰初始化 Tavily 客戶端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    # 為並行執行創建搜尋任務
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # 並行執行所有搜尋查詢並返回結果
    search_results = await asyncio.gather(*search_tasks)
    return search_results


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型總結網頁內容，具有超時保護。

    Args:
        model: 配置用於總結的聊天模型
        webpage_content: 要總結的原始網頁內容

    Returns:
        帶有關鍵摘錄的格式化總結，或總結失敗時的原始內容
    """
    try:
        # 使用當前日期上下文創建提示
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        # 執行總結並使用超時防止掛起
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 總結 60 秒超時
        )

        # 使用結構化章節格式化總結
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # 總結期間超時 - 返回原始內容
        logging.warning(
            "Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # 總結期間的其他錯誤 - 記錄並返回原始內容
        logging.warning(
            f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# 反思工具實用函數
##########################


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """用於研究進度和決策制定戰略反思的工具。

    在每次搜尋後使用此工具來系統地分析結果並規劃下一步。
    這在研究工作流中創建了一個故意的暫停，以便進行品質決策。

    何時使用：
    - 收到搜尋結果後：我找到了什麼關鍵資訊？
    - 決定下一步之前：我有足夠的資訊來全面回答嗎？
    - 評估研究缺口時：我還缺少什麼具體資訊？
    - 結束研究之前：我現在可以提供完整的答案嗎？

    反思應該解決：
    1. 當前發現的分析 - 我收集了什麼具體資訊？
    2. 缺口評估 - 還缺少什麼關鍵資訊？
    3. 品質評估 - 我有足夠的證據/示例來提供好的答案嗎？
    4. 戰略決策 - 我應該繼續搜尋還是提供我的答案？

    Args:
        reflection: 您對研究進度、發現、缺口和下一步的詳細反思

    Returns:
        確認反思已記錄用於決策制定
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP 實用函數
##########################


async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """使用 OAuth token 交換將 Supabase token 交換為 MCP 訪問 token。

    Args:
        supabase_token: 有效的 Supabase 認證 token
        base_mcp_url: MCP 伺服器的基礎 URL

    Returns:
        成功時返回 token 數據字典，失敗時返回 None
    """
    try:
        # 準備 OAuth token 交換請求數據
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # 執行 token 交換請求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功獲得 token
                    token_data = await response.json()
                    return token_data
                else:
                    # 記錄錯誤詳情用於調試
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")

    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    """檢索存儲的認證 token 並進行過期驗證。

    Args:
        config: 包含線程和使用者標識符的運行時配置

    Returns:
        如果有效且未過期則返回 token 字典，否則返回 None
    """
    store = get_store()

    # 從配置中提取必需的標識符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    # 檢索存儲的 token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    # 檢查 token 過期
    expires_in = tokens.value.get("expires_in")  # 過期前的秒數
    created_at = tokens.created_at  # token 創建的日期時間
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # Token 已過期，清理並返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """在配置存儲中存儲認證 token。

    Args:
        config: 包含線程和使用者標識符的運行時配置
        tokens: 要存儲的 token 字典
    """
    store = get_store()

    # 從配置中提取必需的標識符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    # 存儲 token
    await store.aput((user_id, "tokens"), "data", tokens)


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """獲取並刷新 MCP token，如需要則獲取新的。

    Args:
        config: 包含認證詳情的運行時配置

    Returns:
        有效的 token 字典，或無法獲取 token 時返回 None
    """
    # 首先嘗試獲取現有的有效 token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # 提取 Supabase token 用於新的 token 交換
    supabase_token = config.get("configurable", {}).get(
        "x-supabase-access-token")
    if not supabase_token:
        return None

    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # 將 Supabase token 交換為 MCP token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存儲新 token 並返回它們
    await set_tokens(config, mcp_tokens)
    return mcp_tokens


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """用全面的認證和錯誤處理包裝 MCP 工具。

    Args:
        tool: 要包裝的 MCP 結構化工具

    Returns:
        具有認證錯誤處理的增強工具
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """具有 MCP 錯誤處理和使用者友好訊息的增強協程。"""

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在異常鏈中遞歸搜尋 MCP 錯誤。"""
            if isinstance(exc, McpError):
                return exc

            # 通過檢查屬性處理 ExceptionGroup (Python 3.11+)
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # 執行原始工具功能
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # 在異常鏈中搜尋 MCP 特定錯誤
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 錯誤，重新拋出原始異常
                raise original_error

            # 處理 MCP 特定錯誤情況
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # 檢查認證/需要互動錯誤
            if error_code == -32003:  # 需要互動錯誤代碼
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"

                # 如果可用則提取使用者友好訊息
                if isinstance(message_payload, dict):
                    error_message = message_payload.get(
                        "text") or error_message

                # 如果提供了 URL 則附加以供使用者參考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # 對於其他 MCP 錯誤，重新拋出原始錯誤
            raise original_error

    # 用我們的增強版本替換工具的協程
    tool.coroutine = authentication_wrapper
    return tool


async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """載入並配置具有認證的 MCP（模型上下文協議）工具。

    Args:
        config: 包含 MCP 伺服器詳情的運行時配置
        existing_tool_names: 已使用的工具名稱集合，以避免衝突

    Returns:
        準備使用的已配置 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)

    # 步驟 1：如需要則處理認證
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    # 步驟 2：驗證配置要求
    config_valid = (
        configurable.mcp_config and
        configurable.mcp_config.url and
        configurable.mcp_config.tools and
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )

    if not config_valid:
        return []

    # 步驟 3：設置 MCP 伺服器連接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"

    # 如果 token 可用則配置認證標頭
    auth_headers = None
    if mcp_tokens:
        auth_headers = {
            "Authorization": f"Bearer {mcp_tokens['access_token']}"}

    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: 當 OAP 中合併多 MCP 伺服器支援時，更新此代碼

    # 步驟 4：從 MCP 伺服器載入工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # 如果 MCP 伺服器連接失敗，返回空列表
        return []

    # 步驟 5：過濾和配置工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳過具有衝突名稱的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue

        # 僅包含配置中指定的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue

        # 用認證處理包裝工具並添加到列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    return configured_tools


##########################
# 工具實用函數
##########################

async def get_search_tool(search_api: SearchAPI):
    """根據指定的 API 提供者配置並返回搜尋工具。

    Args:
        search_api: 要使用的搜尋 API 提供者（Anthropic、OpenAI、Tavily 或 None）

    Returns:
        指定提供者的已配置搜尋工具對象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic 的原生網頁搜尋，具有使用限制
        return [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]

    elif search_api == SearchAPI.OPENAI:
        # OpenAI 的網頁搜尋預覽功能
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.TAVILY:
        # 使用元數據配置 Tavily 搜尋工具
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # 未配置搜尋功能
        return []

    # 未知搜尋 API 類型的預設回退
    return []


async def get_all_tools(config: RunnableConfig):
    """組裝完整的工具包，包括研究、搜尋和 MCP 工具。

    Args:
        config: 指定搜尋 API 和 MCP 設定的運行時配置

    Returns:
        研究操作的所有已配置和可用工具列表
    """
    # 從核心研究工具開始
    tools = [tool(ResearchComplete), think_tool]

    # 添加已配置的搜尋工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # 追蹤現有工具名稱以防止衝突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in tools
    }

    # 如果已配置則添加 MCP 工具
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """從工具調用訊息中提取筆記。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型提供者原生網頁搜尋實用函數
##########################


def anthropic_websearch_called(response):
    """檢測回應中是否使用了 Anthropic 的原生網頁搜尋。

    Args:
        response: 來自 Anthropic API 的回應對象

    Returns:
        如果調用了網頁搜尋則返回 True，否則返回 False
    """
    try:
        # 導航回應元數據結構
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # 檢查伺服器端工具使用資訊
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # 查找網頁搜尋請求計數
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # 如果進行了任何網頁搜尋請求則返回 True
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # 處理回應結構意外的情況
        return False


def openai_websearch_called(response):
    """檢測回應中是否使用了 OpenAI 的網頁搜尋功能。

    Args:
        response: 來自 OpenAI API 的回應對象

    Returns:
        如果調用了網頁搜尋則返回 True，否則返回 False
    """
    # 檢查回應元數據中的工具輸出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # 在工具輸出中查找網頁搜尋調用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# Token 限制超出實用函數
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """確定異常是否表示超出了 token/上下文限制。

    Args:
        exception: 要分析的異常
        model_name: 可選的模型名稱，用於優化提供者檢測

    Returns:
        如果異常表示超出了 token 限制則返回 True，否則返回 False
    """
    error_str = str(exception).lower()

    # 步驟 1：如果可用則從模型名稱確定提供者
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'

    # 步驟 2：檢查提供者特定的 token 限制模式
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # 步驟 3：如果提供者未知，檢查所有提供者
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """檢查異常是否表示 OpenAI token 限制超出。"""
    # 分析異常元數據
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 檢查這是否是 OpenAI 異常
    is_openai_exception = (
        'openai' in exception_type.lower() or
        'openai' in module_name.lower()
    )

    # 檢查典型的 OpenAI token 限制錯誤類型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']

    if is_openai_exception and is_request_error:
        # 在錯誤訊息中查找與 token 相關的關鍵字
        token_keywords = ['token', 'context',
                          'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # 檢查特定的 OpenAI 錯誤代碼
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')

        if (error_code == 'context_length_exceeded' or
                error_type == 'invalid_request_error'):
            return True

    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """檢查異常是否表示 Anthropic token 限制超出。"""
    # 分析異常元數據
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 檢查這是否是 Anthropic 異常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or
        'anthropic' in module_name.lower()
    )

    # 檢查 Anthropic 特定的錯誤模式
    is_bad_request = class_name == 'BadRequestError'

    if is_anthropic_exception and is_bad_request:
        # Anthropic 對 token 限制使用特定的錯誤訊息
        if 'prompt is too long' in error_str:
            return True

    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """檢查異常是否表示 Google/Gemini token 限制超出。"""
    # 分析異常元數據
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 檢查這是否是 Google/Gemini 異常
    is_google_exception = (
        'google' in exception_type.lower() or
        'google' in module_name.lower()
    )

    # 檢查 Google 特定的資源耗盡錯誤
    is_resource_exhausted = class_name in [
        'ResourceExhausted',
        'GoogleGenerativeAIFetchError'
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # 檢查特定的 Google API 資源耗盡模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True

    return False


# 注意：這可能已過時或不適用於您的模型。請根據需要更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}


def get_model_token_limit(model_string):
    """查找特定模型的 token 限制。

    Args:
        model_string: 要查找的模型標識符字串

    Returns:
        如果找到則返回整數 token 限制，如果模型不在查找表中則返回 None
    """
    # 搜尋已知的模型 token 限制
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # 在查找表中未找到模型
    return None


def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通過移除到最後一個 AI 訊息來截斷訊息歷史。

    這對於通過移除最近的上下文來處理 token 限制超出錯誤很有用。

    Args:
        messages: 要截斷的訊息對象列表

    Returns:
        截斷到（但不包括）最後一個 AI 訊息的訊息列表
    """
    # 向後搜尋訊息以找到最後一個 AI 訊息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回到（但不包括）最後一個 AI 訊息的所有內容
            return messages[:i]

    # 未找到 AI 訊息，返回原始列表
    return messages

##########################
# 雜項實用函數
##########################


def get_today_str() -> str:
    """獲取當前日期，格式化為在提示和輸出中顯示。

    Returns:
        人類可讀的日期字串，格式如 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_config_value(value):
    """從配置中提取值，處理枚舉和 None 值。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """從環境或配置中獲取特定模型的 API 金鑰。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None


def get_tavily_api_key(config: RunnableConfig):
    """從環境或配置中獲取 Tavily API 金鑰。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
