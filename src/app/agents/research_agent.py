"""深度研究代理的主要 LangGraph 實作。"""

import asyncio
from typing import Literal
import time
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..config import (
    Configuration,
)
from ..prompt import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from ..state.research_state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from ..utils.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化一個可在整個代理中使用的可配置模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens"),
)


def print_progress(message: str, level: int = 0):
    """打印進度訊息"""
    # timestamp = datetime.now().strftime("%H:%M:%S")
    indent = "  " * level
    formatted_message = f" {indent}{message}"
    print(formatted_message)

    # 如果存在全域進度回調函數，則調用它
    if hasattr(print_progress, 'progress_callback') and print_progress.progress_callback:
        try:
            print_progress.progress_callback(formatted_message)
        except Exception:
            pass  # 忽略回調錯誤，避免影響主要流程


def _append_timing(state: dict, label: str, start_time: float) -> list[str]:
    """將單步耗時（秒）附加到狀態中的 timings 並回傳最新列表。"""
    elapsed = time.perf_counter() - start_time
    timings = state.get("timings", []) or []

    # 為 supervisor 相關步驟添加迭代次數
    if label in ["supervisor", "supervisor_tools"]:
        iteration = state.get("research_iterations", 0)
        unique_label = f"{label}#{iteration}"
    else:
        unique_label = label

    timings = [*timings, f"{unique_label}: {elapsed:.2f}s"]
    return timings


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析使用者訊息，如果研究範圍不清楚則提出釐清問題。

    此函數決定使用者的請求在進行研究之前是否需要釐清。如果釐清被禁用或不需要，
    則直接進行研究。

    Args:
        state: 包含使用者訊息的當前代理狀態
        config: 包含模型設定和偏好的運行時配置

    Returns:
        以釐清問題結束或進行研究簡報的命令
    """
    # 標準化階段事件：使用者釐清階段
    print_progress("STAGE::clarify_with_user::enter")
    # print_progress("🔍 正在分析研究需求...")

    # 步驟 1：檢查配置中是否啟用釐清
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    # print_progress("✓ 已載入配置設定", 1)

    if not configurable.allow_clarification:
        # 跳過釐清步驟，直接進行研究
        # print_progress("✓ 釐清功能已禁用，跳過釐清步驟", 1)
        return Command(goto="write_research_brief")

    # 步驟 2：為結構化釐清分析準備模型
    # print_progress("✓ 正在準備釐清分析模型...", 1)
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    # 配置具有結構化輸出和重試邏輯的模型
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # 步驟 3：分析是否需要釐清
    # print_progress("✓ 正在分析使用者需求是否明確...", 1)
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # 步驟 4：根據釐清分析進行路由
    if response.need_clarification:
        # 以釐清問題結束給使用者
        # print_progress("✓ 需要釐清問題，等待使用者回應", 1)
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question)],
                "timings": _append_timing(state, "clarify_with_user", _start_t)
            }
        )
    else:
        # 以驗證訊息進行研究
        # print_progress("✓ 需求明確，準備進行研究", 1)
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "timings": _append_timing(state, "clarify_with_user", _start_t)
            }
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """將使用者訊息轉換為結構化研究簡報並初始化監督者。

    此函數分析使用者的訊息並生成一個專注的研究簡報，將指導研究監督者。
    它還使用適當的提示和指示設置初始監督者上下文。

    Args:
        state: 包含使用者訊息的當前代理狀態
        config: 包含模型設定的運行時配置

    Returns:
        進行研究監督者的命令，帶有初始化的上下文
    """
    # 標準化階段事件：研究規劃階段
    print_progress("STAGE::write_research_brief::enter")
    # print_progress("📋 正在制定研究計畫...")

    # 步驟 1：為結構化輸出設置研究模型
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    # print_progress("✓ 已載入研究配置", 1)

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    # 為結構化研究問題生成配置模型
    # print_progress("✓ 正在配置研究問題分析模型...", 1)
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步驟 2：從使用者訊息生成結構化研究簡報
    # print_progress("✓ 正在分析使用者訊息並生成研究簡報...", 1)
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    # print_progress("✓ 研究方向與架構已生成", 1)
    # 將研究簡報內容即時推送給前端
    # print_progress(f"BRIEF::{response.research_brief}")

    # 步驟 3：使用研究簡報和指示初始化監督者
    # print_progress("✓ 正在初始化研究監督者...", 1)
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    # print_progress("✓ 監督者已準備就緒", 1)

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            },
            "timings": _append_timing(state, "write_research_brief", _start_t)
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """領導研究監督者，規劃研究策略並委派給研究者。

    監督者分析研究簡報並決定如何將研究分解為可管理的任務。它可以使用 think_tool
    進行戰略規劃，使用 ConductResearch 將任務委派給子研究者，或在滿意發現時使用
    ResearchComplete。

    Args:
        state: 包含訊息和研究上下文的當前監督者狀態
        config: 包含模型設定的運行時配置

    Returns:
        進行 supervisor_tools 進行工具執行的命令
    """
    # 標準化階段事件：研究執行（監督者）階段，含迭代次數
    current_iteration = state.get("research_iterations", 0) + 1
    print_progress(
        f"STAGE::research_supervisor::enter::iteration={current_iteration}")
    # print_progress("🎯 正在規劃研究策略...")

    # 步驟 1：使用可用工具配置監督者模型
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    # print_progress("✓ 已載入監督者配置", 1)

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    # 可用工具：研究委派、完成信號和戰略思考
    # print_progress("✓ 正在配置可用工具：研究委派、完成信號、深度思考", 1)
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # 配置具有工具、重試邏輯和模型設定的模型
    # print_progress("✓ 正在綁定工具並設置重試邏輯...", 1)
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步驟 2：基於當前上下文生成監督者回應
    # print_progress("✓ 正在分析研究簡報並制定策略...", 1)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    # print_progress("✓ 研究策略已制定", 1)

    # 步驟 3：更新狀態並進行工具執行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "timings": _append_timing(state, "supervisor", _start_t)
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """執行監督者調用的工具，包括研究委派和戰略思考。

    此函數處理三種類型的監督者工具調用：
    1. think_tool - 繼續對話的戰略反思
    2. ConductResearch - 將研究任務委派給子研究者
    3. ResearchComplete - 信號研究階段完成

    Args:
        state: 包含訊息和迭代計數的當前監督者狀態
        config: 包含研究限制和模型設定的運行時配置

    Returns:
        繼續監督循環或結束研究階段的命令
    """
    # 步驟 1：提取當前狀態並檢查退出條件
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定義研究階段的退出標準
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果滿足任何終止條件則退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "timings": _append_timing(state, "supervisor_tools", _start_t)
            }
        )

    # 步驟 2：一起處理所有工具調用（think_tool 和 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 處理 think_tool 調用（戰略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 處理 ConductResearch 調用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        # print_progress(f"✓ 正在委派 {len(conduct_research_calls)} 個研究任務...", 1)
        try:
            # 限制並行研究單位以防止資源耗盡
            allowed_conduct_research_calls = conduct_research_calls[
                :configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[
                configurable.max_concurrent_research_units:]

            # 並行執行研究任務
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(
                            content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # 使用研究結果創建工具訊息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get(
                        "compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 處理溢出的研究調用並提供錯誤訊息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 聚合所有研究結果的原始筆記
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # 處理研究執行錯誤
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token 限制超出或其他錯誤 - 結束研究階段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    # 步驟 3：返回包含所有工具結果的命令
    update_payload["supervisor_messages"] = all_tool_messages
    update_payload["timings"] = _append_timing(
        state, "supervisor_tools", _start_t)
    return Command(
        goto="supervisor",
        update=update_payload
    )

# 監督者子圖構建
# 創建管理研究委派和協調的監督者工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 為研究管理添加監督者節點
supervisor_builder.add_node("supervisor", supervisor)           # 主要監督者邏輯
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具執行處理器

# 定義監督者工作流邊緣
supervisor_builder.add_edge(START, "supervisor")  # 監督者入口點

# 編譯監督者子圖以供主工作流使用
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """對特定主題進行專注研究的個別研究者。

    此研究者由監督者給予特定研究主題，並使用可用工具（搜尋、think_tool、MCP 工具）
    收集全面資訊。它可以在搜尋之間使用 think_tool 進行戰略規劃。

    Args:
        state: 包含訊息和主題上下文的當前研究者狀態
        config: 包含模型設定和工具可用性的運行時配置

    Returns:
        進行 researcher_tools 進行工具執行的命令
    """
    # print_progress("🔍 正在收集研究資料...")

    # 步驟 1：載入配置並驗證工具可用性
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    # print_progress("✓ 已載入研究者配置", 1)

    # 獲取所有可用的研究工具（搜尋、MCP、think_tool）
    # print_progress("✓ 正在載入研究工具...", 1)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        # print_progress("✗ 未找到可用的研究工具", 1)
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    # print_progress(f"✓ 已載入 {len(tools)} 個研究工具", 1)

    # 步驟 2：使用工具配置研究者模型
    # print_progress("✓ 正在配置研究者模型...", 1)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    # 準備系統提示，如果可用則包含 MCP 上下文
    # print_progress("✓ 正在準備研究系統提示...", 1)
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置具有工具、重試邏輯和設定的模型
    # print_progress("✓ 正在綁定工具並設置重試邏輯...", 1)
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 步驟 3：使用系統上下文生成研究者回應
    # print_progress("✓ 正在分析研究主題並制定搜尋策略...", 1)
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    # print_progress("✓ 研究策略已制定，準備執行工具", 1)

    # 步驟 4：更新狀態並進行工具執行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
            "timings": _append_timing(state, "researcher", _start_t)
        }
    )

# 工具執行輔助函數


async def execute_tool_safely(tool, args, config):
    """安全執行工具並處理錯誤。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """執行研究者調用的工具，包括搜尋工具和戰略思考。

    此函數處理各種類型的研究者工具調用：
    1. think_tool - 繼續研究對話的戰略反思
    2. 搜尋工具（tavily_search、web_search）- 資訊收集
    3. MCP 工具 - 外部工具整合
    4. ResearchComplete - 信號個別研究任務完成

    Args:
        state: 包含訊息和迭代計數的當前研究者狀態
        config: 包含研究限制和工具設定的運行時配置

    Returns:
        繼續研究循環或進行壓縮的命令
    """
    # 步驟 1：提取當前狀態並檢查早期退出條件
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 如果沒有進行工具調用則早期退出（包括原生網頁搜尋）
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or
        anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research", update={"timings": _append_timing(state, "researcher_tools", _start_t)})

    # 步驟 2：處理其他工具調用（搜尋、MCP 工具等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 並行執行所有工具調用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(
            tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 從執行結果創建工具訊息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 步驟 3：檢查後期退出條件（處理工具後）
    exceeded_iterations = state.get(
        "tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # 結束研究並進行壓縮
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "timings": _append_timing(state, "researcher_tools", _start_t)
            }
        )

    # 使用工具結果繼續研究循環
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "timings": _append_timing(state, "researcher_tools", _start_t)
        }
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """將研究發現壓縮並合成為簡潔、結構化的摘要。

    此函數獲取研究者的所有研究發現、工具輸出和 AI 訊息，並將它們蒸餾成
    清潔、全面的摘要，同時保留所有重要資訊和發現。

    Args:
        state: 包含累積研究訊息的當前研究者狀態
        config: 包含壓縮模型設定的運行時配置

    Returns:
        包含壓縮研究摘要和原始筆記的字典
    """
    # print_progress("📊 正在整理研究結果...")

    # 步驟 1：配置壓縮模型
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    # print_progress("✓ 已載入壓縮配置", 1)

    # print_progress("✓ 正在配置壓縮模型...", 1)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "tags": ["langsmith:nostream"]
    })

    # 步驟 2：為壓縮準備訊息
    # print_progress("✓ 正在準備研究資料進行壓縮...", 1)
    researcher_messages = state.get("researcher_messages", [])

    # 添加從研究模式切換到壓縮模式的指示
    researcher_messages.append(HumanMessage(
        content=compress_research_simple_human_message))

    # 步驟 3：嘗試壓縮，對 token 限制問題使用重試邏輯
    # print_progress("✓ 正在執行研究資料壓縮...", 1)
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 創建專注於壓縮任務的系統提示
            compression_prompt = compress_research_system_prompt.format(
                date=get_today_str())
            messages = [SystemMessage(
                content=compression_prompt)] + researcher_messages

            # 執行壓縮
            response = await synthesizer_model.ainvoke(messages)

            # 從所有工具和 AI 訊息中提取原始筆記
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # 返回成功的壓縮結果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                "timings": _append_timing(state, "compress_research", _start_t)
            }

        except Exception as e:
            synthesis_attempts += 1

            # 通過移除較舊的訊息處理 token 限制超出
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(
                    researcher_messages)
                continue

            # 對於其他錯誤，繼續重試
            continue

    # 步驟 4：如果所有嘗試都失敗則返回錯誤結果
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
        "timings": _append_timing(state, "compress_research", _start_t)
    }

# 研究者子圖構建
# 創建對特定主題進行專注研究的個別研究者工作流
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)

# 為研究執行和壓縮添加研究者節點
researcher_builder.add_node("researcher", researcher)                 # 主要研究者邏輯
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具執行處理器
researcher_builder.add_node("compress_research", compress_research)   # 研究壓縮

# 定義研究者工作流邊緣
researcher_builder.add_edge(START, "researcher")           # 研究者入口點
researcher_builder.add_edge("compress_research", END)      # 壓縮後退出點

# 編譯研究者子圖以供監督者並行執行
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """使用 token 限制的重試邏輯生成最終的全面研究報告。

    此函數獲取所有收集的研究發現，並使用配置的報告生成模型將它們合成為
    結構良好的全面最終報告。

    Args:
        state: 包含研究發現和上下文的代理狀態
        config: 包含模型設定和 API 金鑰的運行時配置

    Returns:
        包含最終報告和清除狀態的字典
    """
    # 標準化階段事件：報告生成階段
    print_progress("STAGE::final_report_generation::enter")
    # print_progress("📄 正在撰寫最終報告...")

    # 步驟 1：提取研究發現並準備狀態清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # print_progress(f"✓ 正在整合 {len(notes)} 項研究發現...", 1)

    # 步驟 2：配置最終報告生成模型
    _start_t = time.perf_counter()
    configurable = Configuration.from_runnable_config(config)
    # print_progress("✓ 已載入報告生成配置", 1)

    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    # 步驟 3：使用 token 限制重試邏輯嘗試報告生成
    # print_progress("✓ 正在配置報告生成模型...", 1)
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # 創建包含所有研究上下文的全面提示
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # 生成最終報告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # 返回成功的報告生成
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                "timings": _append_timing(state, "final_report_generation", _start_t),
                **cleared_state
            }

        except Exception as e:
            # 使用漸進式截斷處理 token 限制超出錯誤
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # 第一次重試：確定初始截斷限制
                    model_token_limit = get_model_token_limit(
                        configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # 使用 4 倍 token 限制作為截斷的字符近似值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 後續重試：每次減少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)

                # 截斷發現並重試
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token 限制錯誤：立即返回錯誤
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    "timings": _append_timing(state, "final_report_generation", _start_t),
                    **cleared_state
                }

    # 步驟 4：如果所有重試都用盡則返回失敗結果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        "timings": _append_timing(state, "final_report_generation", _start_t),
        **cleared_state
    }

# 主要深度研究者圖構建
# 創建從使用者輸入到最終報告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# 為完整研究過程添加主要工作流節點
deep_researcher_builder.add_node(
    "clarify_with_user", clarify_with_user)           # 使用者釐清階段
deep_researcher_builder.add_node(
    "write_research_brief", write_research_brief)     # 研究規劃階段
deep_researcher_builder.add_node(
    "research_supervisor", supervisor_subgraph)       # 研究執行階段
deep_researcher_builder.add_node(
    "final_report_generation", final_report_generation)  # 報告生成階段

# 定義順序執行的主要工作流邊緣
deep_researcher_builder.add_edge(
    START, "clarify_with_user")                       # 入口點
deep_researcher_builder.add_edge(
    "research_supervisor", "final_report_generation")  # 研究到報告
deep_researcher_builder.add_edge(
    "final_report_generation", END)                   # 最終退出點

# 編譯完整的深度研究者工作流
deep_researcher = deep_researcher_builder.compile()
