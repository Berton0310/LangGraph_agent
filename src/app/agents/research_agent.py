"""
研究代理
負責分析用戶訊息並在需要時詢問澄清問題
"""
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration, get_api_key_for_model, get_llm_with_structured_output
from ..state.research_state import ClarifyWithUser, ReportPlan
from ..state.agent_state import AgentState
from ..tools.simple_tools import tavily_search_tool


# 研究計劃生成指令
RESEARCH_PLAN_INSTRUCTIONS = """
你是一個研究經理，管理著一個研究代理團隊。今天的日期是 {date}。
給定一個研究查詢，你的工作是產生報告的初始大綱（章節標題和關鍵問題），
以及一些背景脈絡。每個章節將分配給團隊中的不同研究員，他們將對該章節進行研究。

你將獲得：
- 初始研究查詢

你的任務是：
1. 通過執行網路搜尋或爬取網站，產生1-2段初始背景脈絡（如需要）
2. 產生報告大綱，包括章節標題列表和每個章節要解決的關鍵問題
3. 提供報告標題，將用作主要標題

指導原則：
- 每個章節應涵蓋一個獨立於其他章節的單一主題/問題
- 每個章節的關鍵問題應包括名稱和域名/網站（如果可用且適用），如果與公司、產品或類似事物相關
- 背景脈絡不應超過2段
- 背景脈絡應非常具體地針對查詢，並包括與報告所有章節的研究員相關的任何資訊
- 背景脈絡應僅來自網路搜尋或爬取結果，而不是先驗知識（即只有在調用工具時才應包含）
- 例如，如果查詢是關於一家公司，背景脈絡應包括關於該公司做什麼的一些基本資訊
- 不要進行超過2次工具調用

只輸出JSON。遵循下面的JSON架構。不要輸出任何其他內容。我將使用Pydantic解析此內容，所以只輸出有效的JSON：
{json_schema}
"""


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> ClarifyWithUser:
    """分析用戶訊息並提供澄清問題和研究方向選項。

    此函數統一執行澄清分析，始終提供澄清問題和3個研究方向選項。
    不再根據 need_clarification 來決定是否澄清。

    Args:
        state: 包含用戶訊息的當前代理狀態
        config: 包含模型設置和偏好的運行時配置

    Returns:
        包含澄清分析結果的字典
    """
    # 步驟 1：獲取配置
    configurable = Configuration.from_runnable_config(config)

    # 步驟 2：為結構化澄清分析準備模型
    messages = state["messages"]  # 支援字典和對象訪問
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置具有結構化輸出和重試邏輯的模型
    clarification_model = get_llm_with_structured_output(
        configurable.research_model,
        ClarifyWithUser  # 定義回傳內容格式
    )

    # 步驟 3：生成澄清問題和選項
    # 建立澄清提示
    clarify_prompt = f"""
    你是一個研究分析師，需要為用戶的研究請求提供澄清問題和研究方向選項。

    用戶的請求：
    {get_buffer_string(messages)}

    請為這個請求提供一個澄清問題和3個具體的研究方向選項，幫助用戶更明確地定義研究範圍。

    始終提供澄清問題和3個研究方向選項，無論請求是否明確。
    """

    response = await clarification_model.ainvoke([HumanMessage(content=clarify_prompt)])

    # 步驟 4：返回澄清分析結果（統一設置為需要澄清）
    return {
        "question": response.question,
        "options": response.options,
        "verification": response.verification
    }


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """生成研究簡報和報告計劃。

    此函數根據用戶查詢生成詳細的研究計劃，包括背景脈絡、報告大綱和標題。
    會使用 Tavily 搜尋工具獲取相關的背景資訊。

    Args:
        state: 包含用戶訊息的當前代理狀態
        config: 包含模型設置和偏好的運行時配置

    Returns:
        包含研究計劃結果的字典
    """
    # 步驟 1：獲取配置和用戶訊息
    configurable = Configuration.from_runnable_config(config)
    messages = state["messages"]  # 支援字典和對象訪問
    user_query = get_buffer_string(messages)

    # 步驟 2：使用 Tavily 工具進行背景搜尋
    search_results = []
    try:
        # 使用 @tool 裝飾器的簡化方式
        search_results = tavily_search_tool.invoke({
            "query": user_query,
            "max_results": 3,
            "search_depth": "basic"
        })
        print(f"🔍 Tavily 搜尋完成，找到 {len(search_results)} 個結果")
    except Exception as e:
        print(f"⚠️ Tavily 搜尋失敗: {e}")
        search_results = []

    # 步驟 3：準備模型配置
    model_config = {
        "model": configurable.plan_model,
        "max_tokens": configurable.plan_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.plan_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 步驟 4：配置具有結構化輸出的模型
    plan_model = get_llm_with_structured_output(
        configurable.plan_model,
        ReportPlan
    )

    # 步驟 5：準備包含搜尋結果的提示
    prompt_content = RESEARCH_PLAN_INSTRUCTIONS.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        json_schema=ReportPlan.model_json_schema()
    )

    # 添加搜尋結果到提示中
    search_context = ""
    if search_results and not any("error" in result for result in search_results):
        search_context = "\n\n搜尋到的背景資訊：\n"
        for i, result in enumerate(search_results, 1):
            search_context += f"{i}. {result.get('title', '無標題')}\n"
            search_context += f"   內容: {result.get('content', '無內容')[:200]}...\n"
            search_context += f"   來源: {result.get('url', '無URL')}\n\n"

    full_prompt = f"{prompt_content}\n\n研究查詢：\n{user_query}{search_context}"

    # 步驟 6：生成研究計劃
    response = await plan_model.ainvoke([HumanMessage(content=full_prompt)])

    # 步驟 7：返回研究計劃結果
    return {
        "background_context": response.background_context,
        "report_outline": [section.model_dump() for section in response.report_outline],
        "report_title": response.report_title,
        "search_results": search_results  # 包含搜尋結果供後續使用
    }


# 輔助函數
def get_buffer_string(messages) -> str:
    """將訊息列表轉換為字串"""
    return "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])


def get_today_str() -> str:
    """獲取今天的日期字串"""
    return datetime.now().strftime("%Y-%m-%d")


# 測試函數
async def test_research_agent():
    """測試研究代理功能"""
    from langchain_core.messages import HumanMessage

    # 模擬狀態
    test_state = {
        "messages": [HumanMessage(content="我想了解人工智慧在醫療領域的最新應用")]
    }

    # 模擬配置
    test_config = {
        "configurable": {
            "allow_clarification": True,
            "research_model": "gemini-2.5-pro",
            "plan_model": "gemini-2.5-pro",
            "research_model_max_tokens": 10000,
            "plan_model_max_tokens": 8192
        }
    }

    print("🧪 測試研究代理")
    print("=" * 40)

    # 測試澄清功能
    print("1. 測試澄清功能...")
    try:
        clarify_result = await clarify_with_user(test_state, test_config)
        print(f"   需要澄清: {clarify_result['need_clarification']}")
        if clarify_result['need_clarification']:
            print(f"   澄清問題: {clarify_result['question']}")
            print(f"   選項: {clarify_result['options']}")
        else:
            print(f"   確認訊息: {clarify_result['verification']}")
    except Exception as e:
        print(f"   澄清測試失敗: {e}")

    print("\n2. 測試研究計劃生成...")
    try:
        plan_result = await write_research_brief(test_state, test_config)
        print(f"   報告標題: {plan_result['report_title']}")
        print(f"   背景脈絡: {plan_result['background_context'][:100]}...")
        print(f"   章節數量: {len(plan_result['report_outline'])}")
        print(f"   搜尋結果數量: {len(plan_result.get('search_results', []))}")
    except Exception as e:
        print(f"   計劃生成測試失敗: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_research_agent())
