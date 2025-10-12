"""
測試 clarify_with_user 函數
"""
import asyncio
from langchain_core.messages import HumanMessage
from src.app.agents.research_agent import clarify_with_user


async def test_clarify_with_user():
    """測試 clarify_with_user 函數"""
    print("🧪 測試 clarify_with_user 函數")
    print("=" * 50)

    # 測試案例 1：明確的查詢
    print("測試案例 1：明確的查詢")
    test_state_1 = {
        "messages": [HumanMessage(content="我想了解人工智慧在醫療領域的最新應用，特別是診斷和治療方面的進展")]
    }

    test_config = {
        "configurable": {
            "allow_clarification": True,
            "research_model": "gemini-2.5-pro",
            "research_model_max_tokens": 10000
        }
    }

    result_1 = await clarify_with_user(test_state_1, test_config)

    print(f"   澄清問題: {result_1['question']}")
    print(f"   選項: {result_1['options']}")

    print("\n" + "-" * 50)
    # 測試案例 2：模糊的查詢
    print("測試案例 2：模糊的查詢")
    test_state_2 = {
        "messages": [HumanMessage(content="我想了解一些技術")]
    }

    result_2 = await clarify_with_user(test_state_2, test_config)

    print(f"   澄清問題: {result_2['question']}")
    print(f"   選項: {result_2['options']}")
    e

    print("\n" + "-" * 50)

    # 測試案例 3：技術相關查詢
    print("測試案例 3：技術相關查詢")
    test_state_3 = {
        "messages": [HumanMessage(content="我想了解機器學習的應用")]
    }

    result_3 = await clarify_with_user(test_state_3, test_config)

    print(f"   澄清問題: {result_3['question']}")
    print(f"   選項: {result_3['options']}")


if __name__ == "__main__":
    asyncio.run(test_clarify_with_user())
