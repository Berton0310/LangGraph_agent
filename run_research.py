#!/usr/bin/env python3
"""
簡化的 Open Deep Research 測試腳本
使用 Gemini 2.5 Pro 模型
"""

import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from pprint import pprint, pformat
from src.app.config import Configuration, SearchAPI
from src.app.agents.research_agent import deep_researcher
import uuid

# 載入環境變數
load_dotenv()


async def test_research():
    """測試研究流程"""

    print("🚀 開始測試 Open Deep Research 完整流程")
    print("使用模型: Gemini 2.5 Pro")
    print("=" * 50)

    # 創建配置
    config = Configuration(
        research_model="gemini-2.5-pro",
        summarization_model="gemini-2.5-pro",
        compression_model="gemini-2.5-pro",
        final_report_model="gemini-2.5-pro",
        search_api=SearchAPI.TAVILY,
        max_concurrent_research_units=2,  # 減少並發數以加快測試
        allow_clarification=False,  # 跳過澄清步驟以加快測試
        max_researcher_iterations=3,  # 減少迭代次數
        max_react_tool_calls=5  # 減少工具調用次數
    )

    # 設置配置
    run_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            **config.model_dump(mode='json')
        }
    }

    # 簡單的研究問題
    question = "請簡要研究人工智慧在醫療診斷方面的最新應用"

    print(f"📋 研究問題: {question}")
    print("⏳ 開始研究...")

    try:
        start_dt = datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        # 執行研究
        result = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            run_config
        )

        print("✅ 研究完成！")

        print(f"結果包含: {list(result.keys())}")

        # 顯示結果摘要
        if "messages" in result:
            print("\n📝 研究過程摘要:")
            print("-" * 30)
            for i, msg in enumerate(result["messages"]):
                if hasattr(msg, 'content'):
                    content = msg.content[:100] + \
                        "..." if len(msg.content) > 100 else msg.content
                    print(f"{i+1}. [{type(msg).__name__}]: {content}")
                    print()

        # 顯示研究筆記
        if "notes" in result and result["notes"]:
            print("📚 研究筆記:")
            print("-" * 30)
            for note in result["notes"]:
                print(f"- {note[:150]}...")
                print()

        end_time = time.perf_counter()
        end_dt = datetime.now().isoformat(timespec="seconds")
        elapsed = end_time - start_time
        print("🎉 測試完成！")
        print(f"⏱️ 執行時間: {elapsed:.2f} 秒 (開始: {start_dt}, 結束: {end_dt})")
        print("\n🔎 完整 result 內容（可能較長）:")
        try:
            pprint(result, depth=3, width=100)
        except Exception:
            print(result)
        # 將完整結果寫入檔案
        try:
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"started_at: {start_dt}\nended_at: {end_dt}\nelapsed_seconds: {elapsed:.2f}\n")
                f.write("\n===== result (pretty) =====\n")
                f.write(pformat(result, depth=5, width=120))
                if isinstance(result, dict) and "final_report" in result:
                    f.write("\n\n===== final_report =====\n")
                    f.write(str(result["final_report"]))
            print("\n💾 已將完整結果寫入 output.txt")
        except Exception as write_err:
            print(f"\n⚠️ 寫入 output.txt 失敗: {write_err}")
        return result
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_research())
