#!/usr/bin/env python3
"""
詳細的環境和模型測試
"""

import asyncio
import sys
import os
from pathlib import Path

# 載入 .env 檔案
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv 未安裝，請運行: pip install python-dotenv")

# 添加src目錄到Python路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def check_environment():
    """檢查環境設置"""
    print("🔧 檢查環境設置")
    print("=" * 50)

    # 檢查環境變數
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    print(f"GEMINI_API_KEY: {'✅ 已設置' if gemini_key else '❌ 未設置'}")
    print(f"TAVILY_API_KEY: {'✅ 已設置' if tavily_key else '⚠️  未設置 (可選)'}")

    if gemini_key:
        print(f"   金鑰長度: {len(gemini_key)} 字符")
        print(f"   金鑰前綴: {gemini_key[:10]}...")

    return bool(gemini_key)


async def test_gemini_model():
    """測試 Gemini 模型"""
    print("\n🧪 測試 Gemini 模型")
    print("=" * 50)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY 未設置")
            return False

        # 創建模型
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            max_output_tokens=1000,
            google_api_key=api_key
        )
        print("✅ Gemini 模型創建成功")

        # 測試簡單對話
        print("📝 發送測試訊息...")
        response = await model.ainvoke([
            HumanMessage(content="請用一句話介紹人工智慧")
        ])

        print("✅ 模型回應成功:")
        print(f"   {response.content}")

        return True

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_research_agent():
    """測試研究代理"""
    print("\n🔍 測試研究代理")
    print("=" * 50)

    try:
        from app.agents.research_agent import deep_researcher
        from app.config import Configuration
        from langchain_core.messages import HumanMessage

        # 創建配置
        config = Configuration()
        print(f"✅ 配置載入成功")
        print(f"   研究模型: {config.research_model}")

        # 創建運行配置
        runnable_config = {
            "configurable": {
                "search_api": config.search_api.value,
                "max_researcher_iterations": 1,  # 限制迭代次數
                "max_concurrent_research_units": 1,
                "research_model": config.research_model,
                "compression_model": config.compression_model,
                "final_report_model": config.final_report_model,
                "summarization_model": config.summarization_model,
                "plan_model": config.plan_model,
                "api_key": os.getenv("GEMINI_API_KEY")
            }
        }

        # 測試簡單的研究主題
        test_topic = "什麼是人工智慧？"
        print(f"🔍 測試主題: {test_topic}")

        # 運行研究代理
        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=test_topic)]},
            config=runnable_config
        )

        print("✅ 研究代理測試成功!")
        if "final_report" in result:
            print("📄 研究報告:")
            print(result["final_report"])

        return True

    except Exception as e:
        print(f"❌ 研究代理測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函數"""
    print("🚀 深度研究代理完整測試")
    print("=" * 60)

    # 檢查環境
    env_ok = check_environment()
    if not env_ok:
        print("\n❌ 環境設置不完整，請先設置 GEMINI_API_KEY")
        print("   例如: set GEMINI_API_KEY=your_api_key_here")
        return

    # 測試 Gemini 模型
    model_ok = await test_gemini_model()
    if not model_ok:
        print("\n❌ Gemini 模型測試失敗")
        return

    # 測試研究代理
    agent_ok = await test_research_agent()
    if not agent_ok:
        print("\n❌ 研究代理測試失敗")
        return

    print("\n🎉 所有測試都通過了！系統準備就緒！")

if __name__ == "__main__":
    asyncio.run(main())
