"""
model.invoke 搭配 think_tool 使用範例 - 結果輸出
"""
from src.app.tools.mcp_tools import think_tool
from src.app.config import get_llm_with_structured_output
import sys
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage

# 添加專案路徑
sys.path.append(
    'C:\\Users\\berto\\Desktop\\capstone project\\my_langgraph_agent')


async def demonstrate_model_think_integration():
    """演示 model.invoke 與 think_tool 的整合使用"""

    print("🚀 model.invoke 搭配 think_tool 使用範例")
    print("=" * 60)

    # 範例 1: 基本整合
    print("\n📝 範例 1: 基本整合 - 人工智慧應用分析")
    print("-" * 50)

    try:
        # 步驟 1: 模型調用
        model = get_llm_with_structured_output("gemini-2.5-pro")
        prompt = "請分析人工智慧在醫療領域的主要應用和發展趨勢"

        response = await model.ainvoke([HumanMessage(content=prompt)])
        print(f"✅ 模型回應: {len(response.content)} 字符")

        # 步驟 2: 使用 think_tool 反思
        reflection = f"""
分析結果評估：
- 回應長度: {len(response.content)} 字符
- 內容涵蓋: 醫療AI應用和趨勢
- 結構完整性: 包含主要應用領域

品質評估：
- 資訊完整性: 涵蓋主要應用
- 實用性: 提供實用信息
- 專業性: 技術分析準確

改進建議：
- 可以添加更多具體案例
- 需要補充統計數據
- 可以增加風險分析
"""

        think_result = think_tool.invoke({"reflection": reflection})
        print(f"✅ 反思完成: {think_result[:50]}...")

        # 步驟 3: 基於反思的改進
        improvement_prompt = f"""
基於以下分析和反思，提供改進建議：

原始分析：
{response.content[:300]}...

反思要點：
{reflection}

請提供：
1. 具體的改進建議
2. 需要補充的內容
3. 下一步行動計劃
"""

        improvement = await model.ainvoke([HumanMessage(content=improvement_prompt)])
        print(f"✅ 改進建議: {len(improvement.content)} 字符")

        example_1_result = {
            "original_length": len(response.content),
            "reflection_length": len(reflection),
            "improvement_length": len(improvement.content),
            "total_length": len(response.content) + len(improvement.content)
        }

    except Exception as e:
        print(f"❌ 範例 1 失敗: {e}")
        example_1_result = None

    # 範例 2: 結構化輸出整合
    print("\n📝 範例 2: 結構化輸出整合 - 研究計劃生成")
    print("-" * 50)

    try:
        from src.app.state.research_state import ClarifyWithUser

        # 步驟 1: 結構化澄清
        structured_model = get_llm_with_structured_output(
            "gemini-2.5-pro", ClarifyWithUser)
        clarify_prompt = "我想了解區塊鏈技術的發展現狀"

        clarify_response = await structured_model.ainvoke([HumanMessage(content=clarify_prompt)])
        print(f"✅ 澄清問題: {clarify_response.question[:50]}...")
        print(f"✅ 選項數量: {len(clarify_response.options)}")

        # 步驟 2: 澄清反思
        clarify_reflection = f"""
澄清分析：
- 問題針對性: 強
- 選項數量: {len(clarify_response.options)}
- 選項內容: {clarify_response.options}

品質評估：
- 問題清晰度: 高
- 選項相關性: 強
- 用戶引導性: 好

戰略決策：
- 建議用戶選擇最感興趣的選項
- 可以基於選項進行深度研究
- 需要進一步澄清具體需求
"""

        clarify_think = think_tool.invoke({"reflection": clarify_reflection})
        print(f"✅ 澄清反思完成")

        example_2_result = {
            "question_length": len(clarify_response.question),
            "options_count": len(clarify_response.options),
            "reflection_length": len(clarify_reflection)
        }

    except Exception as e:
        print(f"❌ 範例 2 失敗: {e}")
        example_2_result = None

    # 範例 3: 多輪反思工作流程
    print("\n📝 範例 3: 多輪反思工作流程 - 深度分析")
    print("-" * 50)

    try:
        # 初始分析
        initial_prompt = "請分析量子計算的技術原理和應用前景"
        initial_response = await model.ainvoke([HumanMessage(content=initial_prompt)])
        print(f"✅ 初始分析: {len(initial_response.content)} 字符")

        # 第一輪反思
        first_reflection = f"""
初始分析評估：
- 分析長度: {len(initial_response.content)} 字符
- 內容深度: 包含技術原理和應用
- 完整性: 涵蓋主要方面

缺口識別：
- 需要更多技術細節
- 缺少實際應用案例
- 需要補充市場分析
"""

        first_think = think_tool.invoke({"reflection": first_reflection})
        print(f"✅ 第一輪反思完成")

        # 基於反思的深度分析
        depth_prompt = f"""
基於以下初始分析和反思，提供更深入的分析：

初始分析：
{initial_response.content[:400]}...

反思要點：
{first_reflection}

請提供：
1. 更詳細的技術原理說明
2. 具體的應用案例
3. 市場規模和投資情況
4. 未來發展預測
"""

        depth_response = await model.ainvoke([HumanMessage(content=depth_prompt)])
        print(f"✅ 深度分析: {len(depth_response.content)} 字符")

        # 第二輪反思
        second_reflection = f"""
深度分析評估：
- 分析長度: {len(depth_response.content)} 字符
- 內容深度: 包含技術細節和案例
- 完整性: 涵蓋市場和預測

品質提升：
- 技術細節更豐富
- 包含實際案例
- 提供市場分析
- 包含未來預測

最終評估：
- 分析深度: 優秀
- 實用價值: 高
- 專業程度: 高
"""

        second_think = think_tool.invoke({"reflection": second_reflection})
        print(f"✅ 第二輪反思完成")

        example_3_result = {
            "initial_length": len(initial_response.content),
            "depth_length": len(depth_response.content),
            "total_length": len(initial_response.content) + len(depth_response.content),
            "reflection_rounds": 2
        }

    except Exception as e:
        print(f"❌ 範例 3 失敗: {e}")
        example_3_result = None

    # 輸出總結
    print("\n📊 範例運行總結")
    print("=" * 60)

    if example_1_result:
        print("✅ 範例 1 (基本整合): 成功")
        print(f"   - 原始回應: {example_1_result['original_length']} 字符")
        print(f"   - 改進建議: {example_1_result['improvement_length']} 字符")
        print(f"   - 總長度: {example_1_result['total_length']} 字符")

    if example_2_result:
        print("✅ 範例 2 (結構化輸出): 成功")
        print(f"   - 澄清問題: {example_2_result['question_length']} 字符")
        print(f"   - 選項數量: {example_2_result['options_count']}")
        print(f"   - 反思長度: {example_2_result['reflection_length']} 字符")

    if example_3_result:
        print("✅ 範例 3 (多輪反思): 成功")
        print(f"   - 初始分析: {example_3_result['initial_length']} 字符")
        print(f"   - 深度分析: {example_3_result['depth_length']} 字符")
        print(f"   - 總長度: {example_3_result['total_length']} 字符")
        print(f"   - 反思輪數: {example_3_result['reflection_rounds']}")

    print("\n💡 使用建議:")
    print("1. model.invoke 提供靈活的模型調用能力")
    print("2. think_tool 提供戰略反思和決策支持")
    print("3. 結構化輸出確保數據格式一致性")
    print("4. 多輪反思可以顯著提升輸出品質")
    print("5. 組合使用可以實現完整的工作流程")

    print("\n🎉 範例演示完成！")


if __name__ == "__main__":
    asyncio.run(demonstrate_model_think_integration())
