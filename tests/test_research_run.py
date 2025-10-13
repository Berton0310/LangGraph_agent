from src.app.state.research_state import AgentState
from src.app.agents import research_agent as ra
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import asyncio
from typing import Any
import sys
sys.path.append('.')


def test_deep_researcher_end_to_end(monkeypatch):
    """端到端測試深度研究流程（以 mock 模型避免外部 API）。"""

    class DummyModel:
        def __init__(self, model_class: Any | None = None):
            self._model_class = model_class

        # 鏈式 API 模擬
        def with_structured_output(self, model_class):
            return DummyModel(model_class)

        def with_retry(self, *args, **kwargs):
            return self

        def with_config(self, *args, **kwargs):
            return self

        def bind_tools(self, *args, **kwargs):
            return self

        # 非同步 API 的同步替代（LangGraph 使用 async 執行）
        async def ainvoke(self, messages):
            # 針對結構化輸出類型回傳最小可用測試資料
            if self._model_class is not None:
                # ClarifyWithUser
                if self._model_class.__name__ == 'ClarifyWithUser':
                    return self._model_class(need_clarification=False, question="", verification="OK")
                # ResearchQuestion
                if self._model_class.__name__ == 'ResearchQuestion':
                    return self._model_class(research_brief="研究簡報：測試主題")
                # 其他結構化模型
                return self._model_class()  # type: ignore

            # 一般模型回傳沒有工具呼叫的 AIMessage（使流程往下走）
            return AIMessage(content="ok", tool_calls=[])

    class DummyFactory:
        # 讓所有入口都返回 DummyModel
        def with_structured_output(self, model_class):
            return DummyModel(model_class)

        def with_retry(self, *args, **kwargs):
            return self

        def with_config(self, *args, **kwargs):
            return DummyModel()

        def bind_tools(self, *args, **kwargs):
            return DummyModel()

    # 替換 research_agent 內的 configurable_model
    monkeypatch.setattr(ra, 'configurable_model', DummyFactory())

    # 建立最小輸入狀態：只要有 messages 即可
    input_state: AgentState = {
        'messages': [HumanMessage(content='請研究一個測試主題')]
    }  # type: ignore

    # 使用 asyncio 執行 async 版本
    result = asyncio.run(ra.deep_researcher.ainvoke(
        input_state, config={"configurable": {"thread_id": "t1"}}))

    # final_report_generation 會回傳 dict，至少應有這些鍵
    assert isinstance(result, dict)
    assert 'messages' in result
    assert 'final_report' in result
