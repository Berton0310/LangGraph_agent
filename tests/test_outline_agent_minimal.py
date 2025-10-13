from src.app.config import set_current_model
import pytest
import sys
sys.path.append('.')


# 如 outline_agent 已被移除，則跳過此測試
outline_mod = pytest.importorskip(
    "src.app.agents.outline_agent", reason="outline_agent 模組不存在，跳過大綱測試")


def test_outline_generation_with_mock_llm(monkeypatch):
    # 使用 gemini-2.5-pro 但避免實際呼叫 API
    set_current_model("gemini-2.5-pro")

    # 準備假的 LLM 物件
    class DummyLLM:
        def invoke(self, messages, config=None):
            class MockResponse:
                def __init__(self, content: str):
                    self.content = content
            return MockResponse(
                """# 測試主題 大綱\n\n## 1. 概述\n- 目的\n- 範圍\n\n## 2. 主要內容\n- 要點A\n- 要點B\n\n## 3. 結論\n- 總結\n"""
            )

    # 在建立 agent 前，替換 config.get_llm，避免需要 API Key
    import src.app.config as cfg
    monkeypatch.setattr(cfg, "get_llm", lambda model_name=None: DummyLLM())

    from src.app.agents.outline_agent import create_outline_agent

    agent = create_outline_agent()

    result = agent.generate_outline(
        topic="測試主題",
        depth_mode=True,
        depth_level="medium",
    )

    assert result["success"] is True
    sections = result["sections"]
    # 驗證至少解析出 3 個章節，且子章節有內容
    assert len(sections) >= 3
    assert any(sec.subsections for sec in sections)
    # 標題至少包含「概述」或「主要內容」或「結論」
    titles = " ".join(sec.title for sec in sections)
    assert any(key in titles for key in ["概述", "主要內容", "結論"])
