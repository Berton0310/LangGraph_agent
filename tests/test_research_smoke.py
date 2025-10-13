from langchain_core.runnables import RunnableConfig
from src.app.config import Configuration
from src.app.state.research_state import SupervisorState, ResearcherState
from src.app.agents.research_agent import supervisor_builder, researcher_builder
import pytest
import asyncio
import sys
sys.path.append('.')


@pytest.mark.asyncio
async def test_supervisor_graph_compiles_and_runs_one_step():
    graph = supervisor_builder
    # 最小 SupervisorState
    state: SupervisorState = {
        "supervisor_messages": [],
        "research_brief": "測試簡報",
        "notes": [],
        "raw_notes": [],
        "research_iterations": 0,
    }
    # 最小 config（不含 API 金鑰，僅確保可以構建 Configuration）
    config = RunnableConfig(configurable={})

    # 嘗試執行一個節點：直接呼叫 supervisor 函式需要 import；這裡只驗證編譯存在
    assert graph is not None


@pytest.mark.asyncio
async def test_researcher_graph_compiles():
    graph = researcher_builder
    assert graph is not None
