"""深度研究代理的圖狀態定義和資料結構。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 結構化輸出
###################
class ConductResearch(BaseModel):
    """調用此工具來進行特定主題的研究。"""
    research_topic: str = Field(
        description="要研究的主題。應該是單一主題，並且應該詳細描述（至少一個段落）。",
    )


class ResearchComplete(BaseModel):
    """調用此工具來表示研究已完成。"""


class Summary(BaseModel):
    """包含關鍵發現的研究摘要。"""

    summary: str
    key_excerpts: str


class ClarifyWithUser(BaseModel):
    """用戶澄清請求的模型。"""

    need_clarification: bool = Field(
        description="是否需要向用戶詢問澄清問題。",
    )
    question: str = Field(
        description="向用戶詢問以澄清報告範圍的問題",
    )
    verification: str = Field(
        description="驗證訊息，表示在用戶提供必要資訊後我們將開始研究。",
    )


class ResearchQuestion(BaseModel):
    """用於指導研究的研究問題和簡報。"""

    research_brief: str = Field(
        description="用於指導研究的研究問題。",
    )


###################
# 狀態定義
###################

def override_reducer(current_value, new_value):
    """允許覆蓋狀態中值的歸約函數。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """輸入狀態僅包含 'messages'。"""


class AgentState(MessagesState):
    """包含訊息和研究資料的主要代理狀態。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation],
                                   override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    timings: Annotated[list[str], override_reducer] = []


class SupervisorState(TypedDict):
    """管理研究任務的主管狀態。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation],
                                   override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    timings: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    """進行研究的個別研究員狀態。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    timings: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """來自個別研究員的輸出狀態。"""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
