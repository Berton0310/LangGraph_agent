"""
代理狀態管理
定義 LangGraph 中使用的狀態結構
"""
from typing import List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    supervisor_messages: List[BaseMessage] = []
    research_brief: Optional[str] = None
    raw_notes: List[str] = []
    notes: List[str] = []
    final_report: str = ""
