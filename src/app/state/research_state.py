"""
研究狀態管理
定義研究流程中的狀態結構和數據模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    """用戶澄清請求模型"""

    need_clarification: bool = Field(
        description="是否需要向用戶詢問澄清問題",
    )
    question: str = Field(
        description="向用戶詢問以澄清報告範圍的問題",
    )
    options: List[str] = Field(
        description="提供給用戶選擇的3個面向或方向",
        min_items=3,
        max_items=3
    )
    verification: str = Field(
        description="確認訊息，表示在用戶提供必要資訊後將開始研究",
    )


class ReportPlanSection(BaseModel):
    """報告計劃章節模型"""

    title: str = Field(
        description="章節標題"
    )
    key_question: str = Field(
        description="該章節要解決的關鍵問題"
    )


class ReportPlan(BaseModel):
    """報告計劃模型"""

    background_context: str = Field(
        description="支持性背景脈絡的摘要"
    )
    report_outline: List[ReportPlanSection] = Field(
        description="報告章節列表"
    )
    report_title: str = Field(
        description="報告標題"
    )
