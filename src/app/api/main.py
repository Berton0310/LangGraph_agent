# 設置 Python 路徑
import logging
from pathlib import Path
from fastapi.responses import FileResponse
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Query
from bson.objectid import ObjectId
import sys
import os
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


# 建立 FastAPI 應用
app = FastAPI()

# 設定日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# 加上 CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境允許所有來源，生產環境應限制特定網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# # 連接 MongoDB
# client = client = MongoClient(
#     "mongodb+srv://root:root123@cluster0.pbz1j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client.test  # database
# collection = db.umodoc_test

# 定義傳入的資料格式


class DocumentContent(BaseModel):
    html: str
    json_data: dict
    text: str


class PageSize(BaseModel):
    label: str
    width: float
    height: float
    default: bool


class Page(BaseModel):
    size: PageSize
    # 加上其他欄位也可以，如 zoomLevel, margin 等（依需求）


class DocumentData(BaseModel):
    content: DocumentContent
    page: Page
    document: dict


# AI 助手相關的資料模型
class AssistantPayload(BaseModel):
    lang: str
    input: str
    command: str
    output: str


class AssistantContent(BaseModel):
    html: str
    text: str
    json_data: dict


class AssistantRequest(BaseModel):
    payload: AssistantPayload
    content: AssistantContent


class AssistantResponse(BaseModel):
    success: bool
    message: str
    content: str = ""
    error: str = ""


# 儲存文件的 API 路由

# @app.post("/save-document")
# async def save_document(data: DocumentData):
#     # 插入資料到 MongoDB
#     result = collection.insert_one(data.model_dump())
#     return {
#         "message": "儲存成功",
#         "inserted_id": str(result.inserted_id),
#     }


# AI 助手處理類別
class AssistantHandler:
    def __init__(self):
        self.commands = {
            "續寫": self._continue_writing,
            "重寫": self._rewrite,
            "縮寫": self._abbreviate,
            "擴寫": self._expand,
            "潤色": self._polish,
            "校閱": self._proofread,
            "翻譯": self._translate,
            "Continuation": self._continue_writing,
            "Rewrite": self._rewrite,
            "Abbreviation": self._abbreviate,
            "Expansion": self._expand,
            "Polish": self._polish,
            "Proofread": self._proofread,
            "Translate": self._translate,
        }

    def process_command(self, payload: AssistantPayload, content: AssistantContent) -> str:
        """處理AI助手指令"""
        command = payload.command.strip()
        selected_text = payload.input.strip()

        print(f"收到AI助手請求:")
        print(f"  指令: {command}")
        print(f"  選中文字: {selected_text}")
        print(f"  語言: {payload.lang}")
        print(f"  文件內容長度: {len(content.text)} 字元")

        # 根據指令處理
        if command in self.commands:
            return self.commands[command](selected_text, content)
        else:
            return self._custom_command(command, selected_text, content)

    def _continue_writing(self, selected_text: str, content: AssistantContent) -> str:
        """續寫功能"""
        if not selected_text:
            return "<p>請先選擇要續寫的文字內容。</p>"

        # 這裡可以整合真實的AI服務，目前返回模擬內容
        continuation = f"<p>{selected_text}... 這是續寫的內容。根據您提供的文字，我將繼續發展這個主題，提供更多相關的資訊和見解。</p>"
        return continuation

    def _rewrite(self, selected_text: str, content: AssistantContent) -> str:
        """重寫功能"""
        if not selected_text:
            return "<p>請先選擇要重寫的文字內容。</p>"

        # 模擬重寫內容
        rewritten = f"<p>重寫版本：{selected_text}</p><p>這是一個重新表達的版本，保持了原意但使用了不同的表達方式。</p>"
        return rewritten

    def _abbreviate(self, selected_text: str, content: AssistantContent) -> str:
        """縮寫功能"""
        if not selected_text:
            return "<p>請先選擇要縮寫的文字內容。</p>"

        # 模擬縮寫內容
        abbreviated = f"<p>縮寫版本：{selected_text[:50]}...</p>"
        return abbreviated

    def _expand(self, selected_text: str, content: AssistantContent) -> str:
        """擴寫功能"""
        if not selected_text:
            return "<p>請先選擇要擴寫的文字內容。</p>"

        # 模擬擴寫內容
        expanded = f"<p>擴寫版本：{selected_text}</p><p>這是一個更詳細的版本，包含了更多背景資訊、例子和解釋，讓內容更加豐富和完整。</p>"
        return expanded

    def _polish(self, selected_text: str, content: AssistantContent) -> str:
        """潤色功能"""
        if not selected_text:
            return "<p>請先選擇要潤色的文字內容。</p>"

        # 模擬潤色內容
        polished = f"<p>潤色版本：{selected_text}</p><p>這個版本經過了語言優化，表達更加流暢自然，用詞更加精準。</p>"
        return polished

    def _proofread(self, selected_text: str, content: AssistantContent) -> str:
        """校閱功能"""
        if not selected_text:
            return "<p>請先選擇要校閱的文字內容。</p>"

        # 模擬校閱內容
        proofread = f"<p>校閱結果：</p><p>原文：{selected_text}</p><p>修正建議：已檢查拼字、語法和表達，內容基本正確。</p>"
        return proofread

    def _translate(self, selected_text: str, content: AssistantContent) -> str:
        """翻譯功能"""
        if not selected_text:
            return "<p>請先選擇要翻譯的文字內容。</p>"

        # 模擬翻譯內容
        translated = f"<p>翻譯結果：</p><p>原文：{selected_text}</p><p>譯文：This is a translated version of the selected text.</p>"
        return translated

    def _custom_command(self, command: str, selected_text: str, content: AssistantContent) -> str:
        """處理自定義指令"""
        return f"<p>收到自定義指令：{command}</p><p>選中文字：{selected_text}</p><p>這是一個自定義的AI處理結果。</p>"


# 建立AI助手處理器實例
assistant_handler = AssistantHandler()


# AI 助手 API 路由
@app.post("/ai-assistant")
async def ai_assistant(request: AssistantRequest):
    try:
        # 處理AI助手請求
        result_content = assistant_handler.process_command(
            request.payload,
            request.content
        )

        return AssistantResponse(
            success=True,
            message="AI助手處理成功",
            content=result_content
        )

    except Exception as e:
        print(f"AI助手處理錯誤: {str(e)}")
        return AssistantResponse(
            success=False,
            message="AI助手處理失敗",
            error=str(e)
        )


# @app.get("/load-document/{user_id}")
# @app.get("/load-document")
# async def load_document():
#     doc = collection.find_one({"_id": ObjectId('68416c1e6c430bbe2eaa074c')})
#     if not doc:
#         # 預設內容
#         return {
#             "title": "新文檔",
#             "content": "<p>尚未有內容</p>",
#             "characterLimit": 10000
#         }
#     print("已經回傳內容")
#     # print(doc["document"])
#     return doc["document"]  # 回傳 document 欄位即可（符合前端格式）


if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    # 添加項目根目錄到 Python 路徑
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    uvicorn.run("src.app.api.main:app",
                host="127.0.0.1", port=8000, reload=True)


# ============== 研究 API ==============

@app.get("/research")
async def research_page():
    """提供研究頁面"""
    html_path = Path(__file__).parent / "research.html"
    return FileResponse(html_path)


@app.post("/research/run")
async def run_research(request: dict):
    """執行研究並返回結果"""
    try:
        question = request.get("question", "")
        if not question:
            return {"error": "請提供研究問題"}

        # 動態匯入研究代理
        from src.app.agents.research_agent import deep_researcher
        from src.app.config import Configuration

        # 建立配置
        config = Configuration()
        configurable = {**config.model_dump(mode="json")}
        configurable["allow_clarification"] = False
        run_config = {"configurable": configurable}

        # 執行研究
        result = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            run_config
        )

        # 返回結果
        return {
            "success": True,
            "final_report": result.get("final_report", "研究完成但未生成報告"),
            "timings": result.get("timings", [])
        }

    except Exception as e:
        logger.error(f"Research error: {e}")
        return {"error": f"研究執行失敗: {str(e)}"}
