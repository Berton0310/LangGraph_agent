#!/usr/bin/env python3
"""
研究服務器啟動腳本（獨立版）
- 將原本 src/app/api/main.py 的程式整合進本檔
- 不再以模組匯入方式載入 app
"""

# 基本與路徑設定
import json
import asyncio
import time
from datetime import datetime
from pprint import pformat
import uvicorn
from bson.objectid import ObjectId  # 保留原結構，用於未來擴充（目前未啟用）
from pymongo import MongoClient  # 保留原結構，用於未來擴充（目前未啟用）
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Request
import sys
import os
import logging
from pathlib import Path
import traceback

# 確保可匯入 src.*（供研究代理與設定使用）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 第三方套件

# ===== 應用與 CORS 設定 =====
app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境允許所有來源，生產環境請限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# （保留，未啟用）MongoDB 連線佔位
# client = MongoClient("mongodb+srv://<user>:<pass>@<cluster>/<db>?...")
# db = client.test
# collection = db.umodoc_test

# ===== SSE 工具 =====


def sse_event(payload: dict) -> str:
    try:
        data = json.dumps(payload, ensure_ascii=False)
    except Exception:
        data = json.dumps(
            {"type": "error", "message": "serialization failed"}, ensure_ascii=False)
    return f"data: {data}\n\n"

# ===== 檔案寫入工具 =====


def _write_result_to_file(result: dict, started_at: str, ended_at: str, elapsed_seconds: float, filename: str = "output.txt"):
    """將研究結果寫入檔案（與 run_research.py 一致風格）。"""
    try:
        # 去重 timings：保留每個標籤的最後一次記錄
        if isinstance(result, dict) and "timings" in result:
            timings = result["timings"]
            if isinstance(timings, list):
                # 建立標籤到最後時間的映射
                last_timings = {}
                for timing in timings:
                    if isinstance(timing, str) and ": " in timing:
                        label, time_str = timing.split(": ", 1)
                        last_timings[label] = time_str

                # 重建去重後的 timings
                result = result.copy()
                result["timings"] = [
                    f"{label}: {time_str}" for label, time_str in last_timings.items()]

        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"started_at: {started_at}\nended_at: {ended_at}\nelapsed_seconds: {elapsed_seconds:.2f}\n")
            f.write("\n===== result (pretty) =====\n")
            f.write(pformat(result, depth=5, width=120))
            if isinstance(result, dict) and "final_report" in result:
                f.write("\n\n===== final_report =====\n")
                f.write(str(result["final_report"]))
    except Exception as write_err:
        logger.warning(f"寫入 {filename} 失敗: {write_err}")

# ===== 簡易文件暫存（供前端 load-document 使用） =====


DOCUMENT_STORE = {
    "title": "研究文件",
    "content": "<p>尚未有內容</p>",
    "characterLimit": 100000,
}


def _update_document_content(html: str):
    """將 final_report 寫入暫存區，供 /load-document 回傳。"""
    try:
        if html and isinstance(html, str):
            # 若內容包含 Markdown 標題，從第一個 '#' 開始截取
            try:
                first_hash_idx = html.find("#")
                normalized = html[first_hash_idx:] if first_hash_idx != -1 else html
            except Exception:
                normalized = html
            DOCUMENT_STORE["content"] = normalized
    except Exception as e:
        logger.warning(f"更新文件暫存失敗: {e}")

# ===== 資料模型（保留原結構） =====


class DocumentContent(BaseModel):
    html: str
    json_data: dict  # 避免與 BaseModel.json() 衝突
    text: str


class PageSize(BaseModel):
    label: str
    width: float
    height: float
    default: bool


class Page(BaseModel):
    size: PageSize
    # 可擴充其他欄位（zoomLevel, margin 等）


class DocumentData(BaseModel):
    content: DocumentContent
    page: Page
    document: dict


# ===== AI 助手（保留原結構） =====
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
        command = payload.command.strip()
        selected_text = payload.input.strip()

        print("收到AI助手請求:")
        print(f"  指令: {command}")
        print(f"  選中文字: {selected_text}")
        print(f"  語言: {payload.lang}")
        print(f"  文件內容長度: {len(content.text)} 字元")

        if command in self.commands:
            return self.commands[command](selected_text, content)
        else:
            return self._custom_command(command, selected_text, content)

    def _continue_writing(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要續寫的文字內容。</p>"
        continuation = (
            f"<p>{selected_text}... 這是續寫的內容。根據您提供的文字，我將繼續發展這個主題，"
            f"提供更多相關的資訊和見解。</p>"
        )
        return continuation

    def _rewrite(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要重寫的文字內容。</p>"
        return (
            f"<p>重寫版本：{selected_text}</p><p>這是一個重新表達的版本，保持了原意但使用了不同的表達方式。</p>"
        )

    def _abbreviate(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要縮寫的文字內容。</p>"
        return f"<p>縮寫版本：{selected_text[:50]}...</p>"

    def _expand(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要擴寫的文字內容。</p>"
        return (
            f"<p>擴寫版本：{selected_text}</p><p>這是一個更詳細的版本，包含了更多背景資訊、例子和解釋，讓內容更加豐富和完整。</p>"
        )

    def _polish(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要潤色的文字內容。</p>"
        return (
            f"<p>潤色版本：{selected_text}</p><p>這個版本經過了語言優化，表達更加流暢自然，用詞更加精準。</p>"
        )

    def _proofread(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要校閱的文字內容。</p>"
        return (
            f"<p>校閱結果：</p><p>原文：{selected_text}</p><p>修正建議：已檢查拼字、語法和表達，內容基本正確。</p>"
        )

    def _translate(self, selected_text: str, content: AssistantContent) -> str:
        if not selected_text:
            return "<p>請先選擇要翻譯的文字內容。</p>"
        return (
            f"<p>翻譯結果：</p><p>原文：{selected_text}</p><p>譯文：This is a translated version of the selected text.</p>"
        )

    def _custom_command(self, command: str, selected_text: str, content: AssistantContent) -> str:
        return f"<p>收到自定義指令：{command}</p><p>選中文字：{selected_text}</p><p>這是一個自定義的AI處理結果。</p>"


assistant_handler = AssistantHandler()


# ===== AI 助手 API =====
@app.post("/ai-assistant")
async def ai_assistant(request: AssistantRequest):
    try:
        result_content = assistant_handler.process_command(
            request.payload,
            request.content,
        )
        return AssistantResponse(
            success=True,
            message="AI助手處理成功",
            content=result_content,
        )
    except Exception as e:
        # 僅在伺服器端記錄完整錯誤，避免向使用者洩漏內部細節
        logger.exception("AI 助手處理失敗")
        return AssistantResponse(
            success=False,
            message="AI助手處理失敗",
            error="系統繁忙，請稍後再試",
        )


# ===== 研究頁面與研究 API（非串流） =====
@app.get("/research")
async def research_page():
    """提供研究頁面"""
    html_path = Path(__file__).parent / "src" / "app" / "api" / "research.html"
    return FileResponse(html_path)


@app.post("/research/run")
async def run_research(request: dict):
    """執行研究並返回結果（非串流）"""
    try:
        question = request.get("question", "")
        if not question:
            return {"error": "請提供研究問題"}

        # 動態匯入研究代理（仍需 src.* 可匯入）
        from src.app.agents.research_agent import deep_researcher
        from src.app.config import Configuration

        config = Configuration()
        configurable = {**config.model_dump(mode="json")}
        configurable["allow_clarification"] = False
        run_config = {"configurable": configurable}

        start_dt = datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        result = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            run_config,
        )
        end_time = time.perf_counter()
        end_dt = datetime.now().isoformat(timespec="seconds")
        elapsed = end_time - start_time

        # 寫入檔案
        _write_result_to_file(result, start_dt, end_dt, elapsed)
        # 更新暫存文件內容供 /load-document 使用
        _update_document_content(result.get("final_report", ""))

        return {
            "success": True,
            "final_report": result.get("final_report", "研究完成但未生成報告"),
            "timings": result.get("timings", []),
        }

    except Exception:
        # 僅在伺服器端記錄完整錯誤堆疊
        logger.exception("研究執行失敗")
        return {"error": "研究執行失敗，請稍後重試"}


# ===== 全域任務管理 =====
active_tasks: dict[str, asyncio.Task] = {}


def cancel_task(task_id: str):
    """取消指定的任務"""
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
        del active_tasks[task_id]


def register_task(task_id: str, task: asyncio.Task):
    """註冊任務到全域管理"""
    active_tasks[task_id] = task

# ===== 研究進度 SSE 串流（即時 print_progress） =====


@app.get("/research/stream")
async def research_stream(
    request: Request,
    question: str = Query(..., description="研究問題"),
):
    async def event_generator():
        yield sse_event({"type": "stage_start", "stage": "deep_researcher", "message": "開始研究"})

        # 以 asyncio.Queue 收集 print_progress 的即時訊息
        progress_queue: asyncio.Queue[str] = asyncio.Queue()

        # 綁定回調（不暴露內部資訊）
        try:
            from src.app.agents import research_agent as ra
        except Exception:
            logger.exception("載入研究代理失敗")
            yield sse_event({"type": "error", "message": "系統初始化失敗"})
            return

        def progress_callback(msg: str):
            try:
                progress_queue.put_nowait(msg)
            except Exception:
                pass

        # 設定回調
        ra.print_progress.progress_callback = progress_callback

        # 啟動研究背景任務
        async def run_research_task():
            try:
                from src.app.config import Configuration
                config = Configuration()
                configurable = {**config.model_dump(mode="json")}
                configurable["allow_clarification"] = False
                run_config = {"configurable": configurable}
                result = await ra.deep_researcher.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    run_config,
                )
                return result
            except Exception:
                logger.exception("研究過程錯誤")
                return {"error": True}

        # 記錄開始時間，完成後與結果一同寫入 output.txt
        started_at = datetime.now().isoformat(timespec="seconds")
        wall_start = time.perf_counter()
        task = asyncio.create_task(run_research_task())

        # 生成任務ID並註冊到全域管理
        task_id = f"research_{int(time.time() * 1000)}_{question[:20]}"
        register_task(task_id, task)

        try:
            # 迴圈推送進度，直到任務完成或客戶端中斷
            while not task.done():
                # 檢查前端是否已中斷連線（使用者取消）
                if await request.is_disconnected():
                    if not task.done():
                        task.cancel()
                    break
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                    # 解析並轉發標準化階段事件
                    if "STAGE::" in msg:
                        try:
                            stage_segment = msg[msg.index("STAGE::"):]
                            # e.g., STAGE::research_supervisor::enter::iteration=3
                            parts = stage_segment.split("::")
                            stage_key = parts[1] if len(parts) > 1 else ""
                            iteration = None
                            if len(parts) >= 4 and parts[3].startswith("iteration="):
                                try:
                                    iteration = int(parts[3].split("=", 1)[1])
                                except Exception:
                                    iteration = None

                            stage_map = {
                                "clarify_with_user": "clarify",
                                "write_research_brief": "plan",
                                "research_supervisor": "execute",
                                "final_report_generation": "report",
                            }
                            stage_name = stage_map.get(stage_key)
                            if stage_name:
                                yield sse_event({
                                    "type": "stage",
                                    "stage": stage_name,
                                    "iteration": iteration,
                                })
                                continue
                        except Exception:
                            # 若解析失敗則退回一般進度訊息
                            pass

                    # 一般進度訊息
                    yield sse_event({"type": "progress", "message": msg})
                except asyncio.TimeoutError:
                    # 空轉，檢查任務是否完成
                    continue
                except Exception:
                    # 靜默忽略單筆推送問題
                    continue

            # 如果是正常完成，推送最終報告（若有）
            if not task.cancelled():
                result = task.result()
                ended_at = datetime.now().isoformat(timespec="seconds")
                elapsed = time.perf_counter() - wall_start
                if isinstance(result, dict) and not result.get("error"):
                    # 寫入檔案（與非串流端點一致）
                    _write_result_to_file(
                        result, started_at, ended_at, elapsed)
                    final_report = result.get("final_report")
                    if final_report:
                        # 更新暫存文件內容供 /load-document 使用
                        _update_document_content(final_report)
                        yield sse_event({"type": "final_report", "report": final_report})
                else:
                    yield sse_event({"type": "error", "message": "研究執行失敗，請稍後重試"})

        except asyncio.CancelledError:
            # 生成器被終止（連線中斷），確保任務被取消
            if not task.done():
                task.cancel()
            raise
        finally:
            # 清理任務註冊
            if task_id in active_tasks:
                del active_tasks[task_id]
            # 統一收尾訊號（若是客戶端取消，這段可能不會被收到）
            try:
                yield sse_event({"type": "stage_complete", "stage": "deep_researcher"})
            except Exception:
                pass

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


# ===== 取消研究任務 API =====
@app.post("/research/cancel")
async def cancel_research(request: dict):
    """取消當前正在執行的研究任務"""
    try:
        question = request.get("question", "")
        if not question:
            return {"error": "請提供研究問題"}

        # 尋找匹配的任務並取消
        canceled_tasks = []
        for task_id, task in list(active_tasks.items()):
            if question in task_id and not task.done():
                task.cancel()
                canceled_tasks.append(task_id)
                del active_tasks[task_id]

        if canceled_tasks:
            logger.info(f"已取消 {len(canceled_tasks)} 個研究任務: {canceled_tasks}")
            return {
                "success": True,
                "message": f"已成功取消 {len(canceled_tasks)} 個研究任務",
                "canceled_tasks": canceled_tasks
            }
        else:
            return {
                "success": False,
                "message": "未找到匹配的研究任務或任務已完成"
            }

    except Exception as e:
        logger.exception("取消研究任務失敗")
        return {"error": "取消研究任務失敗，請稍後重試"}


@app.get("/load-document")
async def load_document():
    # 直接回傳暫存文件（保持與前端期望結構的相容性）
    return {
        "title": DOCUMENT_STORE.get("title", "研究文件"),
        "content": DOCUMENT_STORE.get("content", "<p>尚未有內容</p>"),
        "characterLimit": DOCUMENT_STORE.get("characterLimit", 100000)
    }

if __name__ == "__main__":
    print("🚀 正在啟動研究服務器...")
    print("📍 服務器地址: http://127.0.0.1:8000")
    print("🔬 研究界面: http://127.0.0.1:8000/research")
    print("⏹️  按 Ctrl+C 停止服務器")
    print("-" * 50)

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
