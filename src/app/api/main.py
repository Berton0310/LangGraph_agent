# app/main.py
import os
import time
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# 環境變數（可依需要調整）
# UMODOC_DIST: 指向前端打包輸出的 dist 目錄，存在時走「正式模式」
# UMODOC_BASE: 對應你的 vite.config.ts base，預設 /umo-editor
# VITE_URL:    開發模式下重導的 Vite 伺服器網址
# SPAWN_VITE:  設為 "true" 時，啟動時自動在 UMODOC_EDITOR_DIR 執行 `npm run dev`
# UMODOC_EDITOR_DIR: 前端專案根目錄（含 package.json），用於自動啟動 Vite

UMODOC_DIST = os.getenv("UMODOC_DIST", "dist")
UMODOC_BASE = os.getenv("UMODOC_BASE", "/umo-editor")
VITE_URL = os.getenv("VITE_URL", "http://localhost:9000/umo-editor")
SPAWN_VITE = os.getenv("SPAWN_VITE", "false").lower() == "true"
UMODOC_EDITOR_DIR = os.getenv("UMODOC_EDITOR_DIR", None)

vite_process = None
TEST_HTML_PATH = Path(__file__).parent / "test.html"


def has_dist() -> bool:
    return Path(UMODOC_DIST).is_dir() and any(Path(UMODOC_DIST).iterdir())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vite_process
    if not has_dist() and SPAWN_VITE and UMODOC_EDITOR_DIR:
        # 在本地啟動 Vite（Windows 使用 shell=True 讓 npm 指令可用）
        vite_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=UMODOC_EDITOR_DIR,
            shell=True
        )
        # 粗略等候 Vite 起來，可改為健康檢查
        time.sleep(2)
    try:
        yield
    finally:
        if vite_process and vite_process.poll() is None:
            vite_process.terminate()
            try:
                vite_process.wait(timeout=5)
            except Exception:
                vite_process.kill()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 實際部署應改成你的前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 正式模式：若 dist 存在，掛載為靜態站點
if has_dist():
    # 例如 base=/umo-editor，對應 GET /umo-editor/ 提供 index.html
    app.mount(UMODOC_BASE, StaticFiles(
        directory=UMODOC_DIST, html=True), name="umodoc")


@app.get("/")
def root():
    # 回傳測試頁面，按鈕點擊後才由伺服器決定導向位置
    return FileResponse(TEST_HTML_PATH)


@app.get("/go-umodoc")
def go_umodoc():
    # 由伺服器決定導向到 dist 或開發伺服器
    if has_dist():
        base = UMODOC_BASE if UMODOC_BASE.endswith("/") else f"{UMODOC_BASE}/"
        return RedirectResponse(url=base)
    return RedirectResponse(url=VITE_URL)


@app.get("/load-document")
async def load_document():
    doc = {"title": "模擬後端文檔",
           "content": "<p>模擬後端內容</p>",
           "characterLimit": 10000}
    if not doc:
        # 預設內容
        return {
            "title": "新文檔",
            "content": "<p>尚未有內容</p>",
            "characterLimit": 10000
        }
    print("已經回傳內容")
    # print(doc["document"])
    return doc  # 回傳 document 欄位即可（符合前端格式）

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
