# 深度研究代理系統

這是一個基於 LangGraph 的深度研究代理系統，使用 Gemini-2.5-Pro 模型進行智能研究分析。

## 🚀 快速開始

### 1. 環境設置

#### 方法一：使用設置腳本（推薦）

```bash
# Windows
setup_env.bat

# 或手動設置環境變數
set GEMINI_API_KEY=your_gemini_api_key_here
set TAVILY_API_KEY=your_tavily_api_key_here  # 可選
```

#### 方法二：手動設置

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:TAVILY_API_KEY="your_tavily_api_key_here"

# Windows CMD
set GEMINI_API_KEY=your_gemini_api_key_here
set TAVILY_API_KEY=your_tavily_api_key_here

# Linux/Mac
export GEMINI_API_KEY="your_gemini_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"
```

### 2. 安裝依賴項

```bash
pip install -r requirements.txt
```

### 3. 執行專案

#### 快速測試

```bash
python test_research.py
```

#### 完整研究

```bash
python run_research.py
```

#### 驗證配置

```bash
python verify_models.py
```

## 📋 系統功能

### 🔍 研究能力

- **自動搜尋**: 使用 Tavily API 進行網頁搜尋
- **多層次分析**: 監督者-研究者架構
- **智能壓縮**: 自動整理和總結研究結果
- **報告生成**: 生成結構化的研究報告

### 🤖 模型配置

所有模型統一使用 `gemini-2.5-pro`:

- 研究模型
- 壓縮模型
- 最終報告模型
- 摘要模型
- 規劃模型

### 🛠️ 工具支援

- **Tavily 搜尋**: 網頁搜尋和內容摘要
- **MCP 工具**: 外部工具整合
- **反思工具**: 戰略思考和決策制定

## 📁 專案結構

```
src/app/
├── agents/          # 代理實作
├── config.py        # 配置管理
├── prompt.py        # 提示詞模板
├── state/           # 狀態定義
├── utils/           # 工具函數
└── api/             # API 端點
```

## 🔧 配置選項

### 主要配置

- `search_api`: 搜尋 API (tavily/openai/anthropic/none)
- `max_researcher_iterations`: 最大研究迭代次數
- `max_concurrent_research_units`: 最大並行研究單位
- `allow_clarification`: 是否允許澄清問題

### 模型配置

- `research_model`: 研究模型
- `compression_model`: 壓縮模型
- `final_report_model`: 最終報告模型
- `summarization_model`: 摘要模型

## 🎯 使用範例

### 基本研究

```python
from app.agents.research_agent import deep_researcher
from langchain_core.messages import HumanMessage

# 運行研究
result = await deep_researcher.ainvoke(
    {"messages": [HumanMessage(content="人工智慧的發展歷史")]}
)

print(result["final_report"])
```

### 自定義配置

```python
from app.config import Configuration

config = Configuration()
config.max_researcher_iterations = 5
config.max_concurrent_research_units = 3

# 使用自定義配置運行
result = await deep_researcher.ainvoke(
    {"messages": [HumanMessage(content="研究主題")]},
    config={"configurable": config.dict()}
)
```

## 🐛 故障排除

### 常見問題

1. **"no module named app" 錯誤**

   - 確保在專案根目錄執行
   - 檢查 Python 路徑設置

2. **API 金鑰錯誤**

   - 確認環境變數設置正確
   - 檢查 API 金鑰是否有效

3. **搜尋功能無法使用**
   - 設置 TAVILY_API_KEY
   - 或使用其他搜尋 API

### 日誌和調試

```bash
# 啟用詳細日誌
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_API_KEY=your_langsmith_key

python test_research.py
```

## 📞 支援

如有問題，請檢查：

1. 環境變數設置
2. 依賴項安裝
3. API 金鑰有效性
4. 網路連接狀態

## 🔄 更新日誌

- **v1.0**: 統一所有模型為 gemini-2.5-pro
- **v0.9**: 修復模組導入問題
- **v0.8**: 添加完整的測試和驗證腳本
