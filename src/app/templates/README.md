# 模板系統使用指南

## 概述

模板系統為 LangGraph Agent 提供了標準化的報告結構和風格，確保生成的報告符合特定領域的要求和最佳實踐。

## 檔案結構

```
src/app/templates/
├── __init__.py                 # 模組初始化
├── template_manager.py        # 模板管理器
├── templates.json             # 統一模板配置檔案
└── README.md                 # 使用指南
```

## 統一配置檔案

所有模板配置都集中在 `templates.json` 檔案中，包含：

- **模板定義**: 所有可用模板的完整配置
- **選擇規則**: 自動推薦模板的規則和關鍵詞
- **自定義選項**: 結構修改和風格自定義選項
- **整合資訊**: 與 Agent 系統的整合映射
- **使用範例**: 實際使用案例和範例
- **系統元數據**: 版本資訊和維護資訊

## 可用模板

### 1. 學術模板 (academic)

**用途**: 適用於學術論文、研究報告、學位論文等正式學術文件

**目標受眾**: 學者、研究人員、學生、學術機構

**結構**:

1. 摘要 (Abstract)
2. 研究背景 (Background)
3. 文獻回顧 (Literature Review)
4. 研究方法 (Methodology)
5. 結果與分析 (Results & Discussion)
6. 結論 (Conclusion)
7. 參考文獻 (References)

**風格**:

- 語調: 正式 (formal)
- 引用格式: APA
- 語言: 中文
- 證據等級: 高
- 客觀性: 高

**適用主題**:

- AI 技術在教育領域的應用研究
- 機器學習算法優化分析
- 數據科學方法論研究
- 人工智慧倫理問題探討

### 2. 商業模板 (business)

**用途**: 適用於商業分析、市場研究、投資建議、策略規劃等商業文件

**目標受眾**: 企業管理層、投資者、客戶、合作夥伴

**結構**:

1. 執行摘要 (Executive Summary)
2. 市場現況 (Market Overview)
3. 問題分析 (Pain Points)
4. 解決方案 (Solution Proposal)
5. 財務預測 (Financial Projection)
6. 建議與結論 (Recommendation)

**風格**:

- 語調: 專業 (professional)
- 引用格式: 無
- 語言: 中文
- 證據等級: 中高
- 客觀性: 中

**適用主題**:

- AI 技術市場趨勢分析
- 新興科技投資機會評估
- 數位轉型策略建議
- 競爭對手分析報告

## 使用方法

### 1. 基本使用

```python
from src.app.templates import TemplateManager, recommend_template

# 創建模板管理器
manager = TemplateManager()

# 獲取所有可用模板
templates = manager.get_available_templates()
print(f"可用模板: {templates}")

# 獲取特定模板
academic_template = manager.get_template("academic")
print(f"學術模板: {academic_template['purpose']}")

# 推薦模板
recommended = recommend_template("AI技術在教育領域的應用研究")
print(f"推薦模板: {recommended}")
```

### 2. 模板資訊查詢

```python
# 獲取模板結構
structure = manager.get_template_structure("academic")
print("學術模板結構:")
for i, section in enumerate(structure, 1):
    print(f"  {i}. {section}")

# 獲取模板風格
style = manager.get_template_style("business")
print(f"商業模板風格: {style}")

# 獲取使用案例
use_cases = manager.get_template_use_cases("academic")
print("學術模板使用案例:")
for case in use_cases:
    print(f"  • {case}")
```

### 3. 自定義結構生成

```python
# 根據主題生成自定義結構
topic = "AI技術在教育領域的應用研究"
custom_structure = manager.generate_custom_structure("academic", topic)

print("自定義結構:")
for i, section in enumerate(custom_structure, 1):
    print(f"  {i}. {section}")
```

### 4. 整合到 Agent 系統

```python
from src.app.agents.template_aware_plan_agent import TemplateAwarePlanAgent

# 創建支援模板的規劃代理
agent = TemplateAwarePlanAgent()

# 生成報告計劃
task = "生成一份關於AI技術發展的研究報告大綱"
result = await agent.generate_plan(task)

print("生成的計劃:")
print(result)
```

## 模板選擇指南

### 選擇學術模板的情況

- 需要嚴謹的學術標準
- 目標讀者為學術界
- 需要詳細的文獻回顧
- 要求客觀的研究方法
- 需要標準化的引用格式

### 選擇商業模板的情況

- 需要商業決策支援
- 目標讀者為企業管理層
- 需要實用的解決方案
- 要求清晰的財務分析
- 需要可執行的建議

## 自定義選項

### 結構修改

**學術模板**:

- 可添加附錄 (Appendix)
- 可包含致謝 (Acknowledgments)
- 可增加術語表 (Glossary)
- 可添加圖表目錄 (List of Figures/Tables)

**商業模板**:

- 可添加風險分析 (Risk Analysis)
- 可包含實施計劃 (Implementation Plan)
- 可增加時間表 (Timeline)
- 可添加附錄資料 (Supporting Data)

### 風格自定義

- **語調選項**: formal, professional, casual, technical
- **引用格式**: APA, MLA, Chicago, IEEE, none
- **語言選項**: Chinese, English, bilingual
- **格式化風格**: academic, business, technical, creative

## 與 Agent 系統整合

### Plan Agent

規劃代理會根據選擇的模板生成相應的報告大綱：

```python
# 學術模板 → 生成學術標準的章節結構
# 商業模板 → 生成商業導向的報告框架
```

### Research Agent

研究代理會根據模板要求收集相應類型的資料：

```python
# 學術模板 → 收集學術文獻和研究成果
# 商業模板 → 收集市場數據和商業資訊
```

### Writer Agent

撰寫代理會根據模板風格生成相應格式的內容：

```python
# 學術模板 → 使用學術寫作風格和格式
# 商業模板 → 使用商業簡報風格和格式
```

## 使用範例

### 學術報告範例

**主題**: AI 技術在教育領域的應用研究

**生成結構**:

1. 摘要：AI 教育應用的現狀與前景
2. 研究背景：教育技術發展歷程
3. 文獻回顧：AI 教育應用的相關研究
4. 研究方法：案例研究與問卷調查
5. 結果與分析：AI 教育應用的效果評估
6. 結論：AI 教育應用的挑戰與機會
7. 參考文獻：相關學術文獻

### 商業報告範例

**主題**: AI 技術市場趨勢分析

**生成結構**:

1. 執行摘要：AI 市場發展概況
2. 市場現況：AI 技術市場規模與成長
3. 問題分析：AI 技術面臨的挑戰
4. 解決方案：AI 技術發展策略建議
5. 財務預測：AI 市場未來成長預測
6. 建議與結論：投資與發展建議

## 未來擴展

計劃添加的模板：

1. **技術模板** (technical): 適用於技術規格、API 文檔、系統設計等
2. **創意模板** (creative): 適用於設計提案、創意方案、藝術項目等
3. **政策模板** (policy): 適用於政策分析、法規研究、政府報告等

## 最佳實踐

1. **模板選擇**: 根據目標受眾和報告用途選擇合適的模板
2. **結構調整**: 根據具體需求調整模板結構
3. **風格一致**: 確保整個報告使用一致的風格和格式
4. **內容品質**: 模板提供結構，但內容品質仍需人工審核
5. **持續改進**: 根據使用反饋持續改進模板設計

## 技術支援

如有問題或建議，請參考：

- `template_summary.json`: 完整的模板配置資訊
- `template_manager.py`: 模板管理器的完整實作
- `template_aware_plan_agent.py`: 整合範例

模板系統為您的報告生成提供了強大的結構化支援，確保生成的內容符合專業標準和最佳實踐。
