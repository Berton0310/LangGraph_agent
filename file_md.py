import os
import re
import time
import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Iterable
from langchain_unstructured import UnstructuredLoader
import google.generativeai as genai
import io
import requests
from dotenv import load_dotenv
import base64
load_dotenv()
# 設定 API 金鑰
API_KEY = os.getenv("OLMOCR_API_KEY")


def analyze_image_with_gemini(image_path, prompt="請你將這張圖片內容轉換成markdown表格，並使用中文"):
    """
    使用 Gemini 2.5 Flash 模型分析圖片

    Args:
        image_path (str): 圖片檔案路徑
        prompt (str): 要詢問的問題

    Returns:
        str: Gemini 的回應
    """
    try:
        # 載入圖片
        image = Image.open(image_path)

        # 初始化 Gemini 模型
        model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')

        # 準備圖片和提示詞
        response = model.generate_content([prompt, image])

        return response.text

    except Exception as e:
        return f"錯誤: {str(e)}"


prompt_ocr = '''Return the plain text representation of the attached image as Markdown.
Convert equations to LaTeX, tables to HTML, and figures to Markdown image syntax.
Do NOT include any YAML front matter fields (such as primary_language, rotation info, or diagram flags).
Output only Markdown.
The document is primarily in Traditional Chinese. Preserve all characters and punctuation.'''
'''
將附加圖像的純文本表示形式返回為 Markdown。
將方程式轉換為 LaTeX，將表格轉換為 HTML，將圖形轉換為 Markdown 圖像語法。
請勿包含任何 YAML 前置欄位 （，例如 primary_language、旋轉資訊或圖表旗標） 。
僅輸出 Markdown。
該文件主要以繁體中文為英文。保留所有字元和標點符號。
'''


def analyze_image_with_olmocr(image_paths, prompt=prompt_ocr):
    """使用 OlmoCR 模型分析圖片"""
    try:
        image = Image.open(image_paths)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "model": "allenai/olmOCR-7B-0825",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers=headers,
            json=payload,
            timeout=180
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"錯誤: {str(e)}"


"""
視覺化doc_local,除錯用
"""


def plot_pdf_with_boxes(pdf_page, segments):
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
        # "Table": "forestgreen",
    }
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height)
            for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(
            scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(
                    color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()


def render_page(doc_list: list, page_number: int, file_path: str, print_text=True) -> None:
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get("page_number") == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")


def extract_regions_as_images(
    docs_local: List,
    pdf_path: str,
    pages: List[int] = None,
    output_dir: str = "pdf_images",
    dpi: int = 2,
    categories: Iterable[str] = ("Table", "Image"),
    filename_prefix: str = "region",
) -> List[str]:
    """
    使用 render_page 產生的 segments 座標，擷取指定類別(如 Table / Image)區域並存成圖片。
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # 依頁分組 segments
    page_to_segments = {}
    for d in docs_local:
        page_num = d.metadata.get("page_number")
        if pages is not None and page_num not in pages:
            continue
        page_to_segments.setdefault(page_num, []).append(d)

    saved_paths: List[str] = []

    for page_num in sorted(page_to_segments.keys()):
        segments = page_to_segments[page_num]

        # 依座標排序（與 convert_to_markdown 一致）
        def get_sort_key(d):
            coords = d.metadata.get("coordinates", {})
            points = coords.get("points", [])
            if points:
                y_top = min(p[1] for p in points)
                x_left = min(p[0] for p in points)
            else:
                y_top = 0
                x_left = 0
            return (y_top, x_left)

        segments = sorted(segments, key=get_sort_key)

        page = doc.load_page(page_num - 1)
        page_width_pt, page_height_pt = page.rect.width, page.rect.height

        # 依類別計數
        category_counters = {}

        for d in segments:
            seg = d.metadata
            cat = seg.get("category")

            if cat not in set(categories):
                continue

            coords = seg.get("coordinates", {})
            points = coords.get("points")
            layout_width = coords.get("layout_width")
            layout_height = coords.get("layout_height")
            if not points or not layout_width or not layout_height:
                continue

            # 依比例縮放到頁面座標
            scaled_points = [
                (x * page_width_pt / layout_width,
                 y * page_height_pt / layout_height)
                for x, y in points
            ]
            xs = [p[0] for p in scaled_points]
            ys = [p[1] for p in scaled_points]
            x_min, x_max = max(0, min(xs)), min(page_width_pt, max(xs))
            y_min, y_max = max(0, min(ys)), min(page_height_pt, max(ys))

            # 以 clip 裁切該區塊
            clip_rect = fitz.Rect(x_min, y_min, x_max, y_max)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi, dpi), clip=clip_rect)

            if pix.width <= 0 or pix.height <= 0:
                continue

            # 依類別分別計數
            category_counters[cat] = category_counters.get(cat, 0) + 1
            per_page_index = category_counters[cat]
            out_path = os.path.join(
                output_dir, f"page{page_num}_{filename_prefix}_{cat}_{per_page_index}.png"
            )
            pix.save(out_path)
            saved_paths.append(out_path)
            print(f"✅ {cat} 擷取完成: {out_path}")

    doc.close()
    return saved_paths


def convert_to_markdown(
    docs_local: List,
    output_path: str = "output.md",
    pages: List[int] = None,
    image_dir: str = "table_images",
    include_images: bool = True,
) -> str:
    """
    將 docs_local 內容轉換成 Markdown 格式
    - docs_local: 包含 page_content 和 metadata 的文件列表
    - output_path: 輸出的 Markdown 檔案路徑
    - pages: 要處理的頁面列表，None 表示處理所有頁面
    - image_dir: 圖片所在目錄（用於插入圖片連結）
    - include_images: 是否在 Markdown 中插入圖片連結
    """
    md_lines = []

    # 先過濾要處理的頁面
    filtered_docs = []
    for doc in docs_local:
        page_num = doc.metadata.get("page_number")
        if pages is None or page_num in pages:
            filtered_docs.append(doc)

    # 依頁碼和座標位置排序（由上到下、由左到右）
    def get_sort_key(doc):
        page_num = doc.metadata.get("page_number", 0)
        coords = doc.metadata.get("coordinates", {})
        points = coords.get("points", [])

        # 取得區塊的 y 座標（上方位置）和 x 座標（左側位置）
        if points:
            y_top = min(p[1] for p in points)  # 最小的 y 值（最上方）
            x_left = min(p[0] for p in points)  # 最小的 x 值（最左側）
        else:
            y_top = 0
            x_left = 0

        # 排序優先順序：頁碼 -> y座標 -> x座標
        return (page_num, y_top, x_left)

    sorted_docs = sorted(filtered_docs, key=get_sort_key)

    # 用於追蹤圖片和表格的計數器
    image_counters = {}  # {page_num: counter}

    for doc in sorted_docs:
        page_num = doc.metadata.get("page_number")
        cat = doc.metadata.get("category", "")
        text = doc.page_content.strip()

        if not text and cat not in ("Table", "Image"):
            continue

        # 處理不同類別的內容
        if cat == "Title" and text.startswith("- "):
            # 列表項目的標題
            md_lines.append(text + "\n")
        elif cat == "Title":
            # 一般標題
            md_lines.append(f"# {text}\n\n")
        elif cat == "Header":
            # 主標題
            md_lines.append(f"## {text}\n\n")
        elif cat == "Subheader":
            # 副標題
            md_lines.append(f"### {text}\n\n")
        elif cat == "Text":
            # 一般文字
            md_lines.append(f"{text}\n\n")
        elif cat == "List":
            # 列表項目
            if not text.startswith(("-", "*", "•")):
                text = f"- {text}"
            md_lines.append(f"{text}\n")
        elif cat == "Table":
            # 表格：插入對應的圖片連結
            if include_images:
                if page_num not in image_counters:
                    image_counters[page_num] = {"Table": 0, "Image": 0}
                image_counters[page_num]["Table"] += 1

                img_path = f"{image_dir}/page{page_num}_region_Table_{image_counters[page_num]['Table']}.png"
                md_lines.append(f"![Table]({img_path})\n\n")
                print(img_path)
                result = analyze_image_with_olmocr(img_path)
                md_lines.append(result)
            # 如果有表格的文字內容，也可以加入
            # if text:
            #     md_lines.append(f"*{text}*\n\n")
        elif cat == "Image":
            # 圖片：插入對應的圖片連結
            if include_images:
                if page_num not in image_counters:
                    image_counters[page_num] = {"Table": 0, "Image": 0}
                image_counters[page_num]["Image"] += 1

                # 移除時間戳記，確保檔案名稱一致
                img_path = f"{image_dir}/page{page_num}_region_Image_{image_counters[page_num]['Image']}.png"
                md_lines.append(f"![Image]({img_path})\n\n")
                result = analyze_image_with_olmocr(img_path)
                md_lines.append(result)
            # 如果有圖片說明文字
            # if text:
            #     md_lines.append(f"*{text}*\n\n")
        elif cat == "Caption":
            # 圖片或表格說明
            md_lines.append(f"*{text}*\n\n")
        elif cat == "Footer":
            # 頁尾
            md_lines.append(f"---\n*{text}*\n\n")
        else:
            # 其他類別，直接輸出
            if text:
                md_lines.append(f"{text}\n\n")

    # 寫入檔案
    markdown_content = "".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ Markdown 檔案已生成: {output_path}")
    return markdown_content

# 使用範例


def main(file_path, image_path, output_path):
    # file_path = r"C:\Users\berto\Desktop\capstone project\multi_rag\3.pdf"
    loader_local = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",              # 高分辨率模式，支持复杂文档
        infer_table_structure=True,     # 自动解析表格结构
        languages=["chi_tra", "eng"],    # 支持中英文 OCR
        ocr_engine="paddleocr"          # 指定 PaddleOCR 作为 OCR 引擎
    )

    docs_local = []
    for doc in loader_local.lazy_load():
        docs_local.append(doc)

    match = re.search(r'([^\\/]+)\.pdf$', file_path)
    filename = match.group(1)
    # 1. 先擷取圖片和表格
    image_paths = extract_regions_as_images(
        docs_local,
        file_path,
        pages=None,
        output_dir=image_path,  # r"C:\Users\berto\Desktop\capstone project\tmp\images_table"
        dpi=2,
        categories=("Table", "Image"),
        filename_prefix="region",
    )

    # 2. 轉換成 Markdown
    markdown_content = convert_to_markdown(
        docs_local,
        output_path=os.path.join(output_path, f"{filename}.md"),  # 使用傳入的輸出路徑
        pages=None,
        image_dir=image_path,  # r"C:\Users\berto\Desktop\capstone project\tmp\images_table"
        include_images=True,
    )

    print(f"\n處理完成！共擷取 {len(image_paths)} 個區域圖片")
    print(f"圖片存放路徑: {image_path}")
    print(f"MD 檔案存放路徑: {os.path.join(output_path, f'{filename}.md')}")
