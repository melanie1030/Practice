# 匯入 Streamlit 套件，用於建立網頁應用介面
import streamlit as st

# --- 頁面狀態初始化 ---
if "active_page" not in st.session_state:
    st.session_state.active_page = "main"  # 預設主頁，可依需求調整

# --- 運營診斷報告 AI 生成函數（全域可用）---
def generate_operation_diagnosis(data_summary, risk_summary):
    """生成運營診斷報告"""
    prompt = f"""作為一位跨領域的策略顧問，請根據下列 CSV 數據分析結果，撰寫一份**全面的診斷報告**，內容須結合量化指標與質化洞察，並不限於欄位層面的描述：

數據概況：
{data_summary}

風險分析：
{risk_summary}

請從以下面向進行深入分析：
1. 營運與財務現況
2. 市場與客戶洞察
3. 技術與系統現況
4. 供應鏈與流程效率
5. 人才與組織資源
6. 主要風險與緩解策略
7. 優化建議（短 / 中 / 長期行動計畫）
8. 未來發展方向與機會

請用中文回答，並保持專業、客觀的語氣。"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "你是一位擅長整合商業、技術與管理觀點的策略顧問。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成診斷報告時發生錯誤：{str(e)}"

# ================= 通用資料摘要與風險計算函式 ================= #

def summarize_dataframe_generic(df, max_columns: int = 5):
    """根據任何類型的 DataFrame 產生通用數據概況與風險指標。
    回傳 (data_summary:str, risk_metrics:dict)"""
    import numpy as np
    lines = []
    risk = {}

    # 基本結構
    lines.append(f"1. 資料筆數：{len(df)}")
    lines.append(f"2. 欄位數：{df.shape[1]}")
    col_list_preview = ", ".join(df.columns[:max_columns]) + (" ..." if df.shape[1] > max_columns else "")
    lines.append(f"3. 欄位預覽：{col_list_preview}")

    # 缺失值
    missing_ratio = df.isna().mean().mean()
    risk["整體缺失率"] = f"{missing_ratio:.2%}"
    lines.append(f"4. 整體缺失率：{missing_ratio:.2%}")

    # 數值欄統計
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        for col in num_cols[:max_columns]:
            col_series = df[col].dropna()
            if col_series.empty:
                continue
            lines.append(f"- 數值欄《{col}》：平均 {col_series.mean():.2f}，最小 {col_series.min():.2f}，最大 {col_series.max():.2f}")
    else:
        lines.append("- 無數值欄位")

    # 分類欄預覽
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:max_columns]:
            top_vals = df[col].astype(str).value_counts().nlargest(3)
            preview = ", ".join([f"{idx}({cnt})" for idx, cnt in top_vals.items()])
            lines.append(f"- 分類欄《{col}》：Top3 → {preview}")
    else:
        lines.append("- 無分類欄位")

    # 其他風險指標
    dup_ratio = df.duplicated().mean()
    risk["重複列比例"] = f"{dup_ratio:.2%}"
    if len(num_cols) > 0:
        overall_std = df[num_cols].std().mean()
        risk["數值欄平均標準差"] = f"{overall_std:.2f}"

    data_summary = "\n".join(lines)
    return data_summary, risk

# 設定 Streamlit 頁面屬性（這段必須在第一行）
st.set_page_config(
    page_title="銷貨損益分析小幫手",  # 頁面標題
    page_icon="🏪",               # 標題旁的小圖示（這裡是一家店）
    layout="wide",               # 網頁佈局為寬版
    initial_sidebar_state="expanded"  # 側邊欄預設為展開
)

# --- 側邊欄使用說明 ---
with st.sidebar:
    st.header("使用說明")
    st.markdown("""
    **完整操作指南**  
    1. **上傳主資料**：點擊「選擇檔案」上傳銷售資料 `CSV`，系統將即時解析並呈現摘要。  
       **上傳欄位資料**： 同樣方式上傳欄位說明 CSV/TXT，提高欄位解釋精度。 
    2. **資料品質儀表板**：顯示整體指標、欄位品質評估、數值/分類分佈與相關係數熱圖。 
    3. **資料探索器**：查看完整資料摘要。
    4. **CDO 初步報告**：在「CDO 報告」分頁閱讀 AI 對資料品質與異常的分析摘要。
    5. **AI 分析對話**：在「AI分析對話」分頁與 AI 對話，可即時執行並顯示表格或圖表。 
    6. **PygWalker 互動式探索**：拖放欄位即可生成圖表，並可透過 AI 問答進行分析。
    7. **運營診斷報告**：自動生成「運營診斷報告」並提供使用者下載，基於分析結果提供完整經營診斷與建議。  
    8. **諮詢服務回饋**：使用者能留下需求及需要改進的地方，AI 會即時回覆給您，並回饋給運營團隊，歡迎隨時提供您的需求。 
    """)
    st.markdown("---")

# 以下是主程式模組

# 匯入必要套件
import streamlit as st              # Streamlit：網頁應用框架
import pandas as pd                # pandas：資料處理
import os                          # os：檔案操作
import io                          # io：用於將 CSV 當字串處理
import json                        # json：JSON 格式操作（用於資料摘要）
import datetime                    # datetime：取得目前時間（用於檔案命名）
import matplotlib.pyplot as plt    # matplotlib：繪圖用
import matplotlib                   # 新增，供動態執行環境使用
import seaborn as sns              # seaborn：繪圖用（與 matplotlib 結合）
import seaborn                      # 新增，提供 seaborn 模組給動態執行環境
import numpy as np                 # numpy：數學運算

# 設定 matplotlib 使用的字體為微軟正黑體，避免中文字亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 允許座標軸顯示負號

# 匯入 html 套件，處理 HTML 內容跳脫字元
import html

# --- 使用 Plotly 套件來產生互動式圖表 ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PDF 輸出相關套件 ---
# 安裝方法：pip install reportlab xhtml2pdf
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # 預設樣式表
from reportlab.lib.units import inch                  # 定義單位
from reportlab.lib import colors                      # 顏色樣式
from reportlab.pdfbase import pdfmetrics              # 字體註冊
from reportlab.pdfbase.ttfonts import TTFont          # TrueType 字體
import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0       # 關閉字體缺失警告

# --- LangChain/LLM 人工智慧相關組件（用於文字生成分析）---
from langchain_google_genai import ChatGoogleGenerativeAI  # 使用 Google 的生成式 AI 模型
from langchain_core.prompts import PromptTemplate           # 用於建立提示詞
from langchain.memory import ConversationBufferMemory       # 建立對話記憶體


from openai import OpenAI
client = OpenAI(api_key="")


# 設置 API 金鑰
os.environ["GOOGLE_API_KEY"] = ""  # Google Gemini API 金鑰s
LLM_API_KEY = os.environ["GOOGLE_API_KEY"]  # 使用相同的金鑰作為 LLM

# 郵件設置
EMAIL_SENDER = "skeswinnie@gmail.com"
EMAIL_PASSWORD = "dkyu hpmy tpai rjwf"

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定全域 CSS 樣式，使用 markdown 與 HTML 的方式載入樣式設定
st.markdown("""
    <style>
    /* 背景動畫設定 */
    /* 主要內容區域背景 */
    .stApp {
        background: linear-gradient(-45deg, #FFD1DC, #E0FFE0, #D1E8FF, #E6E6FA); /* 粉色系漸層背景 */
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;  /* 背景動畫：15秒循環 */
        padding: 2rem;
    }

    /* 定義動畫的關鍵影格 */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    
    /* 側邊欄樣式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(255, 209, 220, 0.95), rgba(230, 230, 250, 0.3));
        border-right: 1px solid rgba(209, 232, 255, 0.6);
        box-shadow: 2px 0 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    section[data-testid="stSidebar"]:hover {
        box-shadow: 2px 0 20px rgba(135, 206, 235, 0.2);
    }

    /* 側邊欄內部元素 */
    section[data-testid="stSidebar"] .stMarkdown {
        background-color: transparent;
        transition: all 0.3s ease;
    }
    
    /* 標題樣式 */
    h1, h2, h3, h4, h5, h6 {
        color: #2E4053;
        font-weight: 700;
        padding: 0.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.3));
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    h1:hover, h2:hover, h3:hover, h4:hover, h5:hover, h6:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(135, 206, 235, 0.3);
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.95), rgba(131, 164, 212, 0.2));
    }

    /* 按鈕樣式 */
    .stButton > button {
        background: linear-gradient(120deg, #87CEEB, #ADD8E6);
        color: #2E4053;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 4px 15px rgba(135, 206, 235, 0.4);
        background: linear-gradient(120deg, #ADD8E6, #87CEEB);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* 添加彩虹邊框效果 */
    @keyframes borderGlow {
        0% {
            border-color: #87CEEB;
        }
        25% {
            border-color: #ADD8E6;
        }
        50% {
            border-color: #B6FBFF;
        }
        75% {
            border-color: #FFD700;
        }
        100% {
            border-color: #87CEEB;
        }
    }

    /* 其他樣式保持不變 */
    /* 數據框樣式 */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border: none !important;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .dataframe:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(255, 154, 139, 0.3);
    }

    .dataframe th {
        background-color: #faf9f6;
        color: #333333;
        font-weight: 600;
        padding: 12px !important;
        transition: all 0.3s ease;
    }

    .dataframe td {
        color: #333333;
        border: none !important;
        border-bottom: 1px solid rgba(182, 251, 255, 0.3) !important;
        padding: 12px !important;
        transition: all 0.2s ease;
    }

    .dataframe tr:hover {
        background-color: rgba(131, 164, 212, 0.1);
        transform: scale(1.01);
    }

    /* 圖表容器 */
    .stPlot {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(131, 164, 212, 0.1));
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .stPlot:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 5px 15px rgba(255, 154, 139, 0.3);
    }

    /* 選擇器樣式 */
    .stSelectbox {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 154, 139, 0.1));
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stSelectbox:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 106, 136, 0.2);
    }

    /* 指標卡片樣式 */
    div[data-testid="stMetricValue"] {
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.2));
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 4px 15px rgba(131, 164, 212, 0.3);
    }

    /* Tab 樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(131, 164, 212, 0.1));
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab-list"]:hover {
        box-shadow: 0 4px 15px rgba(255, 106, 136, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.9), rgba(182, 251, 255, 0.2));
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #2E4053;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.95), rgba(131, 164, 212, 0.2));
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(120deg, #87CEEB, #ADD8E6);
        border-radius: 6px;
        transition: all 0.3s ease;
    }

    /* 添加全局動畫效果 */
    * {
        transition: background-color 0.3s ease;
    }

    /* 添加載入動畫 */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# create_figure_layout 函數說明
# 建立圖表的統一樣式佈局
def create_figure_layout():
    return {
        'plot_bgcolor': 'white',     # 圖表內部背景顏色
        'paper_bgcolor': 'white',    # 整張圖表的背景顏色
        'xaxis_title': "",           # X 軸標題留空
        'yaxis_title': "",           # Y 軸標題留空
        'showlegend': False          # 不顯示圖例
    }


# 新增表格和文字樣式
st.markdown("""
    <style>
    /* 標題文字樣式 */
    h1, h2, h3, h4, h5, h6 {
        color: #333333;
        font-weight: 600;
    }

    /* 一般文字樣式 */
    p, span, div {
        color: #333333;
    }

    /* 表格樣式 */
    .dataframe {
        color: #333333;
        background-color: #ffffff;
    }

    .dataframe th {
        background-color: #faf9f6;
        color: #333333;
        font-weight: 600;
        padding: 8px;
    }

    .dataframe td {
        color: #333333;
        padding: 8px;
    }

    /* 表格hover效果 */
    .dataframe tr:hover {
        background-color: #faf9f6;
    }

    /* 數據標籤樣式 */
    .metric-label {
        color: #333333;
        font-weight: 500;
    }

    /* 圖表標題和軸標籤 */
    .plot-container text {
        color: #333333 !important;
        fill: #333333 !important;
    }

    /* Streamlit特定元素樣式 */
    .stMarkdown, .stText {
        color: #333333;
    }

    .st-bb {
        color: #333333;
    }

    .st-bw {
        color: #333333;
    }

    /* 連結顏色 */
    a {
        color: #666666;
    }

    a:hover {
        color: #333333;
    }

    /* 強調文字 */
    .highlight {
        color: #666666;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)    

# --- 組態設定區段 ---

# 從環境變數中取得 LLM API 金鑰，若無則使用預設金鑰
LLM_API_KEY = os.environ.get("LLM_API_KEY", "AIzaSyBXlGN8R-LJiUgG09A8OVnvSthnNZYo0YI")  # 預設 API 金鑰

# 資料儲存目錄，加入 AI 分析子資料夾
TEMP_DATA_STORAGE = "temp_data_simplified_agent/AI analysis/"
os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)  # 若資料夾不存在則自動建立

# 可用的模型清單
AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-05-06"]
DEFAULT_WORKER_MODEL = "gemini-2.0-flash-lite"   # 預設資料分析模型
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"         # 預設判斷/評估模型

# 若未設定有效的 API 金鑰，則顯示錯誤提示
if LLM_API_KEY == "YOUR_API_KEY_HERE" or LLM_API_KEY == "API PLZ":
    st.error("請設定您的 LLM_API_KEY（例如 GOOGLE_API_KEY），可透過環境變數或 Streamlit Secret 設定，以啟用完整功能。")


# PlaceholderLLM 模擬模型（當 API 無效時使用）
# --- LLM 初始化與佔位模擬 ---
class PlaceholderLLM:
    """在 API 金鑰不可用時模擬 LLM 回應。"""

    def __init__(self, model_name="placeholder_model"):
        self.model_name = model_name
        st.warning(f"由於 API 金鑰未設置或無效，正在使用 {self.model_name} 的模擬模型 PlaceholderLLM。")

    def invoke(self, prompt_input):
        # 將提示轉換成字串（若有 to_string 方法則使用）
        prompt_str_content = str(prompt_input.to_string() if hasattr(prompt_input, 'to_string') else prompt_input)

        # 模擬 CDO 初步描述資料集
        if "CDO, your first task is to provide an initial description of the dataset" in prompt_str_content:
            data_summary_json = {}
            try:
                summary_marker = "Data Summary (for context):"
                if summary_marker in prompt_str_content:
                    json_str_part = prompt_str_content.split(summary_marker)[1].split("\n\nDetailed Initial Description by CDO:")[0].strip()
                    data_summary_json = json.loads(json_str_part)
            except Exception:
                pass  # 忽略解析失敗

            cols = data_summary_json.get("columns", ["N/A"])
            num_rows = data_summary_json.get("num_rows", "N/A")
            num_cols = data_summary_json.get("num_columns", "N/A")
            user_desc = data_summary_json.get("user_provided_column_descriptions", "使用者尚未提供欄位描述")
            dtypes_str = "\n".join(
                [f"- {col}: {data_summary_json.get('dtypes', {}).get(col, '未知')}" for col in cols])

            return {"text": f"""
*模擬版 CDO 資料初步描述（模型：{self.model_name}）*

**1. 資料集概覽（模擬 df.info()）**
   - 列數：{num_rows}，欄數：{num_cols}
   - 各欄位資料型態：
{dtypes_str}
   - 預估記憶體使用量：（模擬值）MB

**2. 使用者提供的欄位說明**
   - {user_desc}

**3. 變數可能含意（範例）**
   - `ORDERNUMBER`：每筆訂單的唯一識別碼
   - `QUANTITYORDERED`：該訂單中某項商品的數量
   （以上為範例解釋，實際含義視使用者資料而定）

**4. 初步資料品質評估（範例）**
   - **缺失值**：如「欄位 'ADDRESSLINE2' 有 80% 缺值」
   - **整體結構**：資料結構基本完整。
"""}

        # 模擬 CEO、CFO、CDO 部門觀點
        elif "panel of expert department heads, including the CDO" in prompt_str_content:
            return {"text": f"""
*模擬各部門主管觀點（基於 {self.model_name} 模型）*

**執行長（CEO）**：關注營收趨勢，並考慮欄位意義。
**財務長（CFO）**：評估各地區利潤，結合使用者說明。
**資料長（CDO）**：注意缺值與使用者補充的欄位解釋。
"""}

        # 模擬整合後的分析策略
        elif "You are the Chief Data Officer (CDO) of the company." in prompt_str_content and "synthesize these diverse perspectives" in prompt_str_content:
            return {"text": f"""
*模擬版分析策略綜合結果（由 CDO 統整，模型：{self.model_name}）*

1.  **視覺化核心銷售趨勢**：繪製 'SALES' 對 'ORDERDATE' 折線圖（依使用者描述判讀 SALES）
2.  **商品線表現表格**：按 'PRODUCTLINE' 列出 'SALES'、'PRICEEACH'、'QUANTITYORDERED'
3.  **訂單狀態描述**：統計 'STATUS' 欄位的數量
4.  **主要欄位的資料品質表格**：列出缺值比例
5.  **按國家顯示銷售額**：繪製 'SALES' 對 'COUNTRY' 長條圖
"""}

        # 模擬程式碼生成邏輯
        elif "Python code:" in prompt_str_content and "User Query:" in prompt_str_content:
            user_query_segment = prompt_str_content.split("User Query:")[1].split("\n")[0].lower()

            fallback_script = """
# 標準函式庫
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os

analysis_result = "已執行分析邏輯。如預期有特定輸出，請確認生成的程式碼內容。"
plot_data_df = None

# --- AI 產生程式碼區 ---
# Placeholder：AI 會在這裡填入分析邏輯
# --- AI 程式碼區結束 ---

if 'analysis_result' not in locals() or (isinstance(analysis_result, str) and analysis_result == "已執行分析邏輯。如預期有特定輸出，請確認生成的程式碼內容。"):
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "腳本執行完成。未設置特定輸出變數 'analysis_result'，預設顯示 df.head()。"
        plot_data_df = df.head().copy()
    else:
        analysis_result = "腳本執行完成。未設置 'analysis_result'，且無可用資料框架（df）。"
"""

            # 回傳平均值分析腳本
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['SALES'].mean()\nplot_data_df = None"}

            # 回傳圖表繪製腳本
            elif "plot" in user_query_segment or "visualize" in user_query_segment:
                placeholder_plot_filename = "placeholder_plot.png"
                placeholder_full_save_path = os.path.join(TEMP_DATA_STORAGE, placeholder_plot_filename).replace("\\", "/")

                generated_plot_code = f"""..."""  # 這部分是整段圖表備援腳本，已在原始碼詳列（略）

                return {"text": generated_plot_code}

            # 回傳描述統計表格
            elif "table" in user_query_segment or "summarize" in user_query_segment:
                return {"text": "analysis_result = df.describe()\nplot_data_df = df.describe().reset_index()"}

            # 預設備援腳本
            else:
                return {"text": fallback_script}

        # 模擬文字報告
        elif "Generate a textual report" in prompt_str_content:
            return {
                "text": f"### 模擬分析報告（模型：{self.model_name}）\n此為根據 CDO 分析策略與使用者欄位說明所產出的佔位報告。"}

        # 模擬分析批判意見
        elif "Critique the following analysis artifacts" in prompt_str_content:
            return {"text": f"""
### 模擬分析評論（模型：{self.model_name}）
**整體評估**：模擬內容。分析應反映使用者提供的欄位說明。
**Python 程式碼**：模擬。
**資料內容**：模擬。
**報告內容**：模擬。
**建議（給 AI 模型）**：模擬，應納入使用者背景脈絡。
"""}

        # 模擬 HTML 報告產出
        elif "Generate a single, complete, and runnable HTML file" in prompt_str_content:
            return {"text": """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>模擬 Bento 分析報告</title>
    <style>...</style>
</head>
<body>
    <div class="bento-grid">
        <div class="bento-item"><h2>分析目標</h2><p>模擬：使用者輸入的問題會顯示在此。</p></div>
        <div class="bento-item"><h2>資料摘要與使用者說明</h2><p>模擬：CDO 初步資料說明將出現在此。</p></div>
        <div class="bento-item" style="grid-column: span 2;">
            <h2>重要資料品質警示</h2><p>模擬：此處顯示遺漏值或品質問題。</p>
        </div>
        <div class="bento-item"><h2>可行的洞察</h2><p>模擬：分析結論與可行建議。</p></div>
        <div class="bento-item"><h2>分析批評摘要</h2><p>模擬：AI 對分析結果的評論。</p></div>
    </div>
</body>
</html>"""}

        # 預設無法辨識提示時
        else:
            return {
                "text": f"{self.model_name} 的模擬回應：無法辨識提示，前 200 字如下：\n{prompt_str_content[:200]}..."}


def get_llm_instance(model_name: str):
    """
    取得或初始化一個 LLM 實例。
    使用緩存（st.session_state.llm_cache）存儲已初始化的模型。
    如果 API 金鑰未設置或初始化失敗，則使用 PlaceholderLLM。
    """
    if not model_name:
        st.error("未提供 LLM 初始化的模型名稱。")
        return None
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    if model_name not in st.session_state.llm_cache:
        if not LLM_API_KEY or LLM_API_KEY == "LLM_API_KEY" or LLM_API_KEY == "YOUR_API_KEY_HERE" or LLM_API_KEY == "API PLZ":
            st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
        else:
            try:
                # Set temperature based on whether the model is a judge model or worker model
                temperature = 0.7 if st.session_state.get("selected_judge_model", "") == model_name else 0.2
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=LLM_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True  # Important for some models/versions
                )
                st.session_state.llm_cache[model_name] = llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM ({model_name}): {e}")
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
    return st.session_state.llm_cache[model_name]



# 使用 Streamlit 快取裝飾器，避免重複運算同樣資料（可提升效能）
@st.cache_data
def calculate_data_summary(df_input, user_column_descriptions_content=None):
    """
    計算輸入 DataFrame 的全面摘要。
    包含：行/列數、欄位型態、缺失值狀況、描述性統計、預覽前後資料列，
    以及整合使用者提供的欄位描述（若有提供）。
    """
    if df_input is None or df_input.empty:
        return None

    # 建立一份副本，避免更動原始資料
    df = df_input.copy()

    # 建立摘要字典
    data_summary = {
        "num_rows": len(df),  # 總行數
        "num_columns": len(df.columns),  # 總欄數
        "columns": df.columns.tolist(),  # 欄位名稱
        "dtypes": {col: str(df[col].dtype) for col in df.columns},  # 各欄位的資料型態
        "missing_values_total": int(df.isnull().sum().sum()),  # 總缺失值數量
        "missing_values_per_column": df.isnull().sum().to_dict(),  # 每欄的缺失值數量
        "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",  # 描述統計資料（含所有型別）
        "preview_head": df.head().to_dict(orient='records'),  # 前五筆預覽資料（轉成字典列表）
        "preview_tail": df.tail().to_dict(orient='records'),  # 後五筆預覽資料
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),  # 數值型欄位
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()  # 類別型欄位
    }

    # 計算總缺失值的百分比
    data_summary["missing_values_percentage"] = (
        data_summary["missing_values_total"] /
        (data_summary["num_rows"] * data_summary["num_columns"])
    ) * 100 if (data_summary["num_rows"] * data_summary["num_columns"]) > 0 else 0

    # 整合使用者提供的欄位說明（若有）
    if user_column_descriptions_content:
        data_summary["user_provided_column_descriptions"] = user_column_descriptions_content

    return data_summary



def load_csv_and_get_summary(uploaded_csv_file, uploaded_desc_file=None):
    """
    載入使用者上傳的 CSV 資料與（可選）描述檔案，
    產生資料摘要並更新 Streamlit 的會話狀態。
    同時重置 CDO 分析流程與相關變數。
    """
    try:
        # 讀取 CSV 檔案為 DataFrame
        df = pd.read_csv(uploaded_csv_file)
        st.session_state.current_dataframe = df  # 將目前資料存入會話狀態
        st.session_state.data_source_name = uploaded_csv_file.name  # 記錄來源檔名

        user_column_descriptions_content = None
        if uploaded_desc_file:
            try:
                # 讀入說明檔案並解碼（Streamlit 上傳檔案通常為位元組）
                user_column_descriptions_content = uploaded_desc_file.getvalue().decode('utf-8')
                st.session_state.desc_file_name = uploaded_desc_file.name
            except Exception as e:
                st.error(f"讀取描述檔錯誤「{uploaded_desc_file.name}」：{e}")
                # 預設為警告，不中斷流程
        else:
            st.session_state.desc_file_name = None  # 沒有描述檔時清空狀態

        st.session_state.show_pygwalker = False

        # 清除先前產生的分析結果
        st.session_state.current_analysis_artifacts = {}

        # 計算新的摘要
        summary_for_state = calculate_data_summary(df.copy(), user_column_descriptions_content)

        if summary_for_state:
            summary_for_state["source_name"] = uploaded_csv_file.name  # 加入檔名資訊
        st.session_state.data_summary = summary_for_state  # 儲存摘要

        # 重置 CDO 分析流程狀態
        st.session_state.cdo_initial_report_text = None
        st.session_state.other_perspectives_text = None
        st.session_state.strategy_text = None
        if "cdo_workflow_stage" in st.session_state:
            del st.session_state.cdo_workflow_stage

        return True
    except Exception as e:
        st.error(f"載入 CSV 或產生摘要時發生錯誤：{e}")
        st.session_state.current_dataframe = None
        st.session_state.data_summary = None
        return False



@st.cache_data
def get_overview_metrics(df):
    """
    計算資料表的整體指標：
    回傳包括行數、列數、缺失值百分比、數值欄數、重複列數等。
    """
    if df is None or df.empty:
        return 0, 0, 0, 0, 0

    num_rows = len(df)  # 行數
    num_cols = len(df.columns)  # 列數
    missing_values_total = df.isnull().sum().sum()  # 總缺失值
    total_cells = num_rows * num_cols  # 總儲存格數
    missing_percentage = (missing_values_total / total_cells) * 100 if total_cells > 0 else 0  # 缺失百分比
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)  # 數值型欄位數
    duplicate_rows = df.duplicated().sum()  # 重複列數

    return num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows



@st.cache_data
def get_column_quality_assessment(df_input):
    """
    對 DataFrame 中的每個欄位進行資料品質評估（最多顯示前 max_cols_to_display 欄）。
    會計算：資料型態、缺失百分比、唯一值數量、數值範圍/常見值，以及品質分數。
    最後回傳一個 DataFrame 格式的評估結果表。
    """
    if df_input is None or df_input.empty:
        return pd.DataFrame()
    df = df_input.copy()
    quality_data = []
    max_cols_to_display = 10  # 僅評估前10欄（避免UI太慢）

    for col in df.columns[:max_cols_to_display]:
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_values = df[col].nunique()
        range_common = ""  # 顯示範圍或常見值

        # 根據資料型態決定顯示方式
        if pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].dropna().empty:
                range_common = f"最小值: {df[col].min():.2f}, 最大值: {df[col].max():.2f}"
            else:
                range_common = "無法評估（皆為缺值）"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if not df[col].dropna().empty:
                range_common = f"最小時間: {df[col].min()}, 最大時間: {df[col].max()}"
            else:
                range_common = "無法評估（皆為缺值）"
        else:
            if not df[col].dropna().empty:
                common_vals = df[col].mode().tolist()
                range_common = f"最常見值: {', '.join(map(str, common_vals[:3]))}"
                if len(common_vals) > 3:
                    range_common += "..."
            else:
                range_common = "無法評估（皆為缺值）"

        # 計算品質分數（簡易方式）
        score = 10
        if missing_percent > 50:
            score -= 5
        elif missing_percent > 20:
            score -= 3
        elif missing_percent > 5:
            score -= 1
        if unique_values == 1 and len(df) > 1:
            score -= 2  # 若為常數值（只有一種），扣分
        if unique_values == len(df) and not pd.api.types.is_numeric_dtype(df[col]):
            score -= 1  # 若每筆值都不同，且非數值欄位，可能是 ID，也扣分

        quality_data.append({
            "欄位名稱": col,
            "資料型態": dtype,
            "缺值比例": f"{missing_percent:.2f}%",
            "唯一值數量": unique_values,
            "範圍 / 常見值": range_common,
            "品質評分（滿分10分）": max(0, score)
        })

    return pd.DataFrame(quality_data)



def generate_data_quality_dashboard(df_input):
    """
    使用 Streamlit 生成資料品質儀表板，
    包含：整體概覽指標、欄位品質評估、數值欄分佈圖、分類欄分佈圖、數值欄相關係數熱圖。
    """
    if df_input is None or df_input.empty:
        st.warning("⚠️ 尚未載入資料或 DataFrame 為空，請先上傳 CSV 檔案。")
        return

    df = df_input.copy()
    st.header("📊 資料品質儀表板")
    st.markdown("以下為本資料集的品質與特徵概覽：")

    # --- 關鍵指標 ---
    st.subheader("關鍵資料集指標")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df.copy())
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("總行數", f"{num_rows:,}")
    col2.metric("總欄數", f"{num_cols:,}")
    if missing_percentage > 5:
        col3.metric("缺值比例", f"{missing_percentage:.2f}%", delta_color="inverse",
                    help="整體資料集中缺值所占百分比。超過 5% 顯示紅色警告。")
    else:
        col3.metric("缺值比例", f"{missing_percentage:.2f}%",
                    help="整體資料集中缺值所占百分比。")
    col4.metric("數值欄數", f"{numeric_cols_count:,}")
    col5.metric("重複列數", f"{duplicate_rows:,}", help="完全重複的列數量")
    st.markdown("---")

    # --- 各欄位品質評估 ---
    st.subheader("各欄位資料品質評估")
    if len(df.columns) > 10:
        st.caption(f"⚠️ 僅顯示前 10 欄，實際共 {len(df.columns)} 欄。完整報告請見 PDF。")

    quality_df = get_column_quality_assessment(df.copy())

    if not quality_df.empty:
        # 根據缺值與品質分數加上顏色
        def style_quality_table(df_to_style):
            return df_to_style.style.apply(
                lambda row: ['background-color: #FFCDD2' if float(str(row["缺值比例"]).replace('%', '')) > 20
                             else ('background-color: #FFF9C4' if float(str(row["缺值比例"]).replace('%', '')) > 5 else '')
                             for _ in row], axis=1, subset=["缺值比例"]
            ).apply(
                lambda row: ['background-color: #FFEBEE' if row["品質評分（滿分10分）"] < 5
                             else ('background-color: #FFFDE7' if row["品質評分（滿分10分）"] < 7 else '')
                             for _ in row], axis=1, subset=["品質評分（滿分10分）"]
            )

        st.dataframe(style_quality_table(quality_df), use_container_width=True)
    else:
        st.info("⚠️ 無法產生欄位品質評估表格。")

    from datetime import datetime

    # 確保初始化 current_time
    if "current_time" not in st.session_state:
        st.session_state.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import numpy as np
    import plotly.express as px

    # 註冊中文字型
    font_path = r"C:\Users\ASUS\OneDrive\桌面\專題\text\fonts\SourceHanSansTW-Regular.ttf"
    try:
        pdfmetrics.registerFont(TTFont('CustomFont', font_path))
        default_font = 'CustomFont'
    except:
        st.warning("⚠️ 中文字型載入失敗，請檢查路徑與字型格式")
        default_font = 'Helvetica'

    # 生成資料品質 PDF 報告
    def generate_pdf_report(df, file_name="data_quality_report.pdf"):
        c = canvas.Canvas(file_name, pagesize=letter)
        width, height = letter

        # 標題
        c.setFont(default_font, 16)
        c.drawString(100, height - 40, u"資料品質報告")
        c.setFont(default_font, 12)
        c.drawString(100, height - 80, f"報告日期: {st.session_state.current_time}")

        y_position = height - 120

        # --- 關鍵資料集指標 ---
        c.setFont(default_font, 12)
        c.drawString(100, y_position, "關鍵資料集指標:")
        y_position -= 20

        c.setFont(default_font, 10)
        c.drawString(100, y_position, f"總行數: {df.shape[0]}")
        y_position -= 15
        c.drawString(100, y_position, f"總欄數: {df.shape[1]}")
        y_position -= 15
        missing_percentage = df.isnull().mean().mean() * 100
        c.drawString(100, y_position, f"缺值比例: {missing_percentage:.2f}%")
        y_position -= 15
        numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
        c.drawString(100, y_position, f"數值欄數: {numeric_cols_count}")
        y_position -= 15
        duplicate_rows = df.duplicated().sum()
        c.drawString(100, y_position, f"重複列數: {duplicate_rows}")
        y_position -= 25

        # --- 各欄位資料品質評估 ---
        c.setFont(default_font, 12)
        c.drawString(100, y_position, "各欄位資料品質評估:")
        y_position -= 20

        quality_df = get_column_quality_assessment(df)

        if not quality_df.empty:
            for idx, row in quality_df.iterrows():
                c.setFont(default_font, 10)
                c.drawString(100, y_position, f"{row['欄位名稱']}: 缺值比例 {row['缺值比例']}, 品質評分 {row['品質評分（滿分10分）']}")
                y_position -= 15
        else:
            c.drawString(100, y_position, "無法產生欄位品質評估表格")
            y_position -= 20

        y_position -= 25

        # --- 數值欄位分佈 ---
        c.setFont(default_font, 12)
        c.drawString(100, y_position, "數值欄位分佈圖:")
        y_position -= 20

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            c.setFont(default_font, 10)
            c.drawString(100, y_position, f"{col} 分佈：")
            y_position -= 15
            c.drawString(100, y_position, f"平均數: {df[col].mean():.2f}, 中位數: {df[col].median():.2f}")
            y_position -= 15

        # --- 類別欄位分佈 ---
        c.setFont(default_font, 12)
        c.drawString(100, y_position, "類別欄位分佈圖:")
        y_position -= 20

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            c.setFont(default_font, 10)
            c.drawString(100, y_position, f"{col} 分佈：")
            y_position -= 15
            value_counts = df[col].value_counts(normalize=True).mul(100).round(2)
            for idx, value in value_counts.items():
                c.drawString(100, y_position, f"{idx}: {value}%")
                y_position -= 15

        # --- 數值欄位相關係數熱圖 ---
        c.setFont(default_font, 12)
        c.drawString(100, y_position, "數值欄位相關係數熱圖:")
        y_position -= 20

        numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols_for_corr) < 2:
            c.drawString(100, y_position, "⚠️ 至少需兩個數值欄位才能繪製相關係數熱圖。")
            y_position -= 20
        else:
            try:
                corr_matrix = df[numeric_cols_for_corr].corr()
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                        color_continuous_scale='RdBu_r',
                                        title="數值欄位相關係數熱圖")
                fig_heatmap.update_xaxes(side="bottom")
                fig_heatmap.update_layout(xaxis_tickangle=-45, yaxis_tickangle=0)
                fig_heatmap.write_image("corr_heatmap.png")
                c.drawImage("corr_heatmap.png", 100, y_position, width=400, height=200)
                y_position -= 220
            except ValueError as e:
                c.drawString(100, y_position, f"相關係數計算錯誤: {e}")
                y_position -= 20

        # 儲存 PDF
        c.save()
        st.success(f"資料品質報告已生成：{file_name}")

        # --- 下載按鈕 ---
        with open(file_name, "rb") as pdf_file:
            st.download_button(
                label="下載資料品質報告 PDF",
                data=pdf_file,
                file_name=file_name,
                mime="application/pdf"
            )


    # 更新現有程式碼，讓按鈕觸發 PDF 報告生成
    if st.button("產生完整資料品質 PDF 報告", key="dq_pdf_placeholder"):
        generate_pdf_report(df)

    st.markdown("---")

    # --- 數值欄位分佈 ---
    st.subheader("數值欄位分佈圖")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("⚠️ 本資料集中未偵測到數值型欄位。")
    else:
        selected_numeric_col = st.selectbox("請選擇欲分析分佈的數值欄位：", numeric_cols, key="dq_numeric_select")
        if selected_numeric_col:
            col_data = df[selected_numeric_col].dropna()
            if not col_data.empty:
                fig = px.histogram(col_data, x=selected_numeric_col, marginal="box",
                                   title=f"{selected_numeric_col} 的分佈圖", opacity=0.75,
                                   histnorm='probability density')
                fig.add_trace(go.Scatter(x=col_data, y=[0] * len(col_data),
                                         mode='markers',
                                         marker=dict(color='rgba(0,0,0,0)'), showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**主要統計指標：**")
                stats_cols = st.columns(5)
                stats_cols[0].metric("平均數", f"{col_data.mean():.2f}")
                stats_cols[1].metric("中位數", f"{col_data.median():.2f}")
                stats_cols[2].metric("標準差", f"{col_data.std():.2f}")
                stats_cols[3].metric("最小值", f"{col_data.min():.2f}")
                stats_cols[4].metric("最大值", f"{col_data.max():.2f}")
            else:
                st.info(f"⚠️ 欄位 '{selected_numeric_col}' 僅包含缺失值，無法繪製圖表。")

    st.markdown("---")

    # --- 類別欄位分佈 ---
    st.subheader("類別欄位分佈圖")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        st.info("⚠️ 資料集中未偵測到類別型欄位。")
    else:
        selected_categorical_col = st.selectbox("請選擇欲分析分佈的類別欄位：",
                                                categorical_cols, key="dq_categorical_select")
        if selected_categorical_col:
            col_data = df[selected_categorical_col].dropna()
            if not col_data.empty:
                value_counts = col_data.value_counts(normalize=True).mul(100).round(2)
                count_abs = col_data.value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"{selected_categorical_col} 的分佈情形",
                             labels={'x': selected_categorical_col, 'y': '百分比 (%)'},
                             text=[f"{val:.1f}%（{count_abs[idx]} 筆）" for idx, val in value_counts.items()])
                fig.update_layout(xaxis_title=selected_categorical_col, yaxis_title="百分比 (%)")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"⚠️ 欄位 '{selected_categorical_col}' 僅包含缺失值，無法繪製圖表。")

    st.markdown("---")

    # --- 數值欄位相關係數熱圖 ---
    st.subheader("數值欄位相關係數熱圖")
    numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr) < 2:
        st.info("⚠️ 至少需兩個數值欄位才能繪製相關係數熱圖。")
    else:
        corr_matrix = df[numeric_cols_for_corr].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                color_continuous_scale='RdBu_r',
                                title="數值欄位相關係數熱圖")
        fig_heatmap.update_xaxes(side="bottom")
        fig_heatmap.update_layout(xaxis_tickangle=-45, yaxis_tickangle=0)
        st.plotly_chart(fig_heatmap, use_container_width=True)



class LocalCodeExecutionEngine:
    """
    在受控環境中執行 Python 程式碼字串。
    程式碼預期會對名為 'df' 的 pandas DataFrame 進行處理。
    支援擷取結果、圖表與錯誤資訊。
    """

    def execute_code(self, code_string, df_input):
        """
        執行提供的 Python 程式碼字串。

        參數：
            code_string (str)：欲執行的 Python 程式碼字串。
            df_input (pd.DataFrame)：會以 'df' 的變數名注入程式碼作用域的資料表。

        回傳：
            dict：包含執行類型與結果的字典，可能包含：錯誤訊息、圖表檔案、表格資料、文字內容等。
        """
        if df_input is None:
            return {"type": "error", "message": "未載入任何資料，無法執行程式碼。"}

        # 建立安全的執行環境
        exec_globals = globals().copy()
        exec_globals['plt'] = matplotlib.pyplot
        exec_globals['sns'] = seaborn
        exec_globals['pd'] = pd
        exec_globals['np'] = np
        exec_globals['os'] = os

        # 建立本地作用域，注入資料與常用模組
        local_scope = {
            'df': df_input.copy(),
            'pd': pd,
            'plt': matplotlib.pyplot,
            'sns': seaborn,
            'np': np,
            'os': os,
            'TEMP_DATA_STORAGE': TEMP_DATA_STORAGE
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None
        default_analysis_result_message = "程式碼已執行，但尚未設定 'analysis_result' 結果變數。"

        # 初始化輸出變數
        local_scope['analysis_result'] = default_analysis_result_message
        local_scope['plot_data_df'] = None

        try:
            os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)  # 確保目錄存在
            exec(code_string, exec_globals, local_scope)  # 執行使用者代碼

            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')

            # 若未設定結果變數，給予警告
            if isinstance(analysis_result, str) and analysis_result == default_analysis_result_message:
                st.warning("⚠️ 程式碼未設定 'analysis_result' 結果變數，建議明確指定。")

            # 若為錯誤訊息開頭
            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}

            # 處理圖表結果（若 analysis_result 是圖片檔名）
            if isinstance(analysis_result, str) and any(analysis_result.lower().endswith(ext)
                                                        for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                plot_filename = os.path.basename(analysis_result)
                final_plot_path = os.path.join(TEMP_DATA_STORAGE, plot_filename)

                if not os.path.exists(final_plot_path):
                    # 若給的是完整路徑，仍嘗試使用
                    if os.path.isabs(analysis_result) and os.path.exists(analysis_result):
                        final_plot_path = analysis_result
                        st.warning(f"⚠️ 圖表以絕對路徑儲存：{analysis_result}，建議僅傳入檔名。")
                    else:
                        # 嘗試自動將目前的活躍圖表儲存至預期路徑
                        try:
                            import matplotlib.pyplot as _plt_autosave
                            if _plt_autosave.get_fignums():
                                _plt_autosave.gcf().savefig(final_plot_path, bbox_inches="tight")
                                st.info(f"✅ 偵測到未儲存圖表，已自動將當前圖表存為：{final_plot_path}")
                            else:
                                raise FileNotFoundError
                        except Exception:
                            return {"type": "error", "message": f"找不到圖表檔案 '{plot_filename}'，"
                                                           f"請確認 AI 使用 `os.path.join(TEMP_DATA_STORAGE, '檔名')` "
                                                           f"儲存圖檔，並將 `analysis_result` 設為純檔名。"}

                # 若有繪圖用資料表，另行儲存
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(plot_filename)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                    st.info(f"📂 圖表對應資料已儲存於：{plot_data_df_saved_path}")
                elif plot_data_df is not None:
                    st.warning("⚠️ `plot_data_df` 已定義但不是有效的 DataFrame，無法儲存。")

                return {
                    "type": "plot",
                    "plot_path": final_plot_path,
                    "data_path": plot_data_df_saved_path
                }

            # 處理表格結果（若為 DataFrame 或 Series）
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                analysis_result_df = (analysis_result.to_frame()
                                      if isinstance(analysis_result, pd.Series)
                                      else analysis_result)
                if analysis_result_df.empty:
                    return {"type": "text", "value": "📄 分析結果為空表格。"}
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, f"table_result_{timestamp}.csv")
                analysis_result_df.to_csv(saved_csv_path, index=False)
                return {
                    "type": "table",
                    "data_path": saved_csv_path,
                    "dataframe_result": analysis_result_df
                }

            # 若為純文字輸出
            else:
                return {"type": "text", "value": str(analysis_result)}

        except Exception as e:
            import traceback
            error_message_for_user = f"❌ 程式執行錯誤：{str(e)}\n🔍 追蹤記錄：\n{traceback.format_exc()}"
            current_analysis_res = local_scope.get('analysis_result', default_analysis_result_message)
            if current_analysis_res is None or (
                    isinstance(current_analysis_res, pd.DataFrame) and current_analysis_res.empty):
                local_scope['analysis_result'] = f"執行錯誤：{str(e)}"
            return {
                "type": "error",
                "message": error_message_for_user,
                "final_analysis_result_value": local_scope['analysis_result']
            }


# 建立執行器實例供使用
code_executor = LocalCodeExecutionEngine()



# --- 將分析結果匯出為 PDF 文件 ---
def export_analysis_to_pdf(artifacts, output_filename="analysis_report.pdf"):
    """
    將分析結果（查詢、CDO 報告、圖表、數據、報告文字、評論）導出為 PDF 文件。

    參數：
        artifacts (dict)：包含分析資料的字典，內含路徑與內容。
        output_filename (str)：輸出的 PDF 文件名稱（含副檔名）。
    回傳：
        str：成功生成的 PDF 檔案路徑；若失敗則回傳 None。
    """
    # 註冊中文字體 (嘗試多種常見中文字體)
    chinese_font_available = False
    chinese_font_name = ''
    
    # 嘗試載入不同的中文字體，按優先順序
    font_options = [
        ("MicrosoftJhengHei", "c:/windows/fonts/msjh.ttc"),  # 微軟正黑體
        ("MicrosoftJhengHei", "c:/windows/fonts/msjhbd.ttc"), # 微軟正黑體粗體
        ("DFKaiShu", "c:/windows/fonts/kaiu.ttf"),           # 標楷體
        ("MingLiU", "c:/windows/fonts/mingliu.ttc"),         # 細明體
        ("SimSun", "c:/windows/fonts/simsun.ttc"),           # 新宋體
        ("SimHei", "c:/windows/fonts/simhei.ttf")            # 黑體
    ]
    
    for font_name, font_path in font_options:
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            chinese_font_available = True
            chinese_font_name = font_name
            st.success(f"成功載入中文字體: {font_name}")
            break
        except Exception:
            continue
    
    if not chinese_font_available:
        st.warning("無法載入任何中文字體，PDF 中的中文可能會顯示不正確。請確保系統安裝了中文字體。")
    
    pdf_path = os.path.join(TEMP_DATA_STORAGE, output_filename)
    doc = SimpleDocTemplate(pdf_path)  # 建立 PDF 文件模板
    styles = getSampleStyleSheet()     # 取得預設樣式
    
    # 建立中文樣式
    if chinese_font_available:
        # 複製並修改現有樣式以支援中文
        styles['Title'] = ParagraphStyle('Title', parent=styles['Title'], fontName=chinese_font_name, leading=14)
        styles['h1'] = ParagraphStyle('h1', parent=styles['Heading1'], fontName=chinese_font_name, leading=14)
        styles['h2'] = ParagraphStyle('h2', parent=styles['Heading2'], fontName=chinese_font_name, leading=14)
        styles['Normal'] = ParagraphStyle('Normal', parent=styles['Normal'], fontName=chinese_font_name, leading=12)
        styles['Bullet'] = ParagraphStyle('Bullet', parent=styles['Bullet'], fontName=chinese_font_name, leading=12)
        styles['Italic'] = ParagraphStyle('Italic', parent=styles['Italic'], fontName=chinese_font_name, leading=12)
    
    story = []                         # 用來儲存所有 PDF 頁面內容的 list

    # --- 標題 ---
    story.append(Paragraph("📘 全面分析報告", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # --- 1. 分析目標（使用者查詢）---
    story.append(Paragraph("1. 分析目標（用戶查詢）", styles['h2']))
    analysis_goal = artifacts.get("original_user_query", "未指定。")
    story.append(Paragraph(html.escape(analysis_goal), styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- 2. CDO 初始報告 ---
    story.append(Paragraph("2. CDO 初始數據描述與品質評估", styles['h2']))
    cdo_report_text = st.session_state.get("cdo_initial_report_text", "⚠️ 無法取得 CDO 初始報告。")
    cdo_report_text_cleaned = html.escape(cdo_report_text.replace("**", ""))

    # 每段落分段加入
    for para_text in cdo_report_text_cleaned.split('\n'):
        if para_text.strip().startswith("- "):
            story.append(Paragraph(para_text, styles['Bullet'], bulletText='-'))
        elif para_text.strip():
            story.append(Paragraph(para_text, styles['Normal']))
        else:
            story.append(Spacer(1, 0.1 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())  # 換頁

    # --- 3. 圖表圖片 ---
    story.append(Paragraph("3. 生成的圖表", styles['h2']))
    plot_image_path = artifacts.get("plot_image_path")
    if plot_image_path and os.path.exists(plot_image_path):
        try:
            img = Image(plot_image_path, width=6 * inch, height=4 * inch)
            img.hAlign = 'CENTER'
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"❌ 圖表嵌入錯誤：{html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("⚠️ 找不到圖表或圖表路徑無效。", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- 4. 圖表對應數據 / 執行後資料表 ---
    story.append(Paragraph("4. 圖表對應數據（或執行的數據表）", styles['h2']))
    plot_data_csv_path = artifacts.get("executed_data_path")
    executed_df = artifacts.get("executed_dataframe_result")

    data_to_display_in_pdf = None
    if executed_df is not None and isinstance(executed_df, pd.DataFrame):
        data_to_display_in_pdf = executed_df
    elif plot_data_csv_path and os.path.exists(plot_data_csv_path) and plot_data_csv_path.endswith(".csv"):
        try:
            data_to_display_in_pdf = pd.read_csv(plot_data_csv_path)
        except Exception as e:
            story.append(Paragraph(f"❌ 無法讀取 CSV：{html.escape(str(e))}", styles['Normal']))
            data_to_display_in_pdf = None

    # 顯示表格
    if data_to_display_in_pdf is not None and not data_to_display_in_pdf.empty:
        data_for_table = [data_to_display_in_pdf.columns.astype(str).tolist()] + \
                         data_to_display_in_pdf.astype(str).values.tolist()

        if len(data_for_table) > 1:
            max_rows_in_pdf = 30
            if len(data_for_table) > max_rows_in_pdf:
                data_for_table = data_for_table[:max_rows_in_pdf]
                story.append(Paragraph(f"(📌 僅顯示前 {max_rows_in_pdf - 1} 行資料)", styles['Italic']))

            table = Table(data_for_table, repeatRows=1)
            table_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('WORDWRAP', (0, 0), (-1, -1), 'CJK')  # 中文換行支援
            ]
            
            # 如果有中文字體可用，則使用中文字體
            if chinese_font_available:
                table_style.append(('FONTNAME', (0, 0), (-1, -1), chinese_font_name))
                table_style.append(('FONTNAME', (0, 0), (-1, 0), chinese_font_name))
            else:
                table_style.append(('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'))
                
            table.setStyle(TableStyle(table_style))
            story.append(table)
        else:
            story.append(Paragraph("⚠️ 表格僅含標頭，無可顯示資料。", styles['Normal']))
    else:
        story.append(Paragraph("⚠️ 無法取得或顯示表格資料。", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())

    # --- 5. 生成的文字報告 ---
    story.append(Paragraph("5. 生成的文本報告（特定分析）", styles['h2']))
    report_text_path = artifacts.get("generated_report_path")
    if report_text_path and os.path.exists(report_text_path):
        try:
            with open(report_text_path, 'r', encoding='utf-8') as f:
                report_text_content = f.read()
            report_text_content_cleaned = html.escape(report_text_content.replace("**", ""))
            for para_text in report_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"❌ 無法讀取報告內容：{html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("⚠️ 找不到報告檔案或路徑錯誤。", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- 6. 分析評論 ---
    story.append(Paragraph("6. 分析評論", styles['h2']))
    critique_text_path = artifacts.get("generated_critique_path")
    if critique_text_path and os.path.exists(critique_text_path):
        try:
            with open(critique_text_path, 'r', encoding='utf-8') as f:
                critique_text_content = f.read()
            critique_text_content_cleaned = html.escape(critique_text_content.replace("**", ""))
            for para_text in critique_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"❌ 無法讀取評論檔案：{html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("⚠️ 找不到評論檔案或路徑錯誤。", styles['Normal']))

    # --- 建立 PDF 文件 ---
    try:
        # 使用 try-except 捕捉字體相關錯誤
        doc.build(story)
        st.success(f"✅ PDF 報告已成功生成: {pdf_path}")
        return pdf_path
    except Exception as e:
        st.error(f"❌ PDF 生成失敗：{e}")
        # 如果是字體相關錯誤，嘗試使用基本字體重新生成
        if "font" in str(e).lower() and chinese_font_available:
            st.warning("嘗試使用基本字體重新生成 PDF...")
            try:
                # 重置樣式為基本字體
                styles = getSampleStyleSheet()
                # 重建文檔
                doc = SimpleDocTemplate(pdf_path)
                doc.build(story)
                st.success(f"✅ PDF 報告已使用基本字體成功生成: {pdf_path}")
                return pdf_path
            except Exception as e2:
                st.error(f"❌ 使用基本字體重新生成 PDF 也失敗：{e2}")
        return None
        return None



# --- HTML Bento 報告生成（基於 Python）---
def _generate_html_paragraphs(text_content):
    """將純文字內容轉換為 HTML 段落，並轉義特殊字元避免 HTML 注入"""
    if not text_content or text_content.strip() == "無可用內容":
        return "<p><em>無可用內容</em></p>"
    
    escaped_content = html.escape(text_content)
    paragraphs = "".join([f"<p>{line}</p>" for line in escaped_content.split('\n') if line.strip()])
    return paragraphs if paragraphs else "<p><em>未提供內容。</em></p>"


def _generate_html_table(csv_text_or_df):
    """從 CSV 字串或 pandas DataFrame 生成可嵌入網頁的 HTML 表格"""
    df = None

    # 若為 DataFrame，直接使用
    if isinstance(csv_text_or_df, pd.DataFrame):
        df = csv_text_or_df

    # 若為 CSV 字串，先去除前綴提示語後解析為 DataFrame
    elif isinstance(csv_text_or_df, str):
        prefixes = [
            "Primary Table Data (CSV format for HTML table):\n",
            "Data for Chart (CSV format for Chart.js):\n",
            "DQ Column Assessment (CSV for table):\n",
            "DQ Correlation Matrix (CSV for table):\n"
        ]
        for p in prefixes:
            if csv_text_or_df.startswith(p):
                csv_text_or_df = csv_text_or_df.split(p, 1)[1]
                break

        # 無效資料處理
        if not csv_text_or_df.strip() or "Not available" in csv_text_or_df or "No specific data table" in csv_text_or_df:
            return "<p><em>⚠️ 資料表無法取得或不適用。</em></p>"
        
        try:
            csv_file = io.StringIO(csv_text_or_df)
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            return "<p><em>⚠️ 表格資料為空或不是有效的 CSV 格式。</em></p>"
        except Exception as e:
            return f"<p><em>❌ CSV 解析錯誤：{html.escape(str(e))}。</em></p>"
    else:
        return "<p><em>❌ 資料型別錯誤，無法生成表格。</em></p>"

    if df is None or df.empty:
        return "<p><em>⚠️ 表格資料為空。</em></p>"

    # 轉換為 HTML 表格
    table_html = "<table>\n<thead>\n<tr>"
    for header in df.columns:
        table_html += f"<th>{html.escape(str(header))}</th>"
    table_html += "</tr>\n</thead>\n<tbody>\n"

    for _, row in df.iterrows():
        table_html += "<tr>"
        for cell in row:
            cell_str = str(cell)
            is_numeric = pd.api.types.is_number(cell) and not pd.isna(cell)
            class_attr = ' class="numeric"' if is_numeric else ''
            table_html += f"<td{class_attr}>{html.escape(cell_str)}</td>"
        table_html += "</tr>\n"

    table_html += "</tbody>\n</table>"
    return table_html



def _generate_html_image_embed(image_path_from_artifact):
    """嵌入圖片的 HTML，並提示使用者圖片需與 HTML 同目錄"""
    if not image_path_from_artifact or not os.path.exists(image_path_from_artifact):
        return "<p><em>⚠️ 找不到圖片或路徑無效。</em></p>"

    image_filename = os.path.basename(image_path_from_artifact)
    img_tag = f'<img src="{html.escape(image_filename)}" alt="視覺化" style="max-width: 100%; max-height: 100%; height: auto; display: block; margin: auto; border-radius: 8px;">'
    note = f'<p class="image-note"><strong>圖片註解「{html.escape(image_filename)}」：</strong>請將此圖片放在與 HTML 檔案相同的資料夾中。</p>'
    return f'<div class="visualization-container">{img_tag}</div>{note}'


def _generate_chartjs_embed(csv_text_or_df, chart_id):
    """將 CSV 或 DataFrame 轉換為 Chart.js 圖表的 HTML + JS 程式碼"""
    df = None

    # CSV 或 DataFrame 處理
    if isinstance(csv_text_or_df, pd.DataFrame):
        df = csv_text_or_df
    elif isinstance(csv_text_or_df, str):
        prefixes = [
            "Data for Chart (CSV format for Chart.js):\n",
            "DQ Numeric Dist. Example (CSV for Chart.js):\n",
            "DQ Categorical Dist. Example (CSV for Chart.js):\n"
        ]
        for p in prefixes:
            if csv_text_or_df.startswith(p):
                csv_text_or_df = csv_text_or_df.split(p, 1)[1]
                break

        if not csv_text_or_df.strip() or "Not available" in csv_text_or_df:
            return f"<div class='visualization-container'><p><em>⚠️ Chart.js 資料不可用或不適用。</em></p></div>"
        try:
            csv_file = io.StringIO(csv_text_or_df)
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            return f"<div class='visualization-container'><p><em>⚠️ Chart.js 資料為空或格式錯誤。</em></p></div>"
        except Exception as e:
            return f"<div class='visualization-container'><p><em>❌ CSV 解析錯誤：{html.escape(str(e))}</em></p></div>"
    else:
        return f"<div class='visualization-container'><p><em>❌ 無效資料類型，無法生成 Chart.js 圖表。</em></p></div>"

    if df is None or df.empty or len(df.columns) < 2:
        return f"<div class='visualization-container'><p><em>⚠️ 至少需兩欄且非空的資料才能生成圖表。</em></p></div>"

    # 抽取 labels 與數值資料
    labels = df.iloc[:, 0].astype(str).tolist()
    data_values = df.iloc[:, 1].tolist()

    numeric_data_values = []
    for val in data_values:
        try:
            numeric_data_values.append(float(val))
        except (ValueError, TypeError):
            return f"<div class='visualization-container'><p><em>❌ 第二欄應為數值，發現非數值：'{html.escape(str(val))}'</em></p></div>"

    chart_type = 'bar'
    chart_label = html.escape(str(df.columns[1]))
    if all("-" in str(l) for l in labels[:3]) and len(labels) > 1:
        chart_label = "頻率分佈"

    # HTML 與 JavaScript 結合 Chart.js
    canvas_html = f'<div class="visualization-container"><canvas id="{html.escape(chart_id)}"></canvas></div>'
    script_html = f"""
<script>
    const ctx_{html.escape(chart_id)} = document.getElementById('{html.escape(chart_id)}').getContext('2d');
    new Chart(ctx_{html.escape(chart_id)}, {{
        type: '{chart_type}',
        data: {{
            labels: {json.dumps(labels)},
            datasets: [{{
                label: '{chart_label}',
                data: {json.dumps(numeric_data_values)},
                backgroundColor: 'rgba(199, 120, 221, 0.6)',
                borderColor: 'rgba(74, 35, 90, 1)',
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                y: {{
                    beginAtZero: true,
                    ticks: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }},
                    grid: {{ color: '#3A3D4E' }}
                }},
                x: {{
                    ticks: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }},
                    grid: {{ color: '#3A3D4E' }}
                }}
            }},
            plugins: {{
                legend: {{
                    labels: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }}
                }},
                tooltip: {{
                    backgroundColor: '#2A2D3E',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: '#7DF9FF',
                    borderWidth: 1,
                    titleFont: {{ family: "Inter" }},
                    bodyFont: {{ family: "Inter" }}
                }}
            }}
        }}
    }});
</script>
"""
    return canvas_html + script_html


def generate_bento_html_report_python(artifacts, cdo_initial_report, data_summary_dict, main_df):
    """
    使用 Python 字串格式化的方式，產生 Bento 格式的 HTML 分析報告。

    報告內容包含：
    - 數據品質儀表板區塊（Data Quality Dashboard）
    - CDO 初步報告文字
    - 使用者提供的欄位說明（若有）
    - 圖表與資料表的嵌入區塊（根據 artifacts 自動載入）
    
    參數：
    artifacts (dict): 存放圖表路徑、報告文字等分析成品的字典。
    cdo_initial_report (str): CDO 提供的初始數據分析文字。
    data_summary_dict (dict): 整理後的資料摘要，包含行列數、缺失值、欄位描述等。
    main_df (DataFrame): 主要的原始資料表，用於資料品質檢查與顯示。

    回傳：
    str: 包含所有組件的 HTML 字串。
    """

    # 傳入主資料表 main_df，讓 compile_report_text_for_html_generation 可根據其內容生成 DQ 分析區段
    report_parts_dict = compile_report_text_for_html_generation(
        artifacts, 
        cdo_initial_report, 
        data_summary_dict, 
        main_df
    )


    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 生成的 Bento 報告 - {html.escape(artifacts.get("original_user_query", "分析")[:50])}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Inter', '微軟正黑體', sans-serif; background-color: #1A1B26; color: #E0E0E0; margin: 0; padding: 20px; line-height: 1.6; }}
        .bento-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; max-width: 1600px; margin: 20px auto; padding:0; }}
        .bento-item {{ 
            background-color: #2A2D3E; 
            border: 1px solid #3A3D4E; 
            border-radius: 16px; 
            padding: 25px; 
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            overflow-wrap: break-word; 
            word-wrap: break-word; 
            overflow: hidden; /* 防止內容溢出 */
            display: flex; /* 為了更好的內部對齊 */
            flex-direction: column; /* 堆疊標題和內容 */
        }}
        .bento-item:hover {{ 
            transform: translateY(-5px); 
            border-color: #7DF9FF; 
            box-shadow: 0 10px 20px rgba(0,0,0,0.25);
        }}
        .bento-item h2 {{ 
            color: #FFFFFF; 
            font-size: 1.5rem; /* 略微縮小以保持平衡 */
            font-weight: 600;
            border-bottom: 2px solid #7DF9FF; 
            padding-bottom: 10px; 
            margin-top:0; 
            margin-bottom: 15px; /* 保持一致的間距 */
        }}
        .bento-item .content-wrapper {{ flex-grow: 1; overflow-y: auto; }} /* 允許內容在過長時滾動 */
        .bento-item p {{ color: #C0C0C0; margin-bottom: 10px; }}
        .bento-item p:last-child {{ margin-bottom: 0; }}
        .bento-item strong {{ color: #7DF9FF; font-weight: 600; }}
        .bento-item ul {{ list-style-position: inside; padding-left: 5px; color: #C0C0C0; margin-bottom:10px; }}
        .bento-item li {{ margin-bottom: 6px; }}

        /* 不同屏幕尺寸的跨度規則 */
        .bento-item.large {{ grid-column: span 1; }} /* 小屏幕的預設值 */
        @media (min-width: 768px) {{ /* 平板電腦及以上 */
            .bento-item.large {{ grid-column: span 2; }}
        }}
        /* 對於非常寬的屏幕，可以考慮讓某些項目跨越 3 格（如果需要） */
        @media (min-width: 1200px) {{ 
             .bento-grid {{ grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }}
        }}

        .bento-item.accent-box {{ background-color: #4A235A; border-color: #C778DD; }}
        .bento-item.accent-box h2 {{ border-bottom-color: #C778DD; }}
        .bento-item.accent-box strong {{ color: #FFA6FC; }}


        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; table-layout: auto; /* 改為自動以獲得更好的適配 */ }}
        th, td {{ border: 1px solid #4A4D5E; padding: 10px; text-align: left; word-wrap: break-word; font-size: 0.9rem; }}
        th {{ background-color: #3A3D4E; color: #FFFFFF; font-weight: 600; }}
        td.numeric {{ text-align: right; color: #7DF9FF; }}

        .visualization-container {{ 
            min-height: 300px; /* 調整最小高度 */
            max-height: 400px; /* 調整最大高度 */
            min-height: 300px; /* Adjusted min-height */
            max-height: 400px; /* Adjusted max-height */
            width: 100%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            overflow: hidden; 
            background-color: #202230; 
            border-radius: 8px; 
            padding:10px; 
            margin-bottom: 10px; /* Space before image note */
        }}
        .visualization-container img, .visualization-container canvas {{ 
            max-width: 100%; 
            max-height: 100%; 
            object-fit: contain; /* Ensures entire image/chart is visible */
            display: block; 
            margin: auto; 
            border-radius: 8px;
        }}
        .image-note {{ font-size: 0.8rem; color: #A0A0A0; text-align: center; margin-top: 5px; }}
        .placeholder-text {{ color: #888; font-style: italic; }}
        .user-descriptions {{ border-left: 3px solid #7DF9FF; padding-left: 15px; margin-top:10px; background-color: rgba(125, 249, 255, 0.05); border-radius: 4px; }}
        .user-descriptions p {{ color: #D0D0D0; }}
    </style>
</head>
<body>
    <div class="bento-grid">
"""

   # 指定報告各部分的呈現順序（移除不再使用的圖表部分）
    ordered_keys = [
        "ANALYSIS_GOAL",                         # 分析目標
        "DATA_SNAPSHOT_CDO_REPORT",              # CDO 數據摘要報告
        "USER_PROVIDED_DESCRIPTIONS",            # 用戶欄位說明
        "KEY_DATA_QUALITY_ALERT",                # 重要數據品質警告
        "DATA_PREPROCESSING_NOTE",               # 數據前處理說明
        "DQ_COLUMN_ASSESSMENT_TABLE",            # 欄位數據品質評估
        "DQ_CORRELATION_MATRIX_TABLE",           # 數據相關性矩陣
        "VISUALIZATION_CHART_OR_IMAGE",          # 主要視覺圖表
        "PRIMARY_ANALYSIS_TABLE",                # 主要分析結果表格
        "ACTIONABLE_INSIGHTS_FROM_REPORT",       # 可行建議與洞察
        "CRITIQUE_SUMMARY",                      # 批判摘要
        "SUGGESTIONS_FOR_ENHANCED_ANALYSIS"      # 深化分析的建議
    ]

    # 對每一區塊依序進行 HTML 組裝
    for key_index, key in enumerate(ordered_keys):
        if key not in report_parts_dict:
            continue  # 若某區塊缺少，略過

        # 每區塊的標題對照表（英文轉中文可參考下方）
        title_map = {
            "ANALYSIS_GOAL": "分析目標",
            "DATA_SNAPSHOT_CDO_REPORT": "CDO 數據摘要報告",
            "USER_PROVIDED_DESCRIPTIONS": "用戶提供的欄位說明",
            "KEY_DATA_QUALITY_ALERT": "關鍵數據品質警示",
            "DATA_PREPROCESSING_NOTE": "資料前處理說明",
            "DQ_COLUMN_ASSESSMENT_TABLE": "數據品質：欄位評估表",
            "DQ_CORRELATION_MATRIX_TABLE": "數據品質：相關性矩陣",
            "VISUALIZATION_CHART_OR_IMAGE": "主要視覺化圖表",
            "PRIMARY_ANALYSIS_TABLE": "主要分析表格",
            "ACTIONABLE_INSIGHTS_FROM_REPORT": "可行洞察建議",
            "CRITIQUE_SUMMARY": "批判總結",
            "SUGGESTIONS_FOR_ENHANCED_ANALYSIS": "強化分析建議"
        }

        title = title_map.get(key, key.replace('_', ' ').title())  # 若無對應則轉為標題格式
        raw_content_obj = report_parts_dict.get(key)

        item_classes = ["bento-item"]
        item_content_html = ""

        # 判斷哪些區塊需要使用大區塊版面（.large 樣式）
        large_keys = [
            "KEY_DATA_QUALITY_ALERT", "VISUALIZATION_CHART_OR_IMAGE", "PRIMARY_ANALYSIS_TABLE",
            "DQ_COLUMN_ASSESSMENT_TABLE", "DQ_CORRELATION_MATRIX_TABLE", "ACTIONABLE_INSIGHTS_FROM_REPORT",
            "DATA_SNAPSHOT_CDO_REPORT", "USER_PROVIDED_DESCRIPTIONS"
        ]
        if key in large_keys:
            item_classes.append("large")

        # 特殊處理：「數據品質警告」以條列顯示
        if key == "KEY_DATA_QUALITY_ALERT":
            item_classes.append("accent-box")  # 顯眼樣式
            if isinstance(raw_content_obj, str) and raw_content_obj.strip():
                alert_lines = [f"<li>{html.escape(line.strip())}</li>" for line in raw_content_obj.split('\n') if line.strip().startswith("- ")]
                if alert_lines:
                    item_content_html = f"<ul>{''.join(alert_lines)}</ul>"
                else:
                    item_content_html = _generate_html_paragraphs(raw_content_obj)
            else:
                item_content_html = "<p><em>尚未標註具體的數據品質警告，可能已整合在 CDO 報告中。</em></p>"

        # 用戶提供的欄位說明
        elif key == "USER_PROVIDED_DESCRIPTIONS":
            if isinstance(raw_content_obj, str) and raw_content_obj.strip() and raw_content_obj != "User descriptions not provided.":
                item_content_html = f"<div class='user-descriptions'>{_generate_html_paragraphs(raw_content_obj)}</div>"
            else:
                item_content_html = "<p><em>使用者未提供額外欄位說明。</em></p>"

        # 圖片或 Chart.js 圖表
        elif key == "VISUALIZATION_CHART_OR_IMAGE":
            plot_image_path = artifacts.get("plot_image_path")
            executed_data_for_chart = raw_content_obj
            if plot_image_path and os.path.exists(plot_image_path):
                item_content_html = _generate_html_image_embed(plot_image_path)
            elif isinstance(executed_data_for_chart, pd.DataFrame) and not executed_data_for_chart.empty:
                item_content_html = _generate_chartjs_embed(executed_data_for_chart, f"bentoChartAiAnalysis{key_index}")
            else:
                item_content_html = "<p><em>未提供圖表或資料無法用於可視化。</em></p>"

        # 三種表格處理（欄位品質、相關矩陣、主要分析）
        elif key in ["PRIMARY_ANALYSIS_TABLE", "DQ_COLUMN_ASSESSMENT_TABLE", "DQ_CORRELATION_MATRIX_TABLE"]:
            if isinstance(raw_content_obj, pd.DataFrame):
                item_content_html = _generate_html_table(raw_content_obj)
            else:
                item_content_html = _generate_html_paragraphs(str(raw_content_obj))

        # 其他純文字內容
        elif isinstance(raw_content_obj, str):
            item_content_html = _generate_html_paragraphs(raw_content_obj)
        else:
            item_content_html = "<p><em>此區塊的內容格式無法辨識。</em></p>"

        # HTML 組合區塊
        html_content += f'<div class="{" ".join(item_classes)}">\n'
        html_content += f'<h2>{html.escape(title)}</h2>\n'
        html_content += f'<div class="content-wrapper">{item_content_html}</div>\n'
        html_content += '</div>\n'

    # 收尾 HTML
    html_content += """
    </div>
</body>
</html>
"""

    # 設定檔名：包含查詢關鍵字與時間戳記
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query_part = "".join(c if c.isalnum() else "_" for c in artifacts.get("original_user_query", "report")[:30])
    html_filename = f"bento_report_{safe_query_part}_{timestamp}.html"
    html_filepath = os.path.join(TEMP_DATA_STORAGE, html_filename)

    # 寫入 HTML 檔案
    try:
        with open(html_filepath, "w", encoding='utf-8') as f:
            f.write(html_content)
        return html_filepath
    except Exception as e:
        st.error(f"寫入 HTML 報告時發生錯誤：{e}")
        return None



def get_content_from_path_helper(file_path, default_message="Not available."):
    """
    安全地讀取指定路徑的檔案內容。
    回傳檔案文字內容；若檔案不存在或讀取失敗，則回傳預設訊息。
    """
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"讀取檔案錯誤：{str(e)}"
    return default_message  # 預設訊息



def compile_report_text_for_html_generation(artifacts, cdo_initial_report, data_summary_dict, main_df=None):
    """
    將所有報告相關文字與資料整理為一個字典，供 HTML 生成使用。
    包含 AI 分析表格、視覺圖表、使用者描述與數據品質儀表板等內容。
    """
    # 取得 AI 批判摘要與分析報告
    critique_text = get_content_from_path_helper(
        artifacts.get("generated_critique_path"),
        "無法取得批判內容。"
    )
    generated_report_text = get_content_from_path_helper(
        artifacts.get("generated_report_path"),
        "無法取得生成的文字報告。"
    )

    # 🔍 主要分析表格
    primary_analysis_table_obj = "AI 分析尚未產生主要表格資料。"
    if isinstance(artifacts.get("executed_dataframe_result"), pd.DataFrame):
        primary_analysis_table_obj = artifacts["executed_dataframe_result"]
    elif artifacts.get("executed_data_path") and "table_result" in os.path.basename(artifacts["executed_data_path"]):
        try:
            primary_analysis_table_obj = pd.read_csv(artifacts["executed_data_path"])
        except:
            primary_analysis_table_obj = get_content_from_path_helper(artifacts["executed_data_path"])

    # 📊 圖表資料（Chart.js 或圖片）
    visualization_data_obj = "AI 未提供圖表資料。"
    if isinstance(artifacts.get("plot_specific_data_df"), pd.DataFrame):
        visualization_data_obj = artifacts["plot_specific_data_df"]
    elif artifacts.get("executed_data_path") and "plot_data_for" in os.path.basename(artifacts["executed_data_path"]):
        try:
            visualization_data_obj = pd.read_csv(artifacts["executed_data_path"])
        except:
            visualization_data_obj = get_content_from_path_helper(artifacts["executed_data_path"])

    # 📘 組合報告各部分
    report_parts = {
        "ANALYSIS_GOAL": artifacts.get("original_user_query", "未指定分析目標"),
        "DATA_SNAPSHOT_CDO_REPORT": cdo_initial_report if cdo_initial_report else "無法取得 CDO 數據摘要。",
        "USER_PROVIDED_DESCRIPTIONS": data_summary_dict.get("user_provided_column_descriptions", "使用者未提供欄位說明"),
        "KEY_DATA_QUALITY_ALERT": "",  # 下面會自動產出
        "DATA_PREPROCESSING_NOTE": "資料使用原始格式進行分析，若有前處理，已包含於查詢中或 CDO 報告內。已執行標準格式判斷與載入。",
        "VISUALIZATION_CHART_OR_IMAGE": visualization_data_obj,
        "PRIMARY_ANALYSIS_TABLE": primary_analysis_table_obj,
        "ACTIONABLE_INSIGHTS_FROM_REPORT": generated_report_text,
        "CRITIQUE_SUMMARY": critique_text,
        "SUGGESTIONS_FOR_ENHANCED_ANALYSIS": critique_text  # 目前使用相同內容，可改為呼叫另一模型
    }

    # 🚨 產出缺漏值警示文字
    missing_data_alerts_text = []
    if data_summary_dict and data_summary_dict.get("missing_values_per_column"):
        for col, count in data_summary_dict["missing_values_per_column"].items():
            if count > 0:
                num_rows = data_summary_dict.get("num_rows", 1)
                percentage = (count / num_rows) * 100 if num_rows > 0 else 0
                missing_data_alerts_text.append(
                    f"- 欄位 '{html.escape(col)}'：共 {count} 筆缺漏值（{percentage:.2f}%）"
                )
    if not missing_data_alerts_text:
        missing_data_alerts_text.append(
            "- 自動摘要中未發現明顯缺漏值，或數據品質問題已整合在 CDO 報告中。"
        )
    report_parts["KEY_DATA_QUALITY_ALERT"] = "\n".join(missing_data_alerts_text)

    # 📊 加入資料品質報表（欄位評估、直方圖、分類分佈、相關矩陣）
    if main_df is not None and not main_df.empty:
        quality_assessment_df = get_column_quality_assessment(main_df.copy())
        report_parts["DQ_COLUMN_ASSESSMENT_TABLE"] = (
            quality_assessment_df if not quality_assessment_df.empty else "欄位評估無資料"
        )

        # 數值欄位的分佈圖表
        numeric_cols_dq = main_df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_dq:
            first_numeric_col = numeric_cols_dq[0]
            col_data_numeric = main_df[first_numeric_col].dropna()
            if not col_data_numeric.empty:
                counts, bins = np.histogram(col_data_numeric, bins=10)
                hist_df = pd.DataFrame({
                    'Bin': [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)],
                    'Frequency': counts
                })
                report_parts["DQ_NUMERIC_DIST_CHART"] = hist_df
            else:
                report_parts["DQ_NUMERIC_DIST_CHART"] = f"欄位 '{html.escape(first_numeric_col)}' 無足夠資料產生分佈圖。"
        else:
            report_parts["DQ_NUMERIC_DIST_CHART"] = "資料中未包含數值欄位。"

        # 類別欄位的分佈長條圖
        categorical_cols_dq = main_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols_dq:
            first_cat_col = categorical_cols_dq[0]
            col_data_cat = main_df[first_cat_col].dropna()
            if not col_data_cat.empty:
                cat_counts = col_data_cat.value_counts().head(10)
                cat_df = cat_counts.reset_index()
                cat_df.columns = [first_cat_col, 'Count']
                report_parts["DQ_CATEGORICAL_DIST_CHART"] = cat_df
            else:
                report_parts["DQ_CATEGORICAL_DIST_CHART"] = f"欄位 '{html.escape(first_cat_col)}' 無資料可製圖。"
        else:
            report_parts["DQ_CATEGORICAL_DIST_CHART"] = "資料中未包含類別欄位。"

        # 數值欄位相關係數矩陣
        if len(numeric_cols_dq) >= 2:
            corr_matrix = main_df[numeric_cols_dq].corr().round(2)
            report_parts["DQ_CORRELATION_MATRIX_TABLE"] = corr_matrix
        else:
            report_parts["DQ_CORRELATION_MATRIX_TABLE"] = "數值欄位不足，無法產生相關係數矩陣。"
    else:
        # 若主資料未提供
        default_msg = "主資料缺失，無法生成此區塊。"
        report_parts["DQ_COLUMN_ASSESSMENT_TABLE"] = default_msg
        report_parts["DQ_NUMERIC_DIST_CHART"] = default_msg
        report_parts["DQ_CATEGORICAL_DIST_CHART"] = default_msg
        report_parts["DQ_CORRELATION_MATRIX_TABLE"] = default_msg

    return report_parts



# --- Streamlit App UI ---
st.title("🤖 銷貨損益分析小幫手")  
# 設定應用程式標題，顯示在網頁最上方

st.caption(
    "上傳 CSV 和可選列描述 (.txt)，審查資料質量，進行探索，然後選擇性地執行 CDO 工作流程進行 AI 分析。"
)
# 顯示副標說明，引導使用者如何使用此分析小幫手



# 初始化 Session State 狀態變數
# 基本應用狀態管理
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "您好！選擇模型，上傳 CSV 檔案（也可以上傳包含列描述的 .txt 檔案）即可開始。"}]
# 初始對話訊息，提供歡迎提示

if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
# 使用者上傳的 CSV 檔案所轉換的 DataFrame 將儲存在此變數

if "data_summary" not in st.session_state:
    st.session_state.data_summary = None
# 儲存資料的綜合摘要（欄位統計、缺值資訊等）

if "data_source_name" not in st.session_state:
    st.session_state.data_source_name = None
# 儲存上傳的 CSV 檔案名稱

if "desc_file_name" not in st.session_state:
    st.session_state.desc_file_name = None
# 儲存使用者上傳的列描述（TXT）檔案名稱

if "current_analysis_artifacts" not in st.session_state:
    st.session_state.current_analysis_artifacts = {}
# 用來儲存整體分析結果（包含圖表路徑、報告內容等）的字典



# 模型選擇狀態
# 工作模型（執行分析的 AI 模型）
if "selected_worker_model" not in st.session_state:
    st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL

# 評估模型（用來評估分析品質）
if "selected_judge_model" not in st.session_state:
    st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL



# LangChain 記憶模組（可選）
# LangChain 記憶體，紀錄聊天歷程，讓模型能記住使用者提問脈絡
if "lc_memory" not in st.session_state:
    st.session_state.lc_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        input_key="user_query"
    )

# CDO 工作流程狀態管理（分析的不同階段）
if "cdo_initial_report_text" not in st.session_state:
    st.session_state.cdo_initial_report_text = None
# 儲存由 CDO 產生的初始報告文字

if "other_perspectives_text" not in st.session_state:
    st.session_state.other_perspectives_text = None
# 儲存其他 AI 分析視角

if "strategy_text" not in st.session_state:
    st.session_state.strategy_text = None
# 儲存 AI 給出的商業策略建議

if "cdo_workflow_stage" not in st.session_state:
    st.session_state.cdo_workflow_stage = None
# 追蹤目前分析流程進行到哪一個階段



# 分析觸發旗標（供後端程式判斷是否執行）
if "trigger_code_generation" not in st.session_state:
    st.session_state.trigger_code_generation = False
# 是否啟動程式碼生成（可能是用 LLM 根據分析產生程式碼）

if "trigger_report_generation" not in st.session_state:
    st.session_state.trigger_report_generation = False
# 是否觸發產生分析報告（PDF）

if "trigger_judging" not in st.session_state:
    st.session_state.trigger_judging = False
# 是否觸發 AI 對結果進行判斷（自評或多模型審核）

if "trigger_html_export" not in st.session_state:
    st.session_state.trigger_html_export = False
# 是否觸發產出 HTML 版本報告



# 產出報告時使用的目標資料與內容設定
if "report_target_data_path" not in st.session_state:
    st.session_state.report_target_data_path = None
# 用於報告產出的 CSV 資料檔案路徑

if "report_target_plot_path" not in st.session_state:
    st.session_state.report_target_plot_path = None
# 圖表圖像檔案的路徑

if "report_target_query" not in st.session_state:
    st.session_state.report_target_query = None
# 原始的用戶查詢（自然語言問題）作為分析起點



# --- 提示模板 ---
# 注意：提示現在會自動包含 user_provided_column_descriptions 作為 {data_summary} 的一部分
cdo_initial_data_description_prompt_template = PromptTemplate(input_variables=["data_summary", "chat_history"],
                                                              template="""你是首席資料長（CDO）。使用者已上傳一個 CSV 檔案，並可能附上欄位說明的文字檔。
資料摘要（提供背景資訊，可能包含 'user_provided_column_descriptions'）：
{data_summary}

CDO，你的第一項任務是對資料集進行初步說明，內容包括：
1. 類似 `df.info()` 的概覽（欄位名稱、非空值數量、資料型別）。
2. 若資料摘要中包含 'user_provided_column_descriptions'，請引用並整合至說明中，說明這些描述如何釐清資料內容。
3. 對每個變數／欄位提供你的推測含義或常見解釋，特別是未由使用者說明者。
4. 初步資料品質評估（例如明顯的缺失資料模式、潛在離群值、資料型別一致性等），並考量使用者提供的描述。
這份說明將提供給其他部門主管參考。
對話歷史（供參考）：
{chat_history}

CDO 撰寫的詳細初步說明（納入使用者說明，如有）：""")

individual_perspectives_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report"], template="""你是一組由各部門主管（包含 CDO）組成的專家小組。
使用者已上傳 CSV 檔案（以及可能的欄位說明），而 CDO 已提供初步的資料說明與品質評估。
資料摘要（原始內容，可能包含 'user_provided_column_descriptions'）：
{data_summary}

CDO 初步說明與品質報告（已整合使用者說明）：
--- CDO 報告開始 ---
{cdo_initial_report}
--- CDO 報告結束 ---

請根據原始資料摘要與 CDO 報告，分別提供以下角色的詳細觀點：
每位主管需提出 2 至 3 項具體問題、希望執行的分析或觀察內容，並參考 CDO 發現與使用者提供的欄位意義。

請使用下列結構呈現：
* **CEO（執行長）觀點：**
* **CFO（財務長）觀點：**
* **CTO（技術長）觀點：**
* **COO（營運長）觀點：**
* **CMO（行銷長）觀點：**
* **CDO（重申關鍵點，並引用使用者說明）：**

對話歷史（供參考）：
{chat_history}

各部門主管的詳細觀點（參考 CDO 報告與使用者欄位說明）：""")

synthesize_analysis_suggestions_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report", "generated_perspectives_from_others"],
    template="""你是本公司的首席資料長（CDO）。
使用者已上傳一份 CSV 檔案。你已完成初步的資料說明（若有使用者提供欄位說明亦已納入）。
接著，其它部門主管也根據你的初步發現與資料摘要提供了回饋。

原始資料摘要（可能包含 'user_provided_column_descriptions'）：
{data_summary}

你的初步資料說明與品質評估：
--- CDO 初步報告開始 ---
{cdo_initial_report}
--- CDO 初步報告結束 ---

各部門主管的觀點（CEO、CFO、CTO、COO、CMO）：
--- 主管觀點開始 ---
{generated_perspectives_from_others}
--- 主管觀點結束 ---

你的任務是綜合上述資訊，提出 **5 項清晰且可執行的分析策略建議**。
這些建議應優先聚焦在能產出明確圖表、格式良好的表格，或簡潔描述性摘要的分析方式。
若有使用者提供欄位說明，請充分加以應用。

請以編號條列方式呈現這 5 項建議。每項建議需清楚說明分析類型。

對話歷史（供參考）：
{chat_history}

CDO 統整後的 5 項分析策略建議（著重圖表、表格、描述方式，整合所有先前輸入與使用者說明）：""")

# TEMP_DATA_STORAGE_PROMPT will be replaced with the actual path
TEMP_DATA_STORAGE_PROMPT_PLACEHOLDER = "{TEMP_DATA_STORAGE_PATH_FOR_PROMPT}"

code_generation_prompt_template = PromptTemplate(input_variables=["data_summary", "user_query", "chat_history"],
                                                 template=f"""你是位專業的 Python 資料分析助手。
資料摘要（可能包含 'user_provided_column_descriptions'）：
{'{data_summary}'}
使用者提問："{{user_query}}"
先前對話紀錄（提供背景參考）：
{'{chat_history}'}

你的任務是根據名為 `df` 的 pandas DataFrame，撰寫一段 Python 程式碼來執行指定分析。
**關於 `analysis_result` 與 `plot_data_df` 的關鍵指引：**
1.  **必須設定 `analysis_result`**：主分析結果需指定為 `analysis_result`。
2.  **若為圖表類輸出：**
    a.  將圖表儲存於指定暫存資料夾，使用路徑：`os.path.join(TEMP_DATA_STORAGE, 'your_plot_filename.png')`。你可使用 `TEMP_DATA_STORAGE` 變數。
    b.  `analysis_result` 僅需設定為圖檔名稱字串（例如 'my_plot.png'，不要加上完整路徑）。
    c.  建立名為 `plot_data_df` 的 pandas DataFrame，只包含實際視覺化的資料。若為整個 `df`，則使用 `df.copy()`；若圖表來自彙總，則設定為 `None`。
3.  **若為表格類輸出（DataFrame / Series）：**
    a.  將產出結果指定為 `analysis_result`。
    b.  將 `plot_data_df = analysis_result.copy()`（若為 DataFrame/Series）或設為 `None`（若不適用）。
4.  **若為文字輸出結果：**
    a.  將文字結果設為 `analysis_result`（字串）。
    b.  設定 `plot_data_df = None`。
5.  **預設處理：** 若提問內容較廣泛或未明確要求特定輸出，`analysis_result` 可為 `df.head()` 或簡要描述；此時 `plot_data_df = df.head().copy()` 或 `None`。
6.  **必要匯入：** 請務必引入必要套件（如 `pandas`, `matplotlib.pyplot`, `seaborn`, `numpy`, `os`）。`TEMP_DATA_STORAGE` 路徑變數可直接使用。
7.  **資料夾建立：** 在儲存圖表前，請確認目錄存在，可用 `os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)` 確保。

**保險機制 - 請在腳本結尾加入以下邏輯以防遺漏設定：**
```python
# --- 保險機制 ---
if 'analysis_result' not in locals():
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "腳本執行完畢，AI 主邏輯未指定 'analysis_result'，預設顯示 df.head() 結果。"
        plot_data_df = df.head().copy()
    else:
        analysis_result = "腳本執行完畢，未設定 'analysis_result'，且未偵測到有效的 'df'。"
if 'plot_data_df' not in locals():
    plot_data_df = None
```
請僅輸出純 Python 程式碼，勿包含其他文字或 markdown 語法。
Python 程式碼：""")

report_generation_prompt_template = PromptTemplate(
    input_variables=["table_data_csv_or_text", "original_data_summary", "user_query_that_led_to_data", "chat_history",
                     "plot_info_if_any"],
    template="""你是一位具有洞察力的資料分析師。請根據提供的資料與背景內容撰寫文字報告。
原始資料摘要（可能包含 'user_provided_column_descriptions'）：
{original_data_summary}

使用者的提問（導致此資料或圖表產出）："{user_query_that_led_to_data}"
圖表資訊（若適用，否則請寫 'N/A'）：{plot_info_if_any}
對話歷史（提供背景）：
{chat_history}

分析結果資料（CSV 內容、文字輸出，或為圖表報告時寫 'N/A'）：
```
{table_data_csv_or_text}
```

**報告架構：**
* **1. 重點摘要（1-2 句）：** 濃縮敘述來自 `分析結果資料` 或圖表的主要結論。
* **2. 分析目的（1 句）：** 簡要重述使用者提問所反映的目標。
* **3. 主要觀察（條列 2-4 點）：** 具體說明 `分析結果資料` 或圖表中明確的數據趨勢或觀察，若有欄位說明幫助理解，請提及。
* **4. 可行洞察（1-2 點）：** 這些發現代表什麼意義？可據以採取哪些行動？
* **5. 資料侷限與焦點：** 清楚說明本報告 *僅根據* 上述的「分析結果資料」與／或「圖表資訊」撰寫。若資料為樣本、彙總等也請指出。

**語氣風格：** 專業、清晰、客觀。**請勿**使用「從圖表可以看出…」或「CSV 顯示了…」等語句，請直接陳述觀察結果。
報告：""")

judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_content_for_judge", "report_text_content", "original_user_query",
                     "data_summary",
                     "plot_image_path", "plot_info_for_judge"], template="""你是一位資深的資料科學審查員。請依據使用者的提問與資料上下文，評估 AI 助手產出的各項成果。
使用者原始問題："{original_user_query}"
原始資料摘要（可能包含 'user_provided_column_descriptions'）：
{data_summary}

--- 評估項目 ---
1.  **AI 助手產出的 Python 程式碼：**
    ```python
{python_code}
    ```
2.  **執行程式碼後產生的資料（CSV 內容或 `analysis_result` 的文字輸出）：**
    ```
{data_content_for_judge}
    ```
    **圖表資訊：** {plot_info_for_judge}（可能指出是否產出圖檔 '{plot_image_path}'，或是否產出 `plot_data_df`）

3.  **AI 助手撰寫的文字報告（若有）：**
    ```text
{report_text_content}
    ```
--- 結束評估項目 ---

**評估準則：**
1.  **程式碼品質與合規性（核心指標）：**
    * 正確性：程式碼是否能執行且邏輯正確？
    * 效率與可讀性：是否具備合理效率與清晰結構？
    * 實務準則：是否使用合適的套件與方法？
    * **`analysis_result` 與 `plot_data_df` 使用情形：**
        * 是否正確設定 `analysis_result` 為圖檔檔名（若為圖）？
        * 是否正確設定為 DataFrame / Series 或文字（視結果類型而定）？
        * `plot_data_df` 是否對應圖表資料（或為分析結果的複本）？
    * 圖表儲存：是否將圖表儲存在 `TEMP_DATA_STORAGE` 資料夾？（使用 os.path.join）

2.  **資料分析品質：**
    * 相關性：分析是否符合使用者問題與資料屬性？
    * 準確性：計算與邏輯是否可能正確？
    * 方法選擇：是否選用合適的分析或視覺化方法？
    * `plot_data_df` 內容：是否合理對應圖表資料？

3.  **圖表品質（若有圖）：**
    * 合適性：圖表類型是否適合資料與提問？
    * 清晰性：是否有標題、座標軸、圖例？是否易於理解？

4.  **報告品質（若有報告）：**
    * 清晰與簡潔：報告是否清楚易懂？
    * 洞察力：是否從 `data_content_for_judge` 或圖表中萃取出有價值的發現？
    * 回應查詢：是否回應使用者原始問題？
    * 客觀性：是否以數據為依據，避免主觀推論？

5.  **整體表現與建議：**
    * 請為 AI 助手的回應打分（1-10 分，10 為優秀）
    * 給予 1-2 項具體建議，以提升對此類問題的未來回應品質

審查意見：""")


# --- Sidebar 側邊欄設定 ---
with st.sidebar:
    # 顯示標題「模型選擇」
    st.header("⚙️ 模型選擇")
    st.markdown("""
    **模型選擇**  
    1.    **工作者模式**：針對使用AI 分析對話時的回覆模型
    2.    **評判模式**：針對使用AI 分析對話時工作者模式進行評判
    """)
    # 模型選擇下拉選單（工作模型）
    # 使用者可以選擇用於分析的 LLM 模型
    st.session_state.selected_worker_model = st.selectbox("選擇工作者模式：", AVAILABLE_MODELS,
                                                          index=AVAILABLE_MODELS.index(
                                                              st.session_state.selected_worker_model))

    # 模型選擇下拉選單（評判模型）
    # 使用者可以選擇用於評估分析結果的 LLM 模型
    st.session_state.selected_judge_model = st.selectbox("選擇評判模型：", AVAILABLE_MODELS,
                                                         index=AVAILABLE_MODELS.index(
                                                             st.session_state.selected_judge_model))

    # 顯示標題「上傳數據」
    st.header("📤 上傳數據")

    # CSV 檔案上傳功能（只允許上傳 .csv 檔）
    uploaded_csv_file = st.file_uploader("上傳您的 CSV 檔案：", type="csv", key="csv_uploader")

    # 可選的文字檔描述欄位上傳（只允許上傳 .txt 檔）
    uploaded_desc_file = st.file_uploader("可選：上傳列描述（.txt）：", type="txt", key="desc_uploader")

    # 如果使用者上傳了 CSV 檔案
    if uploaded_csv_file is not None:
        reprocess = False  # 是否需要重新處理檔案的旗標

        # 如果上傳的 CSV 檔與之前的不一樣
        if st.session_state.get("data_source_name") != uploaded_csv_file.name:
            reprocess = True

        # 如果上傳的描述檔不同，也需要重新處理
        if uploaded_desc_file and st.session_state.get("desc_file_name") != uploaded_desc_file.name:
            reprocess = True

        # 如果先前有描述檔，但現在沒上傳，代表描述檔被移除
        if not uploaded_desc_file and st.session_state.get("desc_file_name") is not None:
            reprocess = True

        # 如果只有 CSV，且與之前相同，則不需要重新處理
        if uploaded_desc_file is None and st.session_state.get("desc_file_name") is None and not reprocess:
            pass
        # 如果兩個檔案都存在且與之前相同，也不處理
        elif uploaded_desc_file is not None and \
                st.session_state.get("desc_file_name") == uploaded_desc_file.name and \
                st.session_state.get("data_source_name") == uploaded_csv_file.name:
            pass
        else:
            reprocess = True  # 其他情況則需要重新處理

        # 如果需要重新處理資料
        if reprocess:
            with st.spinner("正在處理 CSV 與描述檔..."):
                # 呼叫函式處理檔案內容與摘要
                if load_csv_and_get_summary(uploaded_csv_file, uploaded_desc_file):
                    st.success(f"CSV 檔案 '{st.session_state.data_source_name}' 已成功處理。")
                    if uploaded_desc_file:
                        st.success(f"描述檔案 '{st.session_state.desc_file_name}' 已成功處理。")
                    else:
                        st.info("未提供描述檔案，或描述檔已移除。")
                    
                    # 系統訊息：處理完成並加入對話記錄
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"已處理 '{st.session_state.data_source_name}'" +
                                   (f" 並使用描述檔 '{st.session_state.desc_file_name}'。" if st.session_state.desc_file_name else "。") +
                                   " 請查看資料品質儀表板或其他分頁。"
                    })

                    st.rerun()  # 重新載入介面
                else:
                    st.error("處理 CSV 或描述檔失敗。")

    # 如果目前已有資料載入
    if st.session_state.current_dataframe is not None:
        st.subheader("目前載入的檔案：")
        st.write(
            f"**{st.session_state.data_source_name}**（{len(st.session_state.current_dataframe)} 筆資料 × {len(st.session_state.current_dataframe.columns)} 欄位）")
        if st.session_state.desc_file_name:
            st.write(f"附帶描述檔：**{st.session_state.desc_file_name}**")

        # 按鈕：清除目前載入資料與對話內容
        if st.button("清除資料與對話內容", key="clear_data_btn"):
            # 要清除的 Session 變數
            keys_to_reset = [
                "current_dataframe", "data_summary", "data_source_name", "desc_file_name",
                "current_analysis_artifacts", "messages", "lc_memory",
                "cdo_initial_report_text", "other_perspectives_text", "strategy_text", "cdo_workflow_stage",
                "trigger_code_generation", "trigger_report_generation", "trigger_judging", "trigger_html_export",
                "report_target_data_path", "report_target_plot_path", "report_target_query"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # 重置對話訊息
            st.session_state.messages = [{"role": "assistant", "content": "資料與對話已重置。請重新上傳 CSV。"}]
            # 重置 LangChain 記憶體
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}

            # 刪除暫存資料夾內的所有檔案
            cleaned_files_count = 0
            if os.path.exists(TEMP_DATA_STORAGE):
                for item_name in os.listdir(TEMP_DATA_STORAGE):
                    item_path = os.path.join(TEMP_DATA_STORAGE, item_name)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                            cleaned_files_count += 1
                        # 如有需要，也可以刪除子資料夾
                        # elif os.path.isdir(item_path):
                        #     shutil.rmtree(item_path)
                    except Exception as e:
                        st.warning(f"無法刪除暫存檔案 {item_path}：{e}")
            st.success(f"資料、對話與 {cleaned_files_count} 個暫存檔案已清除。")
            st.rerun()

    # 下方提示訊息區
    st.markdown("---")  # 水平線分隔

    # 顯示目前選用的模型
    st.info(f"工作模型：**{st.session_state.selected_worker_model}**\n\n評判模型：**{st.session_state.selected_judge_model}**")
    
    # 顯示暫存資料夾路徑
    st.info(f"暫存資料夾位置：`{os.path.abspath(TEMP_DATA_STORAGE)}`")

    # 安全性警告
    st.warning("⚠️ **安全性提醒：** 此程式使用 `exec()` 執行 AI 產生的程式碼，僅供展示用途，請勿用於生產環境。")




# --- 主介面：根據資料是否上傳決定是否顯示分頁 ---
if st.session_state.current_dataframe is not None:
    # 分頁標題
    tab_titles = ["📊 資料品質儀表板", "🔍 資料探索器", "👨‍💼 CDO 分析流程", "💬 AI 分析對話", "🎛️ PyGWalker 探索器", "📋 運營診斷報告", "💼 諮詢服務回饋"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_titles)

    # --- 分頁一：資料品質儀表板 ---
    with tab1:
        generate_data_quality_dashboard(st.session_state.current_dataframe.copy())

    # --- 分頁二：資料探索器 ---
    with tab2:
        st.header("🔍 資料探索器")
        if st.session_state.data_summary:
            with st.expander("查看完整資料摘要（JSON 格式）", expanded=False):
                st.json(st.session_state.data_summary)
            if st.session_state.data_summary.get("user_provided_column_descriptions"):
                with st.expander("使用者提供的欄位說明", expanded=True):
                    st.markdown(st.session_state.data_summary["user_provided_column_descriptions"])
        else:
            st.write("目前尚無資料摘要。")
        with st.expander(f"顯示前 5 筆資料（{st.session_state.data_source_name}）"):
            st.dataframe(st.session_state.current_dataframe.head())
        with st.expander(f"顯示後 5 筆資料（{st.session_state.data_source_name}）"):
            st.dataframe(st.session_state.current_dataframe.tail())

    # --- 分頁三：CDO 分析流程 ---
    with tab3:
        st.header("👨‍💼 CDO 主導分析流程")
        st.markdown("啟動 AI 分析流程：由 CDO 描述資料（可包含使用者欄位說明），各部門 VP 提出觀點，CDO 彙整策略建議。")

        # 啟動按鈕
        if st.button("🚀 啟動 CDO 分析流程", key="start_cdo_workflow_btn"):
            # 初始化流程階段與內容
            st.session_state.cdo_workflow_stage = "initial_description"
            st.session_state.cdo_initial_report_text = None
            st.session_state.other_perspectives_text = None
            st.session_state.strategy_text = None
            # 新增訊息與記憶
            st.session_state.messages.append({"role": "assistant",
                                              "content": f"開始使用 **{st.session_state.selected_worker_model}** 進行 CDO 初始資料描述..."})
            st.session_state.lc_memory.save_context(
                {"user_query": f"使用者啟動了 CDO 分析流程：{st.session_state.data_source_name}"},
                {"output": "請求進行 CDO 初始描述。"})
            st.rerun()

        # 根據目前階段呼叫不同 prompt 執行
        worker_llm = get_llm_instance(st.session_state.selected_worker_model)
        current_stage = st.session_state.get("cdo_workflow_stage")

        # 第一步：CDO 初始資料描述
        if current_stage == "initial_description":
            if worker_llm and st.session_state.data_summary:
                with st.spinner(f"CDO（{st.session_state.selected_worker_model}）正在描述資料..."):
                    try:
                        memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                        prompt_inputs = {
                            "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                            "chat_history": memory_ctx.get("chat_history", "")
                        }
                        response = worker_llm.invoke(
                            cdo_initial_data_description_prompt_template.format_prompt(**prompt_inputs))
                        st.session_state.cdo_initial_report_text = response.content if hasattr(response, 'content') else response.get('text', "CDO 回報產生失敗。")
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": f"**CDO 的初始描述（使用 {st.session_state.selected_worker_model}）:**\n\n{st.session_state.cdo_initial_report_text}"})
                        st.session_state.lc_memory.save_context({"user_query": "CDO 初始描述請求"},
                                                                {"output": f"CDO 回報前 100 字：{st.session_state.cdo_initial_report_text[:100]}..."})
                        st.session_state.cdo_workflow_stage = "departmental_perspectives"
                        st.rerun()
                    except Exception as e:
                        st.error(f"CDO 描述階段發生錯誤：{e}")
                        st.session_state.cdo_workflow_stage = None
            else:
                st.error("缺少模型或資料摘要，無法執行 CDO 流程。")

        # 第二步：VP 部門觀點
        if current_stage == "departmental_perspectives" and st.session_state.cdo_initial_report_text:
            with st.spinner(f"部門主管（{st.session_state.selected_worker_model}）正在提出觀點..."):
                try:
                    memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                    prompt_inputs = {
                        "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                        "chat_history": memory_ctx.get("chat_history", ""),
                        "cdo_initial_report": st.session_state.cdo_initial_report_text
                    }
                    response = worker_llm.invoke(individual_perspectives_prompt_template.format_prompt(**prompt_inputs))
                    st.session_state.other_perspectives_text = response.content if hasattr(response, 'content') else response.get('text', "部門觀點產生失敗。")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**部門主管觀點（{st.session_state.selected_worker_model}）:**\n\n{st.session_state.other_perspectives_text}"})
                    st.session_state.lc_memory.save_context({"user_query": "部門觀點請求"},
                                                            {"output": f"部門觀點前 100 字：{st.session_state.other_perspectives_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "strategy_synthesis"
                    st.rerun()
                except Exception as e:
                    st.error(f"部門觀點階段錯誤：{e}")
                    st.session_state.cdo_workflow_stage = None

        # 第三步：CDO 統整策略建議
        if current_stage == "strategy_synthesis" and st.session_state.other_perspectives_text:
            with st.spinner(f"CDO（{st.session_state.selected_worker_model}）正在統整策略建議..."):
                try:
                    memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                    prompt_inputs = {
                        "data_summary": json.dumps(st.session_state.data_summary, indent=2),
                        "chat_history": memory_ctx.get("chat_history", ""),
                        "cdo_initial_report": st.session_state.cdo_initial_report_text,
                        "generated_perspectives_from_others": st.session_state.other_perspectives_text
                    }
                    response = worker_llm.invoke(synthesize_analysis_suggestions_prompt_template.format_prompt(**prompt_inputs))
                    st.session_state.strategy_text = response.content if hasattr(response, 'content') else response.get('text', "策略彙整失敗。")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**CDO 的最終策略建議（{st.session_state.selected_worker_model}）:**\n\n{st.session_state.strategy_text}\n\n請前往「AI 分析對話」分頁進行後續分析。"})
                    st.session_state.lc_memory.save_context({"user_query": "請求策略統整"},
                                                            {"output": f"策略前 100 字：{st.session_state.strategy_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "completed"
                    st.success("CDO 分析流程已完成 ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"策略統整階段錯誤：{e}")
                    st.session_state.cdo_workflow_stage = None

        # 顯示歷程摘要（展開框）
        if st.session_state.cdo_initial_report_text:
            with st.expander("📋 CDO 初始資料描述", expanded=(current_stage in ["initial_description", "departmental_perspectives", "strategy_synthesis", "completed"])):
                st.markdown(st.session_state.cdo_initial_report_text)
        if st.session_state.other_perspectives_text:
            with st.expander("👥 部門觀點", expanded=(current_stage in ["departmental_perspectives", "strategy_synthesis", "completed"])):
                st.markdown(st.session_state.other_perspectives_text)
        if st.session_state.strategy_text:
            with st.expander("🎯 CDO 最終策略建議", expanded=(current_stage in ["strategy_synthesis", "completed"])):
                st.markdown(st.session_state.strategy_text)

    # --- 分頁四：AI 分析對話 ---
    with tab4:
        st.header("💬 AI 分析對話")
        st.caption("與 Worker AI 對話產生分析，亦可請 Judge AI 評估其品質與建議。")

        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # 顯示執行結果（圖表、表格、文字）
                    if message["role"] == "assistant" and "executed_result" in message:
                        exec_res = message["executed_result"]
                        res_type = exec_res.get("type")
                        orig_query = message.get("original_user_query", st.session_state.current_analysis_artifacts.get("original_user_query", "未知查詢"))

                        # 顯示表格
                        if res_type == "table":
                            if exec_res.get("data_path") and os.path.exists(exec_res["data_path"]):
                                try:
                                    df_disp = pd.read_csv(exec_res["data_path"])
                                    st.dataframe(df_disp)
                                    if st.button(f"📊 為此表格產生報告##{i}", key=f"rep_tbl_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True
                                        st.session_state.report_target_data_path = exec_res["data_path"]
                                        st.session_state.report_target_plot_path = None
                                        st.session_state.report_target_query = orig_query
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"表格顯示錯誤：{e}")
                            elif exec_res.get("dataframe_result") is not None:
                                st.dataframe(exec_res.get("dataframe_result"))
                                st.caption("此表格來自 DataFrame 結果")

                        # 顯示圖表
                        elif res_type == "plot":
                            if exec_res.get("plot_path") and os.path.exists(exec_res["plot_path"]):
                                st.image(exec_res["plot_path"])
                                report_button_label = "📄 為此圖表產生報告"
                                target_data_for_report = None
                                if exec_res.get("data_path") and os.path.exists(exec_res["data_path"]):
                                    report_button_label += "（含資料）"
                                    target_data_for_report = exec_res["data_path"]

                                if st.button(f"{report_button_label}##{i}", key=f"rep_plot_btn_{i}_tab4"):
                                    st.session_state.trigger_report_generation = True
                                    st.session_state.report_target_data_path = target_data_for_report
                                    st.session_state.report_target_plot_path = exec_res["plot_path"]
                                    st.session_state.report_target_query = orig_query
                                    st.rerun()
                            else:
                                st.warning(f"找不到圖表：{exec_res.get('plot_path', '未指定路徑')}")

                        # 顯示文字輸出
                        elif res_type == "text":
                            st.markdown(f"**輸出內容：**\n```\n{exec_res.get('value', '無文字輸出')}\n```")

                        # 顯示報告
                        elif res_type == "report_generated":
                            if exec_res.get("report_path") and os.path.exists(exec_res["report_path"]):
                                st.markdown(f"_報告已產生：`{os.path.abspath(exec_res['report_path'])}`_")

                        # 顯示評審按鈕
                        artifacts_for_judge = st.session_state.get("current_analysis_artifacts", {})
                        can_judge = artifacts_for_judge.get("generated_code") and (
                            artifacts_for_judge.get("executed_data_path") or
                            artifacts_for_judge.get("executed_dataframe_result") is not None or
                            artifacts_for_judge.get("plot_image_path") or
                            artifacts_for_judge.get("executed_text_output") or
                            (res_type == "text" and exec_res.get("value"))
                        )
                        if can_judge:
                            if st.button(f"⚖️ 評估此分析##{i}", key=f"judge_btn_{i}_tab4"):
                                st.session_state.trigger_judging = True
                                st.rerun()

                    # 顯示 Judge AI 的評論
                    if message["role"] == "assistant" and "critique_text" in message:
                        with st.expander(f"查看 {st.session_state.selected_judge_model} 的評論", expanded=True):
                            st.markdown(message["critique_text"])
                        if st.button(f"📄 匯出 PDF 報告##{i}", key=f"pdf_exp_btn_{i}_tab4"):
                            pdf_artifacts = st.session_state.current_analysis_artifacts.copy()
                            if not pdf_artifacts.get("executed_dataframe_result") and pdf_artifacts.get("executed_data_path"):
                                try:
                                    pdf_artifacts["executed_dataframe_result"] = pd.read_csv(pdf_artifacts["executed_data_path"])
                                except:
                                    pass
                            with st.spinner("正在產生 PDF..."):
                                pdf_path = export_analysis_to_pdf(pdf_artifacts)
                                if pdf_path and os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as f_pdf:
                                        st.download_button("下載 PDF 報告", f_pdf, os.path.basename(pdf_path), "application/pdf", key=f"dl_pdf_{i}_tab4_{datetime.datetime.now().timestamp()}")
                                    st.success(f"PDF 報告產生成功：{os.path.basename(pdf_path)}")
                                else:
                                    st.error("PDF 產生失敗")

                        if st.button(f"📄 匯出 Bento HTML 報告##{i}", key=f"html_exp_btn_{i}_tab4"):
                            st.session_state.trigger_html_export = True
                            st.rerun()

        # 最下方輸入欄位
        if user_query := st.chat_input("請輸入分析需求（Worker 模型將自動產生並執行程式碼）...", key="user_query_input_tab4"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
                st.warning("請先從側邊欄上傳並處理 CSV 檔案。")
                st.session_state.messages.append({"role": "assistant", "content": "請先提供 CSV 資料才能進行分析。"})
            else:
                worker_llm_chat = get_llm_instance(st.session_state.selected_worker_model)
                if not worker_llm_chat:
                    st.error(f"Worker 模型 {st.session_state.selected_worker_model} 未啟用")
                    st.session_state.messages.append({"role": "assistant", "content": f"Worker 模型 {st.session_state.selected_worker_model} 無法使用"})
                else:
                    st.session_state.current_analysis_artifacts = {"original_user_query": user_query}
                    st.session_state.trigger_code_generation = True
                    st.rerun()

    # --- 分頁五：PyGWalker ---
    with tab5:
        import streamlit as st
        import pygwalker as pyg
        import streamlit.components.v1 as components

        st.title('🎛️ PygWalker 互動式探索')
        st.subheader('拖放欄位即可生成圖表，並可透過 AI 問答進行分析')

        # 1. 檢查 session_state 中是否已有載入的 DataFrame
        if 'current_dataframe' in st.session_state and st.session_state.current_dataframe is not None:
            
            # 2. 顯示按鈕，當按鈕被點擊時，將 session_state 的狀態設為 True
            st.info("點擊下方按鈕以載入互動分析介面。請注意，載入過程可能需要幾秒鐘。")
            if st.button("🚀 啟動互動式分析 (PyGWalker)"):
                st.session_state.show_pygwalker = True

            # 3. 只有在按鈕被點擊後 (狀態為 True)，才顯示 PyGWalker
            if st.session_state.get('show_pygwalker', False):
                # 從 session_state 獲取 DataFrame
                df = st.session_state.current_dataframe
                
                source_name = st.session_state.get('data_source_name', '已載入的資料')
                st.success(f"✅ 正在分析「{source_name}」。")

                # 使用 pygwalker 生成 HTML 並嵌入
                pyg_html = pyg.walk(df, env='Streamlit', return_html=True, dark='dark')
                
                # 嵌入元件並提供一個按鈕來隱藏它
                components.html(pyg_html, height=800, scrolling=True)
                
                if st.button("收起分析視窗"):
                    st.session_state.show_pygwalker = False
                    st.rerun() # 立即重新整理頁面以隱藏元件

        else:
            # 如果 session_state 中沒有 DataFrame，提示使用者先去上傳檔案
            st.warning("⚠️ 請先在「資料上傳」頁面載入您的 CSV 檔案，才能在此進行探索。")

    # --- 分頁六：運營診斷報告 ---
    with tab6:
        import sys
        sys.path.append('.')
        from generate_pdf_report import generate_pdf_report

        if st.session_state.current_dataframe is not None:
            df = st.session_state.current_dataframe
            # 收集數據概況
            data_summary = f"""
                1. 數據規模：共 {len(df)} 條記錄
                2. 商品類型：{', '.join(df['Item_Type'].unique())}
                3. 門店數量：{len(df['Outlet_Identifier'].unique())} 家
                4. 平均商品價格：{df['Item_MRP'].mean():.2f}
                5. 商品重量範圍：{df['Item_Weight'].min():.2f} - {df['Item_Weight'].max():.2f}
            """
            # 收集風險分析結果
            risk_metrics = {
                '銷售預測準確度': f"{1 - df['Item_MRP'].std() / df['Item_MRP'].mean():.2%}",
                '庫存週轉率': f"{len(df) / len(df['Item_Type'].unique()):.2f}",
                '商品多樣性': f"{len(df['Item_Type'].unique())}",
                '價格區間覆蓋率': f"{(df['Item_MRP'].max() - df['Item_MRP'].min()) / df['Item_MRP'].max():.2%}"
            }
            risk_summary = "\n".join([f"{k}：{v}" for k, v in risk_metrics.items()])
            # 生成診斷報告
            with st.spinner('正在生成運營診斷報告...'):
                diagnosis = generate_operation_diagnosis(data_summary, risk_summary)
            # 顯示診斷報告
            st.markdown("## 📊 數據概況")
            st.text(data_summary)
            st.markdown("## ⚠️ 風險指標")
            for metric, value in risk_metrics.items():
                st.metric(metric, value)
            st.markdown("## 📋 運營診斷報告")
            st.markdown(diagnosis)
            # 生成PDF報告
            pdf_data = generate_pdf_report(data_summary, risk_summary, diagnosis)
            # 添加下載按鈕
            st.download_button(
                label="下載診斷報告 (PDF)",
                data=pdf_data,
                file_name="運營診斷報告.pdf",
                mime="application/pdf"
            )
        else:
            st.info('請先在首頁上傳數據文件')


                    # --- 分頁七：商業諮詢服務回饋 ---
        with tab7:
            st.header("📞 諮詢服務回饋")
            st.markdown("請留下您的需求，我們將盡快與您聯繫！")

            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input("姓名")
                user_email = st.text_input("電子郵件")
            with col2:
                phone = st.text_input("連絡電話")
                subject = st.text_input("主旨", value="諮詢服務回饋")

            message = st.text_area("需求或留言", height=150)

            if st.button("✉️ 送出"):
                if not user_email or not message:
                    st.warning("請填寫電子郵件與留言內容")
                else:
                    try:
                        import smtplib, ssl
                        from email.mime.text import MIMEText
                        from email.mime.multipart import MIMEMultipart

                        receiver = EMAIL_SENDER  # 收件者，可自行調整
                        mime_msg = MIMEMultipart()
                        mime_msg["From"] = EMAIL_SENDER
                        mime_msg["To"] = receiver
                        mime_msg["Subject"] = f"[諮詢回饋] {subject}"

                        body = f"""親愛的客戶您好，

                        感謝您使用我們的諮詢服務。以下是您的諮詢內容：

                        諮詢類型：{subject}

                        您的諮詢內容：
                        {message}
 
                        如果您有任何問題，歡迎隨時與我們聯繫。

                        祝 商祺
                        您的分析團隊
                        """
                        mime_msg.attach(MIMEText(body, "plain", "utf-8"))
                        

                        context = ssl.create_default_context()
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
                            smtp.sendmail(EMAIL_SENDER, receiver, mime_msg.as_string())

                        st.success("已成功送出，感謝您的回饋！")

                        # --- 使用 AI 自動回覆使用者留言 ---
                        with st.spinner("AI 回覆生成中..."):
                            try:
                                ai_prompt = f"""你是一位專業的商業顧問，請針對以下客戶需求提供友善且具體的中文回覆與建議：\n\n客戶需求：\n{message}\n\n回覆："""
                                ai_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "system", "content": "你是一位專業的商業顧問，擅長以清晰、具體的方式解決客戶問題。"},
                                              {"role": "user", "content": ai_prompt}],
                                    temperature=0.7,
                                    max_tokens=500
                                )
                                ai_reply = ai_response.choices[0].message.content.strip()
                            except Exception as e:
                                ai_reply = f"生成回覆時發生錯誤：{e}"
                        st.markdown("---")
                        st.subheader("🤖 AI 回覆")
                        st.markdown(ai_reply)
                    except Exception as e:
                        st.error(f"寄送失敗：{e}")




# --- 若尚未上傳資料，顯示提示與過往對話 ---

elif st.session_state.active_page == 'operation_optimization':
    st.markdown("<h1 style='text-align: center;'>運營優化</h1>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        # --- 通用數據概況與風險 ---
        data_summary, risk_metrics = summarize_dataframe_generic(st.session_state.df)
        risk_summary = "\n".join([f"{k}：{v}" for k, v in risk_metrics.items()])
        
        # 生成診斷報告
        with st.spinner('正在生成運營診斷報告...'):
            diagnosis = generate_operation_diagnosis(data_summary, risk_summary)
            
        # 顯示診斷報告
        st.markdown("## 📊 數據概況")
        st.text(data_summary)
        
        st.markdown("## ⚠️ 風險指標")
        for metric, value in risk_metrics.items():
            st.metric(metric, value)
        
        st.markdown("## 📋 運營診斷報告")
        st.markdown(diagnosis)
        
        # 生成PDF報告
        pdf_data = generate_pdf_report(data_summary, risk_summary, diagnosis)
        
        # 添加下載按鈕
        st.download_button(
            label="下載診斷報告 (PDF)",
            data=pdf_data,
            file_name="運營診斷報告.pdf",
            mime="application/pdf"
        )
        
    else:
        st.info('請先在首頁上傳數據文件')



else:
    st.info("👋 歡迎！請使用側邊欄上傳 CSV 檔案（也可以上傳包含欄位描述的 .txt 檔）以開始分析流程。")
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])






# --- 程式碼生成邏輯 ---
if st.session_state.get("trigger_code_generation", False):
    st.session_state.trigger_code_generation = False  # 關閉觸發器，避免重複觸發
    user_query = st.session_state.messages[-1]["content"]  # 取得使用者的最後一則輸入作為查詢

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        gen_code_str = ""

        # 顯示正在產生程式碼的提示
        msg_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** 正在為下列需求生成程式碼：'{html.escape(user_query)}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} 正在產生 Python 程式碼..."):
            try:
                worker_llm_code_gen = get_llm_instance(st.session_state.selected_worker_model)
                mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                data_sum_prompt = json.dumps(st.session_state.data_summary, indent=2) if st.session_state.data_summary else "{}"

                # 將使用者需求、資料摘要與聊天記憶帶入提示模板中
                formatted_code_gen_prompt = code_generation_prompt_template.format_prompt(
                    data_summary=data_sum_prompt,
                    user_query=user_query,
                    chat_history=mem_ctx.get("chat_history", "")
                )

                # 呼叫 LLM 模型生成程式碼
                response = worker_llm_code_gen.invoke(formatted_code_gen_prompt)
                gen_code_str = response.content if hasattr(response, 'content') else response.get('text', "")

                # 移除 markdown 語法標記（```python）
                for prefix in ["```python\n", "```python", "```\n", "```"]:
                    if gen_code_str.startswith(prefix):
                        gen_code_str = gen_code_str[len(prefix):]
                if gen_code_str.endswith("\n```"):
                    gen_code_str = gen_code_str[:-len("\n```")]
                elif gen_code_str.endswith("```"):
                    gen_code_str = gen_code_str[:-len("```")]
                gen_code_str = gen_code_str.strip()

                # 儲存生成的程式碼到分析物件中
                st.session_state.current_analysis_artifacts["generated_code"] = gen_code_str

                # 顯示程式碼內容，並提示即將執行
                assist_base_content = f"🔍 **{st.session_state.selected_worker_model} 針對 '{html.escape(user_query)}' 所產生的程式碼如下：**\n```python\n{gen_code_str}\n```\n"
                msg_placeholder.markdown(assist_base_content + "\n⏳ 正在執行程式碼...")
            except Exception as e:
                # 錯誤處理
                err_msg = f"產生程式碼時發生錯誤：{html.escape(str(e))}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query},
                                                        {"output": f"程式碼產生錯誤：{e}"})
                if msg_placeholder:
                    msg_placeholder.empty()
                st.rerun()

        if gen_code_str:
            curr_assist_resp_msg = {"role": "assistant", "content": assist_base_content,
                                    "original_user_query": user_query}
            with st.spinner("正在執行生成的 Python 程式碼..."):
                # 執行生成的程式碼
                exec_result = code_executor.execute_code(gen_code_str, st.session_state.current_dataframe.copy())

                # 根據執行結果更新分析物件中的欄位
                if exec_result.get("data_path"):
                    st.session_state.current_analysis_artifacts["executed_data_path"] = exec_result["data_path"]
                if exec_result.get("plot_path"):
                    st.session_state.current_analysis_artifacts["plot_image_path"] = exec_result["plot_path"]
                if exec_result.get("type") == "text" and exec_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = exec_result.get("value")
                if exec_result.get("dataframe_result") is not None:
                    st.session_state.current_analysis_artifacts["executed_dataframe_result"] = exec_result.get("dataframe_result")

                # 特殊處理 plot_specific_data_df：若圖表用資料存在，就讀入
                if exec_result.get("type") == "plot" and exec_result.get("data_path") and "plot_data_for" in os.path.basename(exec_result.get("data_path")):
                    try:
                        st.session_state.current_analysis_artifacts["plot_specific_data_df"] = pd.read_csv(exec_result.get("data_path"))
                    except Exception as e:
                        st.warning(f"無法讀取圖表資料 plot_specific_data_df：{exec_result.get('data_path')}，錯誤：{e}")

                # 根據執行結果顯示訊息與記憶
                llm_mem_output = ""
                if exec_result["type"] == "error":
                    curr_assist_resp_msg["content"] += f"\n⚠️ **執行錯誤：**\n```\n{html.escape(exec_result['message'])}\n```"
                    if str(st.session_state.current_analysis_artifacts.get("executed_text_output", "")).startswith("Code executed, but"):
                        st.session_state.current_analysis_artifacts["executed_text_output"] = f"執行錯誤：{html.escape(str(exec_result.get('final_analysis_result_value', '未知錯誤')))}"
                    llm_mem_output = f"執行錯誤：{html.escape(exec_result['message'][:100])}..."
                else:
                    curr_assist_resp_msg["content"] += "\n✅ **程式碼執行成功！**"
                    curr_assist_resp_msg["executed_result"] = exec_result

                    # 額外補充檔案位置
                    if exec_result.get("data_path"):
                        curr_assist_resp_msg["content"] += f"\n💾 資料儲存於：`{os.path.abspath(exec_result['data_path'])}`"
                    if exec_result.get("plot_path"):
                        curr_assist_resp_msg["content"] += f"\n🖼️ 圖表儲存於：`{os.path.abspath(exec_result['plot_path'])}`"
                    if exec_result.get("data_path") and "plot_data_for" in os.path.basename(exec_result.get("data_path", "")):
                        curr_assist_resp_msg["content"] += "（此為圖表專用資料）"

                    # 產出摘要供記憶使用
                    if exec_result["type"] == "table":
                        llm_mem_output = f"已產生資料表：{os.path.basename(exec_result.get('data_path', 'N/A'))}"
                    elif exec_result["type"] == "plot":
                        llm_mem_output = f"已產生圖表：{os.path.basename(exec_result.get('plot_path', 'N/A'))}"
                        if exec_result.get("data_path"):
                            llm_mem_output += f"（圖表資料：{os.path.basename(exec_result.get('data_path'))}）"
                    elif exec_result["type"] == "text":
                        llm_mem_output = f"已產生文字輸出：{str(exec_result.get('value', ''))[:50]}..."
                    else:
                        llm_mem_output = "程式執行完成，但無法辨識結果類型。"

                # 儲存記憶與更新訊息記錄
                st.session_state.lc_memory.save_context(
                    {"user_query": f"{user_query}\n---生成的程式碼---\n{gen_code_str}\n---結束---"},
                    {"output": llm_mem_output})
                st.session_state.messages.append(curr_assist_resp_msg)
                if msg_placeholder:
                    msg_placeholder.empty()
                st.rerun()




# --- 報告生成邏輯 ---
if st.session_state.get("trigger_report_generation", False):  # 如果觸發生成報告
    st.session_state.trigger_report_generation = False  # 關閉觸發器，避免重複執行

    # 取得必要的資訊
    data_path_rep = st.session_state.get("report_target_data_path")  # 報告所依據的資料路徑
    plot_path_rep = st.session_state.get("report_target_plot_path")  # 圖表圖片路徑
    query_led_to_data = st.session_state.report_target_query         # 觸發這次報告的使用者查詢內容
    worker_llm_rep = get_llm_instance(st.session_state.selected_worker_model)  # 取得 LLM 實例

    # 檢查是否有足夠資訊可生成報告
    if not worker_llm_rep or not st.session_state.data_summary or (
            not data_path_rep and not plot_path_rep):
        st.error("無法生成報告：缺少 LLM 模型、資料摘要或資料/圖表路徑。")
    else:
        # 預設為文字或圖片描述報告
        csv_or_text_content_rep = "無可用資料，僅為描述圖表或缺乏數據。"
        if data_path_rep and os.path.exists(data_path_rep):
            try:
                with open(data_path_rep, 'r', encoding='utf-8') as f_rep_data:
                    csv_or_text_content_rep = f_rep_data.read()
            except Exception as e_read:
                st.error(f"讀取報告用資料檔案時發生錯誤：{html.escape(str(e_read))}")
                st.rerun()
        elif not data_path_rep and plot_path_rep:
            st.info("僅生成圖表的描述性報告。")

        # 建立圖表說明字串
        plot_info_for_prompt = "無圖表可用"
        if plot_path_rep and os.path.exists(plot_path_rep):
            plot_info_for_prompt = f"圖表圖片可於 '{os.path.basename(plot_path_rep)}' 取得。"
            if data_path_rep and "plot_data_for" in os.path.basename(data_path_rep):
                plot_info_for_prompt += f" 與圖表相關的資料位於 '{os.path.basename(data_path_rep)}'。"

        # 顯示 Spinner 與訊息提示
        with st.chat_message("assistant"):
            rep_spinner_container = st.empty()
            rep_spinner_container.markdown(
                f"📝 **{st.session_state.selected_worker_model}** 正在撰寫報告：'{html.escape(query_led_to_data)}'...")
            with st.spinner("正在生成報告..."):
                try:
                    mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                    data_sum_prompt = json.dumps(st.session_state.data_summary, indent=2) if st.session_state.data_summary else "{}"

                    # 組合提示內容
                    prompt_inputs = {
                        "table_data_csv_or_text": csv_or_text_content_rep,
                        "original_data_summary": data_sum_prompt,
                        "user_query_that_led_to_data": query_led_to_data,
                        "chat_history": mem_ctx.get("chat_history", ""),
                        "plot_info_if_any": plot_info_for_prompt
                    }

                    # 呼叫模型生成報告
                    response = worker_llm_rep.invoke(report_generation_prompt_template.format_prompt(**prompt_inputs))
                    rep_text = response.content if hasattr(response, 'content') else response.get('text', "報告生成失敗。")

                    # 儲存報告文字
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in query_led_to_data[:30])
                    filepath = os.path.join(TEMP_DATA_STORAGE, f"report_for_{safe_query}_{timestamp}.txt")
                    with open(filepath, "w", encoding='utf-8') as f_write_rep:
                        f_write_rep.write(rep_text)

                    # 更新 artifacts 與狀態
                    st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                    st.session_state.current_analysis_artifacts["report_query"] = query_led_to_data

                    # 加入聊天訊息與保存記憶
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📊 **{st.session_state.selected_worker_model} 針對 '{html.escape(query_led_to_data)}' 所產生的報告內容：**\n\n{rep_text}",
                        "original_user_query": query_led_to_data,
                        "executed_result": {
                            "type": "report_generated",
                            "report_path": filepath,
                            "data_source_path": data_path_rep or "無",
                            "plot_source_path": plot_path_rep or "無"
                        }
                    })
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"使用者請求產生報告：'{query_led_to_data}'"},
                        {"output": f"產出報告內容（前100字）：{rep_text[:100]}..."}
                    )
                    if rep_spinner_container:
                        rep_spinner_container.empty()
                    st.rerun()
                except Exception as e_rep_gen:
                    # 錯誤處理
                    err_msg_rep = f"產生報告過程中發生錯誤：{html.escape(str(e_rep_gen))}"
                    st.error(err_msg_rep)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg_rep})
                    if rep_spinner_container:
                        rep_spinner_container.empty()
                    st.rerun()

        # 最後清除報告用的暫存資料路徑
        for key in ["report_target_data_path", "report_target_plot_path", "report_target_query"]:
            if key in st.session_state:
                del st.session_state[key]


# --- 評判邏輯 ---
if st.session_state.get("trigger_judging", False):  # 如果觸發評判
    st.session_state.trigger_judging = False  # 重設觸發器

    # 讀取相關資訊
    artifacts_judge = st.session_state.current_analysis_artifacts
    judge_llm_instance = get_llm_instance(st.session_state.selected_judge_model)
    orig_query_artifacts = artifacts_judge.get("original_user_query", "（無原始查詢內容）")

    # 若模型不可用或無代碼可評，顯示錯誤
    if not judge_llm_instance or not artifacts_judge.get("generated_code"):
        st.error("無法取得評判模型或未找到可供評估的 AI 生成程式碼。")
    else:
        try:
            code_content = artifacts_judge.get("generated_code", "（未提供 Python 程式碼）")

            # 資料內容，預設為無資料
            data_content_for_judge = "尚未提供 AI 工作者輸出的資料或檔案。"
            if artifacts_judge.get("executed_dataframe_result") is not None:
                data_content_for_judge = f"資料框架結果（僅顯示前 5 筆）：\n{artifacts_judge['executed_dataframe_result'].head().to_string()}"
            elif artifacts_judge.get("executed_data_path") and os.path.exists(artifacts_judge["executed_data_path"]):
                with open(artifacts_judge["executed_data_path"], 'r', encoding='utf-8') as f_data_judge:
                    data_content_for_judge = f_data_judge.read(500)
                    if len(data_content_for_judge) == 500:
                        data_content_for_judge += "\n...（內容截斷）"
            elif artifacts_judge.get("executed_text_output"):
                data_content_for_judge = f"AI 工作者輸出文字：{artifacts_judge.get('executed_text_output')}"

            # 分析報告內容
            report_content_judge = "尚未產生文字報告。"
            if artifacts_judge.get("generated_report_path") and os.path.exists(artifacts_judge["generated_report_path"]):
                with open(artifacts_judge["generated_report_path"], 'r', encoding='utf-8') as f_report_judge:
                    report_content_judge = f_report_judge.read()

            # 圖像資料
            plot_img_path_judge = artifacts_judge.get("plot_image_path", "N/A")
            plot_info_for_judge_prompt = "未產生圖表或未提及圖像。"
            if plot_img_path_judge != "N/A":
                if os.path.exists(plot_img_path_judge):
                    plot_info_for_judge_prompt = f"已產生圖表，位置：'{os.path.basename(plot_img_path_judge)}'。"
                    if artifacts_judge.get("plot_specific_data_df") is not None and not artifacts_judge.get("plot_specific_data_df").empty:
                        plot_info_for_judge_prompt += f" 對應圖表資料（前 5 筆）：\n{artifacts_judge['plot_specific_data_df'].head().to_string()}"
                    elif artifacts_judge.get("executed_data_path") and "plot_data_for" in os.path.basename(artifacts_judge.get("executed_data_path", "")):
                        plot_info_for_judge_prompt += f" 對應圖表資料檔：'{os.path.basename(artifacts_judge.get('executed_data_path'))}'。"
                else:
                    plot_info_for_judge_prompt = f"預期圖像檔 '{os.path.basename(plot_img_path_judge)}' 並未找到。"

            # 評判過程：顯示 Spinner 與提示
            with st.chat_message("assistant"):
                critique_spinner_container = st.empty()
                critique_spinner_container.markdown(
                    f"⚖️ **{st.session_state.selected_judge_model}** 正在對以下查詢進行評價：'{html.escape(orig_query_artifacts)}'...")
                with st.spinner("正在生成評價..."):
                    data_sum_prompt = json.dumps(st.session_state.data_summary, indent=2) if st.session_state.data_summary else "{}"

                    # 建構提示內容
                    judge_inputs = {
                        "python_code": code_content,
                        "data_content_for_judge": data_content_for_judge,
                        "report_text_content": report_content_judge,
                        "original_user_query": orig_query_artifacts,
                        "data_summary": data_sum_prompt,
                        "plot_image_path": plot_img_path_judge,
                        "plot_info_for_judge": plot_info_for_judge_prompt
                    }

                    # 呼叫 LLM 評判模型
                    response = judge_llm_instance.invoke(judging_prompt_template.format_prompt(**judge_inputs))
                    critique_text = response.content if hasattr(response, 'content') else response.get('text', "評價生成失敗。")

                    # 儲存評價結果
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in orig_query_artifacts[:30])
                    critique_filepath = os.path.join(TEMP_DATA_STORAGE, f"critique_on_{safe_query}_{timestamp}.txt")
                    with open(critique_filepath, "w", encoding='utf-8') as f_critique:
                        f_critique.write(critique_text)

                    # 更新狀態並顯示訊息
                    st.session_state.current_analysis_artifacts["generated_critique_path"] = critique_filepath
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚖️ **模型 {st.session_state.selected_judge_model} 對查詢 '{html.escape(orig_query_artifacts)}' 的評價已產生（檔案儲存於 `{os.path.abspath(critique_filepath)}`）：**",
                        "critique_text": critique_text
                    })
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"已請求評價：'{orig_query_artifacts}'"},
                        {"output": f"評價前 100 字：{critique_text[:100]}..."}
                    )
                    if critique_spinner_container:
                        critique_spinner_container.empty()
                    st.rerun()

        except Exception as e_judge:
            # 捕捉例外並提示
            err_msg_judge = f"評價過程中發生錯誤：{html.escape(str(e_judge))}"
            st.error(err_msg_judge)
            st.session_state.messages.append({"role": "assistant", "content": err_msg_judge})
            if 'critique_spinner_container' in locals() and critique_spinner_container:
                critique_spinner_container.empty()
            st.rerun()




# --- HTML 報告匯出邏輯 ---
if st.session_state.get("trigger_html_export", False):
    # 重設觸發狀態，準備相關資料
    st.session_state.trigger_html_export = False  # 重設匯出狀態
    artifacts_html = st.session_state.current_analysis_artifacts  # 包含程式碼與分析結果的資訊
    cdo_report_html = st.session_state.get("cdo_initial_report_text")  # CDO 初步報告內容（純文字）
    data_summary_html_dict = st.session_state.get("data_summary")  # 資料摘要（字典格式）
    main_df_for_html = st.session_state.get("current_dataframe")  # 主資料表（DataFrame）
    # 驗證關鍵資料是否齊全
    if not artifacts_html or not cdo_report_html or not data_summary_html_dict or main_df_for_html is None:
        st.error(
            "無法產生 HTML：缺少關鍵資訊（分析結果、CDO 報告、資料摘要或主要資料集）。請確認已上傳 CSV，執行過 CDO 流程與分析指令。")
    # 生成 HTML 報告並提供下載
    else:
        with st.chat_message("assistant"):
            html_spinner_container = st.empty()  # 占位顯示區
            html_spinner_container.markdown(f"⏳ 正在產生 Bento 風格 HTML 報告...")
            with st.spinner("產生 HTML 報告中..."):
                html_file_path = generate_bento_html_report_python(artifacts_html, cdo_report_html,
                                                                   data_summary_html_dict, main_df_for_html)

            # 若產生成功，顯示下載按鈕與成功訊息
                if html_file_path and os.path.exists(html_file_path):
                    st.session_state.current_analysis_artifacts["generated_html_report_path"] = html_file_path
                    with open(html_file_path, "rb") as fp_html:
                        st.download_button(
                            label="📥 下載 Bento HTML 報告",
                            data=fp_html,
                            file_name=os.path.basename(html_file_path),
                            mime="text/html",
                            key=f"download_html_{datetime.datetime.now().timestamp()}"
                        )

                    success_msg_html = f"Bento HTML 報告已產生：**{os.path.basename(html_file_path)}**。"
                    st.success(success_msg_html)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📄 {success_msg_html}（完整路徑：`{os.path.abspath(html_file_path)}`）"
                    })

# 若失敗，顯示錯誤訊息
                else:
                    error_msg_html = "產生 Bento HTML 報告失敗，請檢查日誌。"
                    st.error(error_msg_html)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，產生 HTML 報告時發生錯誤。錯誤訊息：{error_msg_html}"
                    })

                if html_spinner_container:
                    html_spinner_container.empty()
