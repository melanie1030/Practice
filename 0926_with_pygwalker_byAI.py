import streamlit as st
import pandas as pd
import os
import io
import time
import dotenv
from PIL import Image
import numpy as np
import json
import re

# --- Plotly 和 Gemini/Langchain/OpenAI 等核心套件 ---
import plotly.express as px
import google.generativeai as genai
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 初始化與常數定義 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

# --- 基礎輔助函數 ---
def save_uploaded_file(uploaded_file):
    """儲存上傳的檔案到指定目錄"""
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    """處理圖片上傳，準備隨下一則訊息發送"""
    try:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.pending_image_for_main_gemini = Image.open(file_path)
        st.image(st.session_state.pending_image_for_main_gemini, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
    except Exception as e: st.error(f"處理上傳圖片時出錯: {e}")

# --- RAG 核心函式 ---
@st.cache_resource
def create_lc_retriever(file_path: str, openai_api_key: str):
    """建立 LangChain 的 RAG 檢索器"""
    with st.status("正在建立 RAG 知識庫...", expanded=True) as status:
        try:
            status.update(label="步驟 1/3：載入與切割文件...")
            loader = CSVLoader(file_path=file_path, encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            status.update(label=f"步驟 1/3 完成！已切割成 {len(docs)} 個區塊。")
            
            status.update(label="步驟 2/3：呼叫 OpenAI API 生成向量嵌入...")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vector_store = LangChainFAISS.from_documents(docs, embeddings)
            status.update(label="步驟 2/3 完成！向量嵌入已生成。")
            
            status.update(label="步驟 3/3：檢索器準備完成！", state="complete", expanded=False)
            return vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            st.error(f"建立知識庫過程中發生嚴重錯誤: {e}")
            status.update(label="建立失敗", state="error")
            return None

# --- Gemini API 相關函式 ---
def get_gemini_client(api_key):
    """取得 Gemini 客戶端"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def get_gemini_response_with_history(client, history, user_prompt):
    """帶有歷史紀錄的 Gemini 對話"""
    gemini_history = []
    if not isinstance(history, list): history = []
    for msg in history:
        role = "user" if msg.get("role") in ["human", "user"] else "model"
        content = msg.get("content", "")
        if not isinstance(content, str): content = str(content)
        gemini_history.append({"role": role, "parts": [content]})
    
    chat = client.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt)
    return response.text

def get_gemini_response_for_image(api_key, user_prompt, image_pil):
    """處理圖片輸入的 Gemini 請求"""
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([user_prompt, image_pil])
        st.session_state.pending_image_for_main_gemini = None
        return response.text
    except Exception as e: return f"錯誤: {e}"

def get_gemini_executive_analysis(api_key, executive_role_name, full_prompt):
    """
    執行高管分析的核心 Gemini API 呼叫。
    此版本強制要求 AI 必須提供圖表建議。
    """
    if not api_key: return f"錯誤：專業經理人 ({executive_role_name}) 未能獲取 Gemini API Key。"
    
    plotting_instruction = """
**[圖表建議格式指令]**:
在你的分析文字結束後，你**必須**根據你的分析，提供一個最能總結核心觀點的圖表建議。請**務必**提供一個 JSON 物件，格式如下，不得省略：
```json
{"plotting_suggestion": {"plot_type": "類型", "x": "X軸欄位名", "y": "Y軸欄位名", "title": "圖表標題", "explanation": "一句話解釋此圖表的核心洞見"}}
```
其中 `plot_type` 必須是 `bar`, `scatter`, `line`, `histogram` 中的一種。對於 `histogram`，`y` 欄位可以省略或設為 `null`。你**絕對不能**回答 `{"plotting_suggestion": null}` 或省略這個 JSON 物件。
"""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        final_prompt = f"{full_prompt}\n\n{plotting_instruction}"
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e: return f"錯誤: {e}"

def generate_data_profile(df, is_simple=False):
    """生成 DataFrame 的文字摘要"""
    if df is None or df.empty: return "沒有資料可供分析。"
    if is_simple:
        preview_rows = min(5, df.shape[0])
        return f"資料共有 {df.shape[0]} 行, {df.shape[1]} 個欄位。\n前 {preview_rows} 筆資料預覽:\n{df.head(preview_rows).to_string()}"
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    profile_parts = [f"資料形狀: {df.shape}", f"欄位資訊:\n{buffer.getvalue()}"]
    try: profile_parts.append(f"\n數值欄位統計:\n{df.describe(include='number').to_string()}")
    except: pass
    try: profile_parts.append(f"\n類別欄位統計:\n{df.describe(include=['object', 'category']).to_string()}")
    except: pass
    profile_parts.append(f"\n前 5 筆資料:\n{df.head().to_string()}")
    return "\n".join(profile_parts)

# --- 圖表生成輔助函式 ---
def parse_plotting_suggestion(response_text: str):
    """從 AI 的回應中解析出圖表建議 JSON 和分析文字"""
    json_pattern = r"```json\s*(\{.*?\})\s*```|(\{.*plotting_suggestion.*?\})"
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    if not match:
        return None, response_text.strip()
        
    json_str = match.group(1) if match.group(1) else match.group(2)
    
    try:
        analysis_text = response_text.replace(match.group(0), "").strip()
        suggestion_data = json.loads(json_str)
        plotting_info = suggestion_data.get("plotting_suggestion")
        return plotting_info, analysis_text
        
    except (json.JSONDecodeError, AttributeError):
        analysis_text = response_text.replace(match.group(0), "").strip()
        st.warning("AI 提供了格式不正確的圖表建議，已忽略。")
        return None, analysis_text

def create_plot_from_suggestion(df: pd.DataFrame, suggestion: dict):
    """根據 AI 提供的結構化建議來生成 Plotly 圖表"""
    if not suggestion: return None
    
    plot_type = suggestion.get("plot_type", "").lower()
    x_col = suggestion.get("x")
    y_col = suggestion.get("y")
    title = suggestion.get("title", f"AI 建議圖表")

    if not all([plot_type, x_col]):
        st.warning(f"AI 建議的資訊不完整 (缺少圖表類型或X軸)，無法繪圖。")
        return None

    if x_col not in df.columns or (y_col and y_col not in df.columns):
        st.warning(f"AI 建議的欄位 '{x_col}' 或 '{y_col}' 不存在於資料中，無法繪圖。")
        return None

    fig = None
    try:
        if plot_type == "bar":
            if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                grouped_df = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.bar(grouped_df, x=x_col, y=y_col, title=title)
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', title=title)
        elif plot_type == "scatter":
            if not y_col: st.warning("散佈圖需要 y 軸欄位。"); return None
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif plot_type == "line":
            if not y_col: st.warning("折線圖需要 y 軸欄位。"); return None
            sorted_df = df.sort_values(by=x_col)
            fig = px.line(sorted_df, x=x_col, y=y_col, title=title)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_col, title=title)
        else:
            st.warning(f"尚不支援的圖表類型: '{plot_type}'"); return None
        return fig
    except Exception as e:
        st.error(f"根據 AI 建議 '{title}' 繪製圖表時發生錯誤: {e}")
        return None

# --- 資料探索器核心函數 ---
@st.cache_data
def get_overview_metrics(df):
    if df is None or df.empty: return 0, 0, 0, 0, 0
    num_rows, num_cols = df.shape
    missing_percentage = (df.isnull().sum().sum() / (num_rows * num_cols)) * 100 if (num_rows * num_cols) > 0 else 0
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
    duplicate_rows = df.duplicated().sum()
    return num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows

@st.cache_data
def get_column_quality_assessment(df):
    if df is None or df.empty: return pd.DataFrame()
    quality_data = [{"欄位": col, "資料類型": str(df[col].dtype), "缺失值比例 (%)": (df[col].isnull().sum() / len(df)) * 100 if len(df) > 0 else 0, "唯一值數量": df[col].nunique()} for col in df.columns]
    return pd.DataFrame(quality_data)

# --- 圖表生成 Agent 核心函式 ---
def get_df_context(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_string()
    context = f"""
以下是您需要分析的 Pandas DataFrame 的詳細資訊。DataFrame 變數名稱為 `df`。
1. DataFrame 的基本資訊 (df.info()):
{info_str}
2. DataFrame 的前 5 筆資料 (df.head()):
{head_str}"""
    return context

def run_pandas_analyst_agent(api_key: str, df: pd.DataFrame, user_query: str) -> str:
    try:
        llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo", temperature=0)
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        agent_prompt = f"""
作為一名資深的數據分析師，你的任務是深入探索提供的 DataFrame (`df`)。
使用者的目標是："{user_query}"
請你執行以下步驟：
1. 徹底地探索和分析 `df`，找出其中最重要、最有趣、最值得透過視覺化來呈現的一個核心洞見。
2. 不要生成任何繪圖程式碼。
3. 你的最終輸出**必須是**一段簡潔的文字摘要。這段摘要需要清楚地描述你發現的洞見，並建議應該繪製什麼樣的圖表來展示這個洞見。
現在，請開始分析。"""
        response = pandas_agent.invoke({"input": agent_prompt})
        return response['output']
    except Exception as e:
        return f"Pandas Agent 執行時發生錯誤: {e}"

def generate_plot_code(api_key: str, df_context: str, user_query: str, analyst_conclusion: str = None) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        if analyst_conclusion:
            prompt = f"""
你是一位頂尖的 Python 數據視覺化專家，精通使用 Plotly Express 函式庫。
你的任務是根據數據分析師的結論和使用者的原始目標，編寫一段 Python 程式碼來生成最合適的圖表。
**數據分析師的結論:**
{analyst_conclusion}
**原始使用者目標:**
"{user_query}"
**DataFrame 的資訊:**
{df_context}
**嚴格遵守以下規則:**
1. 你只能生成 Python 程式碼，絕對不能包含任何文字解釋、註解或 ```python 標籤。
2. 程式碼必須基於上述**數據分析師的結論**來生成。
3. 生成的程式碼必須使用 `plotly.express` (匯入為 `px`)。
4. DataFrame 的變數名稱固定為 `df`。
5. 最終生成的圖表物件必須賦值給一個名為 `fig` 的變數。
現在，請生成程式碼："""
        else:
            prompt = f"""
你是一位頂尖的 Python 數據視覺化專家，精通使用 Plotly Express 函式庫。
你的任務是根據提供的 DataFrame 資訊和使用者的要求，編寫一段 Python 程式碼來生成一個圖表。
**嚴格遵守以下規則:**
1. 你只能生成 Python 程式碼，絕對不能包含任何文字解釋、註解或 ```python 標籤。
2. 生成的程式碼必須使用 `plotly.express` (匯入為 `px`)。
3. DataFrame 的變數名稱固定為 `df`。
4. 最終生成的圖表物件必須賦值給一個名為 `fig` 的變數。
**DataFrame 的資訊:**
{df_context}
**使用者的繪圖要求:**
"{user_query}"
現在，請生成程式碼："""
        response = model.generate_content(prompt)
        code = response.text.strip().replace("```python", "").replace("```", "").strip()
        return code
    except Exception as e:
        return f"繪圖程式碼生成時發生錯誤: {e}"

# --- 可重用的高管工作流函式 ---
def run_executive_workflow(api_key: str, df: pd.DataFrame, user_query: str, rag_context: str, conversation_history: str = ""):
    """
    執行完整的高管分析工作流 (整合式或階段式)。
    - user_query: 使用者當前的問題或指令。
    - conversation_history: 格式化後的歷史對話紀錄。
    """
    data_profile = generate_data_profile(df)
    
    history_prompt_injection = ""
    if conversation_history:
        history_prompt_injection = f"""
**[先前對話的完整歷史紀錄]:**
---
{conversation_history}
---
請務必將上述歷史紀錄納入考量，以確保你的分析具有連續性，避免重複已經討論過的觀點，並根據最新的指示進行調整。
"""

    final_report = ""
    plot_suggestion = None

    if not st.session_state.use_multi_stage_workflow:
        # --- 整合式 (單一記憶) 工作流 ---
        with st.spinner("AI 經理人團隊正在協作分析中..."):
            single_stage_prompt = f"""
你將扮演一個由 CFO、COO 和 CEO 組成的高階主管團隊，對一份資料進行分析。
{history_prompt_injection}
**當前使用者目標/指令:** {user_query}
**資料摘要:**\n{data_profile}
**相關知識庫上下文 (RAG):** {rag_context if rag_context else "無"}

**你的任務:**
請嚴格按照以下順序和格式，生成一份全新的、完整的分析報告來回應**當前使用者目標/指令**：
1.  **CFO 分析:**
    - 以 `### CFO (財務長) 分析報告` 作為開頭。
    - 從財務角度分析。
2.  **COO 分析:**
    - 以 `### COO (營運長) 分析報告` 作為開頭。
    - 從營運效率角度分析。
3.  **CEO 總結:**
    - 以 `### CEO (執行長) 戰略總結` 作為開頭。
    - **整合** CFO 和 COO 的觀點，並針對**當前使用者目標/指令**提出戰略總結與建議。

**最終圖表建議:**
在所有分析結束後，由 CEO 決定並**提供一個最關鍵的圖表**來總結本次分析，並遵循指定的 JSON 格式。"""
            full_response = get_gemini_executive_analysis(api_key, "Executive Team", single_stage_prompt)
            plot_suggestion, final_report = parse_plotting_suggestion(full_response)
    else:
        # --- 階段式 (多重記憶) 工作流 ---
        cfo_analysis_text, coo_analysis_text, ceo_summary_text = "", "", ""
        with st.spinner("CFO 正在分析中..."):
            cfo_prompt = f"""
作為專業的財務長(CFO)，請根據以下資訊進行分析。
{history_prompt_injection}
**當前使用者目標/指令:** {user_query}
**資料摘要:**\n{data_profile}
**相關知識庫上下文 (RAG):** {rag_context if rag_context else "無"}
**你的任務:** 從財務角度分析，提供數據驅動的洞見。**在此階段不需提供圖表建議。**"""
            cfo_response = get_gemini_executive_analysis(api_key, "CFO", cfo_prompt)
            _, cfo_analysis_text = parse_plotting_suggestion(cfo_response)

        with st.spinner("COO 正在分析中..."):
            coo_prompt = f"""
作為專業的營運長(COO)，請根據以下資訊進行分析。
{history_prompt_injection}
**CFO 已完成的分析:**\n{cfo_analysis_text}
**當前使用者目標/指令:** {user_query}
**資料摘要:**\n{data_profile}
**相關知識庫上下文 (RAG):** {rag_context if rag_context else "無"}
**你的任務:** 從營運效率角度分析。**在此階段不需提供圖表建議。**"""
            coo_response = get_gemini_executive_analysis(api_key, "COO", coo_prompt)
            _, coo_analysis_text = parse_plotting_suggestion(coo_response)

        with st.spinner("CEO 正在總結中..."):
            ceo_prompt = f"""
作為公司的執行長(CEO)，你的任務是基於你的團隊分析，提供全面的戰略總結。
{history_prompt_injection}
**財務長 (CFO) 的分析報告:**\n{cfo_analysis_text}
**營運長 (COO) 的分析報告:**\n{coo_analysis_text}
**當前使用者目標/指令:** {user_query}
**你的任務:** 整合 CFO 和 COO 的觀點，針對**當前使用者目標/指令**提供高層次的戰略總結和建議。**最後，你必須提供一個最關鍵的圖表建議來總結本次分析。**"""
            ceo_response = get_gemini_executive_analysis(api_key, "CEO", ceo_prompt)
            plot_suggestion, ceo_summary_text = parse_plotting_suggestion(ceo_response)
        
        final_report = f"### CFO (財務長) 分析報告\n{cfo_analysis_text}\n\n---\n\n### COO (營運長) 分析報告\n{coo_analysis_text}\n\n---\n\n### CEO (執行長) 戰略總結\n{ceo_summary_text}"

    return final_report, plot_suggestion

# --- 主應用入口 ---
def main():
    st.set_page_config(page_title="Gemini 多功能 AI 助理", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理")

    executive_session_id = "executive_chat"
    keys_to_init = {
        "use_rag": False, "use_multi_stage_workflow": False,
        "retriever_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "chat_histories": {},
        "executive_user_query": "",
        "plot_code": "",
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state: st.session_state[key] = default_value
    if executive_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[executive_session_id] = []

    with st.sidebar:
        st.header("⚙️ 功能與模式設定")
        st.session_state.use_rag = st.checkbox("啟用 RAG 知識庫", value=st.session_state.use_rag)
        st.session_state.use_multi_stage_workflow = st.checkbox("啟用階段式工作流 (多重記憶)", value=st.session_state.use_multi_stage_workflow, help="預設(不勾選): AI 一次完成所有角色分析 (單一記憶)。勾選: AI 依序完成各角色分析 (多重記憶)，開銷較大。")
        st.divider()
        st.header("🔑 API 金鑰")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        st.text_input("請輸入您的 OpenAI API Key", type="password", key="openai_api_key_input", help="RAG 功能與圖表Agent的分析模式會使用此金鑰。")
        st.divider()
        st.header("📁 資料上傳")
        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                st.session_state.last_uploaded_filename = uploaded_file.name
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                st.success(f"檔案 '{uploaded_file.name}' 上傳成功！")
                st.session_state.retriever_chain = None
                if st.session_state.use_rag:
                    openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key: st.error("RAG 功能已啟用，請在上方輸入您的 OpenAI API Key！")
                    else: st.session_state.retriever_chain = create_lc_retriever(file_path, openai_api_key)
        if st.session_state.retriever_chain: st.success("✅ RAG 知識庫已啟用！")
        
        st.header("🖼️ 圖片分析")
        uploaded_image = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg"])
        if uploaded_image: add_user_image_to_main_chat(uploaded_image)
        st.divider()
        if st.button("🗑️ 清除所有對話與資料"):
            settings = {k: st.session_state.get(k) for k in ['gemini_api_key_input', 'openai_api_key_input', 'use_rag', 'use_multi_stage_workflow']}
            st.session_state.clear()
            for key, value in settings.items(): st.session_state[key] = value
            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

    tab_titles = ["💬 主要聊天室", "💼 專業經理人", "📊 圖表生成 Agent"]
    tabs = st.tabs(tab_titles)

    gemini_api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")

    if not gemini_api_key:
        st.warning("請在側邊欄輸入您的 Google Gemini API Key 以啟動主要功能。")
        st.stop()
    gemini_client = get_gemini_client(gemini_api_key)
    
    df = None
    if st.session_state.uploaded_file_path:
        try:
            df = pd.read_csv(st.session_state.uploaded_file_path)
        except Exception as e:
            st.error(f"讀取 CSV 檔案失敗: {e}")

    with tabs[0]:
        st.header("💬 主要聊天室")
        st.caption("可進行一般對話、圖片分析。RAG 問答功能可由側邊欄開關啟用。")
        session_id = "main_chat"
        if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
        for msg in st.session_state.chat_histories[session_id]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if user_input := st.chat_input("請對數據、圖片提問或開始對話..."):
            st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
            with st.chat_message("human"): st.markdown(user_input)
            with st.chat_message("ai"):
                with st.spinner("正在思考中..."):
                    prompt_context = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(user_input)])
                        prompt_context = f"請根據以下上下文回答問題。\n\n[上下文]:\n{context}\n\n"
                    elif not st.session_state.use_rag and df is not None:
                        prompt_context = f"請參考以下資料摘要來回答問題。\n\n[資料摘要]:\n{generate_data_profile(df.head(), is_simple=True)}\n\n"
                        
                    if st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(gemini_api_key, f"{prompt_context} [問題]:\n{user_input}", st.session_state.pending_image_for_main_gemini)
                    else:
                        response = get_gemini_response_with_history(gemini_client, st.session_state.chat_histories[session_id][:-1], f"{prompt_context}[問題]:\n{user_input}")
                    
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    with tabs[1]:
        st.header("💼 專業經理人")
        st.caption(f"目前模式：{'階段式 (多重記憶)' if st.session_state.use_multi_stage_workflow else '整合式 (單一記憶)'} | RAG：{'啟用' if st.session_state.use_rag else '停用'}")

        if df is None:
            st.info("請先在側邊欄上傳 CSV 檔案以啟用此功能。")
        else:
            user_query = st.text_input("請輸入您的分析目標或追問：", key="executive_query", placeholder="例如：分析各產品線的銷售表現")

            if st.button("提交分析 / 追問", key="start_executive_analysis"):
                if not user_query:
                    st.warning("請先輸入您的分析目標！")
                else:
                    is_first_run = not st.session_state.chat_histories[executive_session_id]
                    
                    if is_first_run:
                        st.session_state.executive_user_query = user_query
                        st.session_state.chat_histories[executive_session_id] = []
                        st.session_state.chat_histories[executive_session_id].append({"role": "user", "content": user_query})
                        history_str = ""
                    else:
                        st.session_state.chat_histories[executive_session_id].append({"role": "user", "content": user_query})
                        history_list = []
                        for msg in st.session_state.chat_histories[executive_session_id][:-1]:
                           role = "使用者" if msg['role'] == 'user' else "AI經理人團隊"
                           history_list.append(f"{role}:\n{msg['content']}")
                        history_str = "\n\n".join(history_list)

                    rag_context = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(user_query)])

                    new_report, new_plot_suggestion = run_executive_workflow(
                        api_key=gemini_api_key,
                        df=df,
                        user_query=user_query,
                        rag_context=rag_context,
                        conversation_history=history_str
                    )

                    st.session_state.chat_histories[executive_session_id].append({
                        "role": "ai",
                        "content": new_report,
                        "plot_suggestion": new_plot_suggestion
                    })
                    st.rerun()

            for msg in st.session_state.chat_histories[executive_session_id]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "ai":
                        plot_suggestion = msg.get("plot_suggestion")
                        if plot_suggestion:
                            st.markdown("---")
                            st.write(f"**建議圖表:** {plot_suggestion.get('title', '')}")
                            st.caption(plot_suggestion.get("explanation", ""))
                            fig = create_plot_from_suggestion(df, plot_suggestion)
                            if fig: st.plotly_chart(fig, use_container_width=True)
                            else: st.warning("無法生成建議的圖表。")
                        # 移除 "無需圖表" 的 else 區塊，因為 AI 現在必須提供圖表

    with tabs[2]:
        st.header("📊 圖表生成 Agent")
        st.caption("這是一個使用 Agent 來生成圖表程式碼的範例。")
        
        if df is None:
            st.info("請先在側邊欄上傳 CSV 檔案以啟用此功能。")
        else:
            st.write("#### DataFrame 預覽")
            st.dataframe(df.head())
            
            mode = st.radio("選擇模式：", ("AI 分析師建議", "直接下指令"), horizontal=True, key="agent_mode")

            user_plot_query = st.text_input("請輸入您的繪圖目標：", key="agent_query", placeholder="例如：我想看各個城市的平均房價")

            if st.button("生成圖表", key="agent_generate"):
                df_context = get_df_context(df)
                if mode == "AI 分析師建議":
                    if not openai_api_key:
                        st.error("此模式需要 OpenAI API Key，請在側邊欄輸入。")
                    else:
                        with st.spinner("AI 分析師正在思考最佳圖表..."):
                            analyst_conclusion = run_pandas_analyst_agent(openai_api_key, df, user_plot_query)
                            st.info(f"**分析師結論:** {analyst_conclusion}")
                        with st.spinner("視覺化專家正在生成程式碼..."):
                            code = generate_plot_code(gemini_api_key, df_context, user_plot_query, analyst_conclusion)
                            st.session_state.plot_code = code
                else: # 直接下指令
                    with st.spinner("視覺化專家正在根據您的指令生成程式碼..."):
                        code = generate_plot_code(gemini_api_key, df_context, user_plot_query)
                        st.session_state.plot_code = code
            
            if st.session_state.plot_code:
                st.write("#### 最終圖表")
                try:
                    exec_scope = {'df': df, 'px': px}
                    exec(st.session_state.plot_code, exec_scope)
                    fig = exec_scope.get('fig')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("程式碼未成功生成名為 'fig' 的圖表物件。")
                except Exception as e:
                    st.error(f"執行生成的程式碼時出錯: {e}")

                with st.expander("查看/編輯生成的程式碼"):
                    st.code(st.session_state.plot_code, language='python')


if __name__ == "__main__":
    main()
