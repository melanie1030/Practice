import streamlit as st
import pandas as pd
import os
import io
import time
import dotenv
import json # 新增
from PIL import Image
import numpy as np
import pygwalker as pyw # 新增
from pygwalker.api.streamlit import StreamlitRenderer 

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
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    try:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.pending_image_for_main_gemini = Image.open(file_path)
        st.image(st.session_state.pending_image_for_main_gemini, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
    except Exception as e: st.error(f"處理上傳圖片時出錯: {e}")

# --- RAG 核心函式 ---
@st.cache_resource
def create_lc_retriever(file_path: str, openai_api_key: str):
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
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response_with_history(client, history, user_prompt):
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
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([user_prompt, image_pil])
        st.session_state.pending_image_for_main_gemini = None
        return response.text
    except Exception as e: return f"錯誤: {e}"

def get_gemini_executive_analysis(api_key, executive_role_name, full_prompt):
    if not api_key: return f"錯誤：專業經理人 ({executive_role_name}) 未能獲取 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e: return f"錯誤: {e}"
    
def generate_data_profile(df, is_simple=False):
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

def display_simple_data_explorer(df):
    st.subheader("互動式資料探索")
    st.markdown("---")
    st.markdown("##### 關鍵指標")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df)
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("總行數", f"{num_rows:,}")
    kpi_cols[1].metric("總列數", f"{num_cols:,}")
    kpi_cols[2].metric("缺失值比例", f"{missing_percentage:.2f}%")
    kpi_cols[3].metric("數值型欄位", f"{numeric_cols_count}")
    kpi_cols[4].metric("重複行數", f"{duplicate_rows:,}")
    st.markdown("##### 欄位品質評估")
    st.dataframe(get_column_quality_assessment(df), use_container_width=True)
    st.markdown("---")
    st.markdown("##### 欄位資料分佈")
    plot_col1, plot_col2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    with plot_col1:
        if numeric_cols:
            selected_numeric = st.selectbox("選擇一個數值型欄位查看分佈:", numeric_cols, key="explorer_numeric")
            if selected_numeric:
                st.plotly_chart(px.histogram(df, x=selected_numeric, title=f"'{selected_numeric}' 的分佈", marginal="box"), use_container_width=True)
        else: st.info("無數值型欄位可供分析。")
    with plot_col2:
        if categorical_cols:
            selected_categorical = st.selectbox("選擇一個類別型欄位查看分佈:", categorical_cols, key="explorer_categorical")
            if selected_categorical:
                top_n = st.slider("顯示前 N 個類別", 5, 20, 10, key="explorer_top_n")
                counts = df[selected_categorical].value_counts().nlargest(top_n)
                st.plotly_chart(px.bar(counts, x=counts.index, y=counts.values, title=f"'{selected_categorical}' 的前 {top_n} 個類別分佈", labels={'index':selected_categorical, 'y':'數量'}), use_container_width=True)
        else: st.info("無類別型欄位可供分析。")
    st.markdown("##### 數值欄位相關性熱力圖")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr_matrix, text_auto=True, aspect="auto", title="數值欄位相關性熱力圖", color_continuous_scale='RdBu_r'), use_container_width=True)
    else: st.info("需要至少兩個數值型欄位才能計算相關性。")


# --- 【新功能】圖表生成 Agent 核心函式 ---
def get_df_context(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_string()
    context = f"""
以下是您需要分析的 Pandas DataFrame 的詳細資訊。
DataFrame 變數名稱為 `df`。

1. DataFrame 的基本資訊 (df.info()):
{info_str}

2. DataFrame 的前 5 筆資料 (df.head()):
{head_str}
    """
    return context

def run_pandas_analyst_agent(api_key: str, df: pd.DataFrame, user_query: str) -> str:
    try:
        llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo", temperature=0)
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        agent_prompt = f"""
作為一名資深的數據分析師，你的任務是深入探索提供的 DataFrame (`df`)。
使用者的目標是："{user_query}"

請你執行以下步驟：
1.  徹底地探索和分析 `df`，找出其中最重要、最有趣、最值得透過視覺化來呈現的一個核心洞見。
2.  不要生成任何繪圖程式碼。
3.  你的最終輸出**必須是**一段簡潔的文字摘要。這段摘要需要清楚地描述你發現的洞見，並建議應該繪製什麼樣的圖表來展示這個洞見。

現在，請開始分析。
"""
        response = pandas_agent.invoke({"input": agent_prompt})
        return response['output']
    except Exception as e:
        return f"Pandas Agent 執行時發生錯誤: {e}"

# 【已修改】generate_plot_code -> generate_plot_params
def generate_plot_params(api_key: str, df_context: str, user_query: str, analyst_conclusion: str = None) -> str:
    """
    讓 AI 回傳一個包含繪圖參數的 JSON 字串，而不是 Python 程式碼。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        allowed_plot_types = "['scatter', 'bar', 'histogram', 'line', 'box', 'pie', 'area']"

        if analyst_conclusion:
            prompt_core = f"""
            根據以下數據分析師的結論和使用者目標，決定如何繪圖。
            **數據分析師的結論:**
            {analyst_conclusion}
            **原始使用者目標:**
            "{user_query}"
            """
        else:
            prompt_core = f"""
            根據以下使用者要求，決定如何繪圖。
            **使用者的繪圖要求:**
            "{user_query}"
            """

        full_prompt = f"""
你是一位頂尖的 Python 數據視覺化專家。你的任務是根據要求，決定生成一個圖表所需的參數。

{prompt_core}

**DataFrame 的資訊:**
{df_context}

**嚴格遵守以下規則:**
1.  你**絕對不能**生成 Python 程式碼。
2.  你的最終輸出**必須是**一個格式嚴格的 JSON 物件，不包含任何 markdown 標籤如 ```json。
3.  JSON 物件必須包含以下幾個鍵 (key):
    - `plot_type`: (字串) 圖表類型，必須是以下列表中的一個: {allowed_plot_types}。
    - `x`: (字串或 null) 對應 DataFrame 的欄位名作為 X 軸。
    - `y`: (字串或 null) 對應 DataFrame 的欄位名作為 Y 軸。
    - `color`: (字串或 null) 對應 DataFrame 的欄位名作為顏色分類。
    - `title`: (字串) 為圖表取一個有意義的標題。
4.  如果某個參數不適用（例如直方圖沒有 y 軸），請將其值設為 `null`。
5.  欄位名稱必須與 DataFrame 資訊中提供的完全一致。

現在，請只生成 JSON 物件：
"""
        response = model.generate_content(full_prompt)
        clean_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return clean_response
    except Exception as e:
        return json.dumps({"error": f"生成繪圖參數時發生錯誤: {e}"})

# 【新函式】將 AI 參數轉換為 Pygwalker Spec
def convert_params_to_pygwalker_spec(params: dict, df: pd.DataFrame) -> str:
    """
    將 AI 生成的簡單參數字典，轉換為 Pygwalker 需要的 spec JSON 字串。
    """
    mark_map = {
        "bar": "bar", "line": "line", "scatter": "point", "area": "area",
        "histogram": "bar", "box": "boxplot"
    }
    
    plot_type = params.get("plot_type", "point")
    vis_spec = {"mark": mark_map.get(plot_type, "point"), "encodings": {}}
    
    channels = {"x": params.get("x"), "y": params.get("y"), "color": params.get("color")}
    
    for channel, field_name in channels.items():
        if field_name and field_name in df.columns:
            dtype = "quantitative" if pd.api.types.is_numeric_dtype(df[field_name]) else "nominal"
            vis_spec["encodings"][channel] = {"field": field_name, "type": dtype}

    # 針對特殊圖表類型進行調整
    if plot_type == "histogram":
        vis_spec["encodings"]["x"]["bin"] = True # 對 x 軸進行分箱
        if "y" not in vis_spec["encodings"]: # 如果 y 沒設定，則設為計數
             vis_spec["encodings"]["y"] = {"aggregate": "count", "type": "quantitative"}

    full_spec = {
        "config": { "geoms": ["auto"], "stack": "stack", "showActions": True, "interactiveScale": True, "layoutMode": "auto" },
        "visSpec": [vis_spec]
    }
    return json.dumps(full_spec)


# --- 主應用入口 ---
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理 ")

    executive_session_id = "executive_chat"
    keys_to_init = {
        "use_rag": False, "use_multi_stage_workflow": False, "use_simple_explorer": False,
        "retriever_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "chat_histories": {},
        "executive_workflow_stage": "idle", "executive_user_query": "",
        "executive_data_profile_str": "", "executive_rag_context": "", "cfo_analysis_text": "",
        "coo_analysis_text": "", "ceo_summary_text": "",
        "sp_workflow_stage": "idle", "sp_user_query": "", "sp_final_report": "",
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state: st.session_state[key] = default_value
    if executive_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[executive_session_id] = []

    with st.sidebar:
        st.header("⚙️ 功能與模式設定")
        st.session_state.use_rag = st.checkbox("啟用 RAG 知識庫", value=st.session_state.use_rag)
        st.session_state.use_multi_stage_workflow = st.checkbox("啟用階段式工作流 (多重記憶)", value=st.session_state.use_multi_stage_workflow, help="預設(不勾選): AI 一次完成所有角色分析 (單一記憶)。勾選: AI 依序完成各角色分析 (多重記憶)，開銷較大。")
        st.session_state.use_simple_explorer = st.checkbox("啟用簡易資料探索器", value=st.session_state.use_simple_explorer, help="勾選後，將在工作流的統計摘要區塊顯示互動式圖表。")
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
            settings = {k: st.session_state.get(k) for k in ['gemini_api_key_input', 'openai_api_key_input', 'use_rag', 'use_multi_stage_workflow', 'use_simple_explorer']}
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
                    elif not st.session_state.use_rag and st.session_state.get("uploaded_file_path"):
                        try:
                            df = pd.read_csv(st.session_state.uploaded_file_path)
                            prompt_context = f"請參考以下資料摘要來回答問題。\n\n[資料摘要]:\n{generate_data_profile(df.head(), is_simple=True)}\n\n"
                        except Exception as e: st.warning(f"讀取 CSV 檔案時發生錯誤: {e}")
                    if st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(gemini_api_key, f"{prompt_context} [問題]:\n{user_input}", st.session_state.pending_image_for_main_gemini)
                    else:
                        response = get_gemini_response_with_history(gemini_client, st.session_state.chat_histories[session_id][:-1], f"{prompt_context}[問題]:\n{user_input}")
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    with tabs[1]:
        st.header("💼 專業經理人")
        st.caption(f"目前模式：{'階段式 (多重記憶)' if st.session_state.use_multi_stage_workflow else '整合式 (單一記憶)'} | RAG：{'啟用' if st.session_state.use_rag else '停用'} | 簡易探索器：{'啟用' if st.session_state.use_simple_explorer else '停用'}")

        if st.session_state.use_multi_stage_workflow:
            st.info("**方法說明**：此流程將依序（CFO->COO->CEO）進行分析，每一步完成後會立刻顯示結果，並自動觸發下一步。")
            st.session_state.executive_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("executive_user_query", ""), height=100, key="original_workflow_query")
            can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
            if st.button("🚀 啟動階段式分析", disabled=not can_start or st.session_state.executive_workflow_stage != "idle", key="exec_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                st.session_state.cfo_analysis_text, st.session_state.coo_analysis_text, st.session_state.ceo_summary_text, st.session_state.executive_rag_context = "", "", "", ""
                st.rerun()

            stage = st.session_state.executive_workflow_stage
            if stage == "cfo_analysis_pending":
                with st.spinner("CFO 正在分析中..."):
                    df = pd.read_csv(st.session_state.uploaded_file_path)
                    st.session_state.executive_data_profile_str = generate_data_profile(df)
                    query = st.session_state.executive_user_query
                    rag_context_for_prompt = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(query)])
                        st.session_state.executive_rag_context = rag_context
                        rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{rag_context}"
                    cfo_prompt = f"作為財務長(CFO)，請基於你的專業知識，並嚴格參考以下提供的資料，為商業問題提供財務角度的簡潔分析。\n\n[商業問題]:\n{query}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}{rag_context_for_prompt}"
                    st.session_state.cfo_analysis_text = get_gemini_executive_analysis(gemini_api_key, "CFO", cfo_prompt)
                    st.session_state.executive_workflow_stage = "coo_analysis_pending"
                    st.rerun() 
            
            elif stage == "coo_analysis_pending":
                with st.spinner("COO 正在分析中..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    coo_prompt = f"作為營運長(COO)，請基於商業問題、統計摘要、CFO 的財務分析以及相關數據，提供營運層面的策略與潛在風險。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}{rag_context_for_prompt}"
                    st.session_state.coo_analysis_text = get_gemini_executive_analysis(gemini_api_key, "COO", coo_prompt)
                    st.session_state.executive_workflow_stage = "ceo_summary_pending"
                    st.rerun()
            
            elif stage == "ceo_summary_pending":
                with st.spinner("CEO 正在總結中..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    ceo_prompt = f"作為執行長(CEO)，請整合所有資訊，提供一個高層次的決策總結。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[COO 的營運分析]:\n{st.session_state.coo_analysis_text}{rag_context_for_prompt}"
                    st.session_state.ceo_summary_text = get_gemini_executive_analysis(gemini_api_key, "CEO", ceo_prompt)
                    st.session_state.executive_workflow_stage = "completed"
                    full_report = f"### 📊 財務長 (CFO) 分析\n{st.session_state.cfo_analysis_text}\n\n### 🏭 營運長 (COO) 分析\n{st.session_state.coo_analysis_text}\n\n### 👑 執行長 (CEO) 最終決策\n{st.session_state.ceo_summary_text}"
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": full_report})
                    st.rerun()

        else: # 整合分析模式
            st.info("**方法說明**：此為預設流程。模擬一個全能的 AI 專業經理人團隊，只發送**一次**請求，AI 在一次生成中完成所有角色思考。")
            st.session_state.sp_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("sp_user_query", ""), height=100, key="sp_workflow_query")
            can_start_sp = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("sp_user_query"))
            
            if st.button("🚀 啟動整合分析", disabled=not can_start_sp or st.session_state.sp_workflow_stage != "idle", key="sp_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.sp_final_report = ""
                st.session_state.sp_workflow_stage = "running"
                st.rerun()

            if st.session_state.sp_workflow_stage == "running":
                with st.spinner("AI 專業經理人團隊正在進行全面分析..."):
                    df = pd.read_csv(st.session_state.uploaded_file_path)
                    data_profile = generate_data_profile(df)
                    st.session_state.executive_data_profile_str = data_profile
                    query = st.session_state.sp_user_query
                    rag_context_str = ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(query)])
                        st.session_state.executive_rag_context = rag_context
                        rag_context_str = f"\n\n**[RAG 檢索出的相關數據]:**\n{rag_context}"
                    
                    mega_prompt = f"""你是一個頂尖的 AI 商業分析團隊，由三位專家組成：財務長(CFO)、營運長(COO)和執行長(CEO)。
你的任務是根據以下提供的商業問題和數據資料，協同完成一份專業的商業分析報告。

**[商業問題]:**
{query}

**[數據統計摘要]:**
{data_profile}

{rag_context_str}

**請嚴格遵循以下格式和步驟輸出報告：**

1.  **📊 財務長 (CFO) 分析:**
    * 從財務角度分析數據，找出關鍵的財務指標、成本結構、營收趨勢或潛在的盈利機會與風險。
    * 使用數據進行支撐，條理清晰地呈現。

2.  **🏭 營運長 (COO) 分析:**
    * 基於CFO的見解和原始數據，從營運效率、生產流程、供應鏈、庫存管理或客戶行為等角度進行分析。
    * 提出具體的營運策略建議或指出需要警惕的營運風險。

3.  **👑 執行長 (CEO) 最終決策:**
    * 整合 CFO 和 COO 的分析。
    * 提供一個高層次的、戰略性的總結。
    * 基於所有資訊，提出 2-3 個明確、可執行的下一步行動建議或最終決策。

現在，請開始分析並生成報告。
"""
                    response = get_gemini_executive_analysis(gemini_api_key, "IntegratedTeam", mega_prompt)
                    st.session_state.sp_final_report = response
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": response})
                    st.session_state.sp_workflow_stage = "completed"
                    st.rerun()

        workflow_has_started = (st.session_state.executive_workflow_stage != "idle" or st.session_state.sp_workflow_stage != 'idle')
        if workflow_has_started:
            if st.session_state.get('executive_data_profile_str'):
                with st.expander("查看統計摘要與資料探索" if st.session_state.use_simple_explorer else "查看統計摘要", expanded=False):
                    st.subheader("純文字統計摘要")
                    st.text(st.session_state.executive_data_profile_str)
                    if st.session_state.use_simple_explorer and st.session_state.get("uploaded_file_path"):
                        st.divider(); display_simple_data_explorer(pd.read_csv(st.session_state.uploaded_file_path))
            if st.session_state.use_rag and st.session_state.get('executive_rag_context'):
                with st.expander("查看 RAG 檢索出的相關資料"):
                    st.markdown(st.session_state.executive_rag_context)

            st.divider()
            st.subheader("分析報告與後續對話")
            
            if executive_session_id in st.session_state.chat_histories:
                for msg in st.session_state.chat_histories[executive_session_id]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            if st.session_state.executive_workflow_stage == "completed" or st.session_state.sp_workflow_stage == "completed":
                if follow_up_query := st.chat_input("針對報告內容進行追問..."):
                    st.session_state.chat_histories[executive_session_id].append({"role": "user", "content": follow_up_query})
                    with st.chat_message("user"):
                        st.markdown(follow_up_query)
                    
                    with st.chat_message("ai"):
                        with st.spinner("AI 正在思考中..."):
                            history_for_follow_up = st.session_state.chat_histories[executive_session_id][:-1]
                            response = get_gemini_response_with_history(gemini_client, history_for_follow_up, follow_up_query)
                            st.markdown(response)
                            st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": response})
                            st.rerun()

    # --- 【 tabs[2] 已完全重構 】---
    with tabs[2]:
        st.header("📊 自然語言圖表生成 Agent")
        st.markdown("上傳 CSV，AI 將為您推薦圖表並在下方互動介面中開啟，您也可以繼續自由探索！")
        
        if not st.session_state.get("uploaded_file_path"):
            st.warning("請先在側邊欄上傳一個 CSV 檔案以啟用此功能。")
        else:
            try:
                df = pd.read_csv(st.session_state.uploaded_file_path)
                st.subheader("資料預覽")
                st.dataframe(df.head())
                st.divider()

                st.subheader("請選擇操作模式")
                agent_mode = st.radio(
                    "模式選擇",
                    ["直接繪圖模式", "分析與繪圖模式"],
                    captions=[
                        "您很清楚要畫什麼圖，請下達具體指令。",
                        "您不確定要畫什麼，希望 AI 先分析數據找出洞見再畫圖。"
                    ],
                    horizontal=True,
                    key="plotting_agent_mode"
                )
                
                st.subheader("請下達您的指令")
                if agent_mode == "直接繪圖模式":
                    user_query = st.text_area(
                        "請輸入具體的繪圖指令：",
                        "範例：畫出 x 軸是 'sepal_length'，y 軸是 'sepal_width' 的散點圖",
                        height=100,
                        key="plot_direct_query"
                    )
                else:
                    user_query = st.text_area(
                        "請輸入模糊的、高層次的分析目標：",
                        "範例：分析這份數據，幫我找出最重要的趨勢並視覺化",
                        height=100,
                        key="plot_analysis_query"
                    )

                if st.button("🚀 生成圖表並在互動介面中開啟", key="plot_generate_button", disabled=(not user_query)):
                    analyst_conclusion = None
                    if agent_mode == "分析與繪圖模式":
                        if not openai_api_key or not gemini_api_key:
                            st.error("分析模式需要同時在側邊欄輸入 Google Gemini 和 OpenAI 的 API Keys！")
                        else:
                            with st.status("執行分析與繪圖工作流...", expanded=True) as status:
                                st.write("第一階段：Pandas Agent 正在進行深度數據分析...")
                                analyst_conclusion = run_pandas_analyst_agent(openai_api_key, df, user_query)
                                st.write("✅ 分析完成！")
                                status.update(label="第一階段分析完成！")
                                st.subheader("🧐 Pandas Agent 的分析結論")
                                st.info(analyst_conclusion)
                    
                    with st.spinner("AI 正在分析並設定圖表..."):
                        df_context = get_df_context(df)
                        # 1. 呼叫 AI 生成簡單參數
                        params_json_str = generate_plot_params(gemini_api_key, df_context, user_query, analyst_conclusion)
                        
                        try:
                            params = json.loads(params_json_str)
                            if "error" in params:
                                st.error(params["error"])
                                st.session_state.pyg_spec = None # 出錯時清除 spec
                            else:
                                st.info("🤖 AI 建議的圖表參數:")
                                st.json(params)

                                # 2. 將 AI 參數翻譯成 Pygwalker Spec
                                st.session_state.pyg_spec = convert_params_to_pygwalker_spec(params, df)

                        except json.JSONDecodeError:
                            st.error(f"AI 回傳的內容不是有效的 JSON 格式: {params_json_str}")
                            st.session_state.pyg_spec = None
                        except Exception as e:
                            st.error(f"轉換參數時發生錯誤: {e}")
                            st.session_state.pyg_spec = None

                # 將 Pygwalker 的渲染邏輯放在按鈕判斷之外，這樣它可以在 spec 更新後重新渲染
                st.divider()
                st.subheader("📊 互動式圖表探索介面 (Pygwalker)")

                # 獲取 spec，如果 session 中沒有，就使用預設值 None
                current_spec = st.session_state.get('pyg_spec', None)

                if current_spec:
                    st.success("AI 已為您設定好初始圖表！您現在可以繼續自由拖曳和探索。")
                else:
                    st.info("請下達指令讓 AI 為您設定圖表，或直接在此介面中手動操作。")

                # 3. 渲染 Pygwalker 元件
                # key 參數是必要的，可以幫助 Streamlit 在 spec 變化時識別並更新元件
                StreamlitRenderer(df, spec=current_spec, dark='dark', key="pygwalker_renderer").explorer()

            except Exception as e:
                st.error(f"處理檔案或繪圖時發生錯誤: {e}")


if __name__ == "__main__":
    main()
