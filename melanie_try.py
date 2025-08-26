import streamlit as st
import pandas as pd
import os
import io
import json
import datetime
import re
import dotenv
from PIL import Image
import numpy as np

# --- Plotly 和 Gemini/Langchain/OpenAI 等核心套件 ---
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from openai import OpenAI
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 初始化與常數定義 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

ROLE_DEFINITIONS = {
    "summarizer": { "name": "📝 摘要專家", "system_prompt": "你是一位專業的摘要專家。你的任務是將提供的任何文本或對話，濃縮成清晰、簡潔的繁體中文摘要。專注於要點和關鍵結論。", "session_id": "summarizer_chat" },
    "creative_writer": { "name": "✍️ 創意作家", "system_prompt": "你是一位充滿想像力的創意作家。你的任務是幫助使用者完成創意寫作，例如寫故事、詩歌、劇本或腦力激盪，全部使用繁體中文。", "session_id": "creative_writer_chat" }
}

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
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    gemini_history = []
    if not isinstance(history, list):
        history = []
    for msg in history:
        role = "user" if msg["role"] == "human" else "model"
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        gemini_history.append({"role": role, "parts": [content]})

    chat = client.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt)
    return response.text

def get_gemini_response_for_image(api_key, user_prompt, image_pil):
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content([user_prompt, image_pil])
        st.session_state.pending_image_for_main_gemini = None
        return response.text
    except Exception as e: return f"錯誤: {e}"

def get_gemini_executive_analysis(api_key, executive_role_name, full_prompt):
    if not api_key: return f"錯誤：專業經理人 ({executive_role_name}) 未能獲取 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e: return f"錯誤: {e}"

def generate_data_profile(df, is_simple=False):
    if df is None or df.empty: return "沒有資料可供分析。"
    
    if is_simple:
        preview_rows = min(5, df.shape[0])
        profile_parts = [
            f"資料共有 {df.shape[0]} 行, {df.shape[1]} 個欄位。",
            f"前 {preview_rows} 筆資料預覽:\n{df.head(preview_rows).to_string()}"
        ]
        return "\n".join(profile_parts)

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
    quality_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_values = df[col].nunique()
        quality_data.append({"欄位": col, "資料類型": dtype, "缺失值比例 (%)": missing_percent, "唯一值數量": unique_values})
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
    quality_df = get_column_quality_assessment(df)
    st.dataframe(quality_df, use_container_width=True)
    st.markdown("---")
    st.markdown("##### 欄位資料分佈")
    plot_col1, plot_col2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    with plot_col1:
        if numeric_cols:
            selected_numeric = st.selectbox("選擇一個數值型欄位查看分佈:", numeric_cols, key="explorer_numeric")
            if selected_numeric:
                fig = px.histogram(df, x=selected_numeric, title=f"'{selected_numeric}' 的分佈", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("無數值型欄位可供分析。")
    with plot_col2:
        if categorical_cols:
            selected_categorical = st.selectbox("選擇一個類別型欄位查看分佈:", categorical_cols, key="explorer_categorical")
            if selected_categorical:
                top_n = st.slider("顯示前 N 個類別", 5, 20, 10, key="explorer_top_n")
                counts = df[selected_categorical].value_counts().nlargest(top_n)
                fig = px.bar(counts, x=counts.index, y=counts.values, title=f"'{selected_categorical}' 的前 {top_n} 個類別分佈", labels={'index':selected_categorical, 'y':'數量'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("無類別型欄位可供分析。")
    st.markdown("##### 數值欄位相關性熱力圖")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="數值欄位相關性熱力圖", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else: st.info("需要至少兩個數值型欄位才能計算相關性。")

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
        "sp_workflow_stage": "idle", "sp_user_query": "", "sp_final_report": ""
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if executive_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[executive_session_id] = []

    with st.sidebar:
        st.header("⚙️ 功能與模式設定")
        st.session_state.use_rag = st.checkbox("啟用 RAG 知識庫 (需 OpenAI Key)", value=st.session_state.use_rag)
        st.session_state.use_multi_stage_workflow = st.checkbox("啟用階段式工作流 (多重記憶)", value=st.session_state.use_multi_stage_workflow, help="預設(不勾選): AI 一次完成所有角色分析 (單一記憶)。勾選: AI 依序完成各角色分析 (多重記憶)，開銷較大。")
        st.session_state.use_simple_explorer = st.checkbox("啟用簡易資料探索器", value=st.session_state.use_simple_explorer, help="勾選後，將在工作流的統計摘要區塊顯示互動式圖表。")
        st.divider()

        st.header("🔑 API 金鑰")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        if st.session_state.use_rag:
            st.text_input("請輸入您的 OpenAI API Key (RAG 功能需要)", type="password", key="openai_api_key_input")
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
            settings = {
                'gemini_api_key_input': st.session_state.get('gemini_api_key_input'),
                'openai_api_key_input': st.session_state.get('openai_api_key_input'),
                'use_rag': st.session_state.get('use_rag'),
                'use_multi_stage_workflow': st.session_state.get('use_multi_stage_workflow'),
                'use_simple_explorer': st.session_state.get('use_simple_explorer')
            }
            st.session_state.clear()
            for key, value in settings.items(): st.session_state[key] = value
            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

    tab_titles = ["💬 主要聊天室", "💼 專業經理人"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    gemini_api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.warning("請在側邊欄輸入您的 Google Gemini API Key 以啟動聊天功能。")
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
                    response = ""
                    prompt_context = ""

                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        retrieved_docs = st.session_state.retriever_chain.invoke(user_input)
                        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                        prompt_context = f"請根據以下上下文回答問題。\n\n[上下文]:\n{context}\n\n"
                    
                    elif not st.session_state.use_rag and st.session_state.get("uploaded_file_path"):
                        try:
                            df = pd.read_csv(st.session_state.uploaded_file_path)
                            data_summary = generate_data_profile(df.head(), is_simple=True)
                            prompt_context = f"請參考以下資料摘要來回答問題。\n\n[資料摘要]:\n{data_summary}\n\n"
                        except Exception as e:
                            st.warning(f"讀取 CSV 檔案時發生錯誤: {e}")
                    
                    if st.session_state.pending_image_for_main_gemini:
                        final_prompt = f"{prompt_context} [問題]:\n{user_input}"
                        response = get_gemini_response_for_image(gemini_api_key, final_prompt, st.session_state.pending_image_for_main_gemini)
                    
                    else:
                        history = st.session_state.chat_histories[session_id][:-1]
                        final_prompt = f"{prompt_context}[問題]:\n{user_input}"
                        response = get_gemini_response_with_history(gemini_client, history, final_prompt)
                        
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    with tabs[1]:
        st.header("💼 專業經理人")
        st.caption(f"目前模式：{'階段式 (多重記憶)' if st.session_state.use_multi_stage_workflow else '整合式 (單一記憶)'} | RAG：{'啟用' if st.session_state.use_rag else '停用'} | 簡易探索器：{'啟用' if st.session_state.use_simple_explorer else '停用'}")

        # --- 模式一：階段式工作流 ---
        if st.session_state.use_multi_stage_workflow:
            st.info("**方法說明**：此流程模擬三位獨立的專家。每一步都是一次獨立的 API 請求，後一位專家的分析基於前一位的書面報告。")
            st.session_state.executive_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("executive_user_query", ""), height=100, key="original_workflow_query")
            can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
            
            if st.button("🚀 啟動階段式分析", disabled=not can_start, key="exec_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                st.session_state.cfo_analysis_text, st.session_state.coo_analysis_text, st.session_state.ceo_summary_text, st.session_state.executive_rag_context = "", "", "", ""
                st.rerun()

            # --- 階段式工作流的後端處理 ---
            if st.session_state.executive_workflow_stage == "cfo_analysis_pending":
                with st.spinner("CFO 正在獨立分析..."):
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
            
            elif st.session_state.executive_workflow_stage == "coo_analysis_pending":
                with st.spinner("COO 正在分析..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    coo_prompt = f"作為營運長(COO)，請基於商業問題、統計摘要、CFO 的財務分析以及相關數據，提供營運層面的策略與潛在風險。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}{rag_context_for_prompt}"
                    st.session_state.coo_analysis_text = get_gemini_executive_analysis(gemini_api_key, "COO", coo_prompt)
                    st.session_state.executive_workflow_stage = "ceo_summary_pending"
                    st.rerun()

            elif st.session_state.executive_workflow_stage == "ceo_summary_pending":
                with st.spinner("CEO 正在總結..."):
                    rag_context_for_prompt = f"\n\n[RAG 檢索出的相關數據]:\n{st.session_state.executive_rag_context}" if st.session_state.executive_rag_context else ""
                    ceo_prompt = f"作為執行長(CEO)，請整合所有資訊，提供一個高層次的決策總結。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[COO 的營運分析]:\n{st.session_state.coo_analysis_text}{rag_context_for_prompt}"
                    st.session_state.ceo_summary_text = get_gemini_executive_analysis(gemini_api_key, "CEO", ceo_prompt)
                    st.session_state.executive_workflow_stage = "completed"
                    full_report = f"### 📊 財務長 (CFO) 分析\n{st.session_state.cfo_analysis_text}\n\n### 🏭 營運長 (COO) 分析\n{st.session_state.coo_analysis_text}\n\n### 👑 執行長 (CEO) 最終決策\n{st.session_state.ceo_summary_text}"
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": full_report})
                    st.rerun()

        # --- 模式二：整合式工作流 ---
        else:
            st.info("**方法說明**：此為預設流程。模擬一個全能的 AI 專業經理人團隊，只發送**一次**請求，AI 在一次生成中完成所有角色思考。")
            st.session_state.sp_user_query = st.text_area("請輸入商業問題以啟動分析:", value=st.session_state.get("sp_user_query", ""), height=100, key="sp_workflow_query")
            can_start_sp = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("sp_user_query"))

            if st.button("🚀 啟動整合分析", disabled=not can_start_sp, key="sp_flow_button"):
                st.session_state.chat_histories[executive_session_id] = []
                st.session_state.sp_workflow_stage = "running"
                with st.spinner("AI 專業經理人團隊正在進行全面分析..."):
                    df = pd.read_csv(st.session_state.uploaded_file_path)
                    data_profile = generate_data_profile(df)
                    st.session_state.executive_data_profile_str = data_profile
                    query = st.session_state.sp_user_query
                    rag_context_str, rag_context = "", ""
                    if st.session_state.use_rag and st.session_state.retriever_chain:
                        rag_context = "\n---\n".join([doc.page_content for doc in st.session_state.retriever_chain.invoke(query)])
                        st.session_state.executive_rag_context = rag_context
                        rag_context_str = f"\n\n**[RAG 檢索出的相關數據]:**\n{rag_context}"
                    
                    mega_prompt = f"""你是一個頂尖的 AI 商業分析團隊，能夠在一次思考中扮演多個專業經理人角色。你的任務是針對給定的商業問題和數據，生成一份包含三個部分的完整分析報告。

請嚴格按照以下結構和要求進行輸出，使用 Markdown 標題來區分每個部分：
---
### 📊 財務長 (CFO) 分析
在此部分，請完全以財務長的角度思考。專注於財務指標、成本效益、投資回報率、毛利率、潛在的財務風險等。你的分析必須完全基於提供的數據。

### 🏭 營運長 (COO) 分析
在此部分，轉換為營運長的角色。你需要思考，在CFO會提出的財務考量下，營運上是否可行？分析潛在的流程、供應鏈、人力資源或執行風險。你的分析需要務實且著重於可執行性。

### 👑 執行長 (CEO) 最終決策
在此部分，作為CEO，請綜合上述的財務(CFO)和營運(COO)分析。不要重複細節，而是提供一個高層次的戰略總結。最終，給出一個明確、果斷的**決策**（例如：批准、駁回、需要更多資料），並列出 2-3 個最重要的**後續行動建議**。
---
現在，請根據以下資訊開始分析：

**[商業問題]:**
{query}

**[資料統計摘要]:**
{data_profile}{rag_context_str}
"""
                    response = get_gemini_executive_analysis(gemini_api_key, "IntegratedTeam", mega_prompt)
                    st.session_state.sp_final_report = response
                    st.session_state.sp_workflow_stage = "completed"
                    st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": response})
                    st.rerun()

        # --- 【核心修正】統一的資料摘要與對話區塊 ---
        workflow_completed = st.session_state.executive_workflow_stage == "completed" or st.session_state.sp_workflow_stage == 'completed'

        if workflow_completed:
            if st.session_state.get('executive_data_profile_str'):
                expander_title = "查看統計摘要與資料探索" if st.session_state.use_simple_explorer else "查看統計摘要"
                with st.expander(expander_title, expanded=False):
                    st.subheader("純文字統計摘要")
                    st.text(st.session_state.executive_data_profile_str)
                    if st.session_state.use_simple_explorer and st.session_state.get("uploaded_file_path"):
                        st.divider()
                        display_simple_data_explorer(pd.read_csv(st.session_state.uploaded_file_path))
            
            if st.session_state.use_rag and st.session_state.get('executive_rag_context'):
                with st.expander("查看 RAG 檢索出的相關資料"):
                    st.markdown(st.session_state.executive_rag_context)

            st.divider()
            st.subheader("分析報告與後續對話")

            for msg in st.session_state.chat_histories[executive_session_id]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_input := st.chat_input("針對以上報告進行提問..."):
                st.session_state.chat_histories[executive_session_id].append({"role": "human", "content": user_input})
                st.chat_message("human").markdown(user_input)

                with st.chat_message("ai"):
                    with st.spinner("正在分析您的問題..."):
                        history = st.session_state.chat_histories[executive_session_id][:-1]
                        system_prompt = "你是一位頂尖的 AI 商業分析顧問，請根據已有的對話上下文（包含初始報告和你之前的回答）來專業地回答使用者的後續問題。"
                        final_prompt = f"{system_prompt}\n\n[新的問題]:\n{user_input}"
                        response = get_gemini_response_with_history(gemini_client, history, final_prompt)
                        st.markdown(response)
                        st.session_state.chat_histories[executive_session_id].append({"role": "ai", "content": response})
                        st.rerun()

    role_tab_offset = 2 
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + role_tab_offset]:
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"])
            session_id = role_info["session_id"]
            if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
            for msg in st.session_state.chat_histories[session_id]:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            if user_input := st.chat_input(f"與 {role_info['name']} 對話...", key=session_id):
                st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
                with st.chat_message("human"): st.markdown(user_input)
                with st.chat_message("ai"):
                    with st.spinner("正在生成回應..."):
                        history = st.session_state.chat_histories[session_id][:-1]
                        final_prompt = f"{role_info['system_prompt']}\n\n{user_input}"
                        response = get_gemini_response_with_history(gemini_client, history, final_prompt)
                        st.markdown(response)
                        st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()
