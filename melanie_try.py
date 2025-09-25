import streamlit as st
import pandas as pd
import os
import io
import time
import dotenv
from PIL import Image
import numpy as np

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
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    gemini_history = []
    # 確保 history 是一個 list
    if not isinstance(history, list):
        history = []
        
    for msg in history:
        # 相容舊格式與 Langchain 格式
        role = "user" if msg.get("role") in ["human", "user"] else "model"
        content = msg.get("content", "")
        # 確保 content 是 string
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
        st.plotly_chart(px.imshow(df[numeric_cols].corr(numeric_only=True), text_auto=True, aspect="auto", title="數值欄位相關性熱力圖", color_continuous_scale='RdBu_r'), use_container_width=True)
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
