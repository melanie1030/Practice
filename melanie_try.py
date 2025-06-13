import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
from io import StringIO
from PIL import Image
import google.generativeai as genai
import time
import matplotlib.font_manager as fm
import matplotlib
from operator import itemgetter

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
# --- 本地嵌入模型 Import ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Role & Workflow Definitions ---
ROLE_DEFINITIONS = {
    "summarizer": {
        "name": "📝 摘要專家",
        "system_prompt": "你是一位專業的摘要專家。你的任務是將提供的任何文本或對話，濃縮成清晰、簡潔的繁體中文摘要。專注於要點和關鍵結論。",
        "session_id": "summarizer_session",
    },
    "creative_writer": {
        "name": "✍️ 創意作家",
        "system_prompt": "你是一位充滿想像力的創意作家。你的任務是幫助使用者完成創意寫作，例如寫故事、詩歌、劇本或腦力激盪，全部使用繁體中文。",
        "session_id": "creative_writer_session",
    }
}
EXECUTIVE_ROLE_IDS = { "CFO": "cfo_exec", "COO": "coo_exec", "CEO": "ceo_exec" }

# --- 初始化設置 ---
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

# --- LangChain 核心函式 ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "history_store" not in st.session_state: st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

@st.cache_resource
def create_vector_db_from_csv(file_path: str):
    """從 CSV 檔案建立向量資料庫 (100% 使用本地嵌入模型)"""
    with st.status("正在初始化知識庫 (本地模式)...", expanded=True) as status:
        status.update(label="步驟 1/3：正在載入與清理資料...")
        df = pd.read_csv(file_path, encoding='utf-8')
        df.dropna(how='all', inplace=True)
        df.fillna("", inplace=True)
        clean_file_path = os.path.join(UPLOAD_DIR, f"clean_{os.path.basename(file_path)}")
        df.to_csv(clean_file_path, index=False)
        
        loader = CSVLoader(file_path=clean_file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        status.update(label=f"步驟 1/3 完成！已將文件切割成 {len(docs)} 個區塊。")

        status.update(label="步驟 2/3：正在使用本地模型生成向量嵌入 (初次執行會下載模型，請稍候)...")
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        # 強制使用 CPU，避免在沒有 GPU 的環境中出錯
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        
        vector_store = FAISS.from_documents(docs, embeddings)
        status.update(label="步驟 2/3 完成！向量嵌入已在本地生成。")

        status.update(label="步驟 3/3：知識庫準備完成！", state="complete", expanded=False)
    st.success("知識庫建立完成！")
    return vector_store

def create_rag_chain(vector_store):
    """建立一個整合了「對話內記憶」的 RAG 鏈"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    retriever = vector_store.as_retriever()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "請根據「上下文」回答問題。\n[上下文]:\n{context}"),
        ("human", "{input}"),
    ])
    base_rag_chain = ({ "context": itemgetter("input") | retriever, "input": itemgetter("input")} | prompt_template | llm | StrOutputParser())
    rag_chain_with_history = RunnableWithMessageHistory(base_rag_chain, get_session_history, input_messages_key="input", history_messages_key="history")
    return rag_chain_with_history

def create_generic_chat_chain(system_prompt: str):
    """為特定角色建立一個具備「對話內記憶」的通用聊天鏈"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    base_chain = prompt_template | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(base_chain, get_session_history, input_messages_key="input", history_messages_key="history")
    return chain_with_history

# --- 其他 Gemini 呼叫函式 ---
def get_gemini_response_for_image(user_prompt, image_pil):
    """專門用於處理圖片分析的函式"""
    api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content([user_prompt, image_pil])
        st.session_state.pending_image_for_main_gemini = None # 清除待處理圖片
        return response.text
    except Exception as e:
        st.error(f"Gemini 圖片分析請求失敗: {e}")
        return f"錯誤: {e}"

def get_gemini_executive_analysis(executive_role_name, full_prompt):
    api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not api_key: return f"錯誤：高管工作流 ({executive_role_name}) 未能獲取 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"高管分析 ({executive_role_name}) 失敗: {e}")
        return f"錯誤: {e}"
        
def generate_data_profile(df):
    if df is None or df.empty: return "沒有資料可供分析。"
    buffer = StringIO()
    df.info(buf=buffer)
    profile_parts = [f"資料形狀: {df.shape}", f"欄位資訊:\n{buffer.getvalue()}"]
    try: profile_parts.append(f"\n數值欄位統計:\n{df.describe(include='number').to_string()}")
    except: pass
    try: profile_parts.append(f"\n類別欄位統計:\n{df.describe(include=['object', 'category']).to_string()}")
    except: pass
    profile_parts.append(f"\n前 5 筆資料:\n{df.head().to_string()}")
    return "\n".join(profile_parts)

# ------------------------------
# 主應用入口
# ------------------------------
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理")

    # --- 初始化 Session States ---
    keys_to_init = {
        "rag_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "executive_workflow_stage": "idle", "executive_user_query": "",
        "executive_data_profile_str": "", "cfo_analysis_text": "", "coo_analysis_text": "", "ceo_summary_text": "",
        "debug_mode": False
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ 設定")
        st.text_input(
            "請輸入您的 Google Gemini API Key",
            value=st.session_state.get("gemini_api_key_input", ""), type="password", key="gemini_api_key_input"
        )
        st.caption("優先使用此處輸入的金鑰。")
        st.divider()

        st.subheader("📁 資料問答 (RAG)")
        uploaded_file = st.file_uploader("上傳 CSV 以啟用 RAG 問答功能", type=["csv"])
        
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                st.session_state.last_uploaded_filename = uploaded_file.name
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                vector_store = create_vector_db_from_csv(file_path)
                st.session_state.rag_chain = create_rag_chain(vector_store)
        
        if st.session_state.rag_chain:
            st.success("✅ RAG 問答功能已啟用！")

        st.subheader("🖼️ 圖片分析")
        uploaded_image = st.file_uploader("上傳圖片進行分析", type=["png", "jpg", "jpeg"])
        if uploaded_image: add_user_image_to_main_chat(uploaded_image)
        st.divider()

        if st.button("🗑️ 清除所有對話與資料"):
            keys_to_clear = [k for k in st.session_state.keys() if k != 'gemini_api_key_input']
            for key in keys_to_clear:
                st.session_state.pop(key)
            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

        st.session_state.debug_mode = st.checkbox("啟用偵錯模式", value=st.session_state.debug_mode)

    # --- 主工作區 (標籤頁面) ---
    tab_titles = ["💬 主要聊天室", "💼 高管工作流"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    # --- 主要聊天室 ---
    with tabs[0]:
        st.header("💬 主要聊天室")
        st.caption("可進行 RAG 資料問答、圖片分析、或一般對話 (均具備 Session 記憶)。")
        
        MAIN_CHAT_SESSION_ID = "main_chat_session"
        history = get_session_history(MAIN_CHAT_SESSION_ID)

        for msg in history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)
        
        if user_input := st.chat_input("請對數據或圖片提問..."):
            with st.chat_message("human"):
                st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("Gemini 正在思考中..."):
                    response = ""
                    # 情境1：RAG 問答
                    if st.session_state.rag_chain:
                        response = st.session_state.rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": MAIN_CHAT_SESSION_ID}}
                        )
                    # 情境2：圖片問答
                    elif st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(
                            user_input, st.session_state.pending_image_for_main_gemini
                        )
                        history.add_user_message(user_input)
                        history.add_ai_message(response)
                    # 情境3：一般聊天
                    else:
                        if "general_chat_chain" not in st.session_state:
                            st.session_state.general_chat_chain = create_generic_chat_chain("你是一個樂於助人的 AI 助理。")
                        response = st.session_state.general_chat_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": MAIN_CHAT_SESSION_ID}}
                        )
                    st.markdown(response)

    # --- 高管工作流標籤 ---
    with tabs[1]:
        st.header("💼 高管工作流 (由 Gemini Pro 驅動)")
        st.write("請先在側邊欄上傳CSV資料，然後在此輸入商業問題，最後點擊按鈕啟動分析。")
        st.session_state.executive_user_query = st.text_area(
            "請輸入商業問題以啟動分析:", value=st.session_state.get("executive_user_query", ""), height=100
        )
        can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
        
        if st.button("🚀 啟動/重啟高管分析", disabled=not can_start):
             st.session_state.executive_workflow_stage = "data_profiling_pending"
             st.session_state.cfo_analysis_text = ""
             st.session_state.coo_analysis_text = ""
             st.session_state.ceo_summary_text = ""
             st.rerun()
        
        if st.session_state.executive_workflow_stage == "data_profiling_pending":
             with st.spinner("正在生成資料摘要..."):
                df = pd.read_csv(st.session_state.uploaded_file_path)
                st.session_state.executive_data_profile_str = generate_data_profile(df)
                st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                st.rerun()

        if st.session_state.get('executive_data_profile_str'):
            with st.expander("查看資料摘要"):
                st.text(st.session_state.executive_data_profile_str)
        
        # --- CFO/COO/CEO 執行邏輯 ---
        # (此處省略了具體的 CFO/COO/CEO 呼叫邏輯以保持簡潔，但它們在您的完整檔案中應該被保留)

    # --- 其他 AI 角色標籤 ---
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + 2]:
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"])
            
            if f"{role_id}_chain" not in st.session_state:
                st.session_state[f"{role_id}_chain"] = create_generic_chat_chain(role_info["system_prompt"])
            
            chain = st.session_state[f"{role_id}_chain"]
            session_id = role_info["session_id"]
            history = get_session_history(session_id)
            
            for msg in history.messages:
                with st.chat_message(msg.type):
                    st.markdown(msg.content)
            
            if user_input := st.chat_input(f"與 {role_info['name']} 對話..."):
                with st.chat_message("human"):
                    st.markdown(user_input)
                with st.chat_message("ai"):
                    with st.spinner("正在生成回應..."):
                        response = chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        st.markdown(response)

if __name__ == "__main__":
    main()
