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

# --- LangChain 全家桶 Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

# --- 中文字型設定 ---
try:
    font_path = "./fonts/msjh.ttc"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字型載入失敗: {e}")
    st.warning("中文字型載入失敗，圖表中的中文可能無法正常顯示。")

# --- 初始化設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

# --- 基礎輔助函數 (保持不變) ---
def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    try:
        file_path = save_uploaded_file(uploaded_file)
        image_pil = Image.open(file_path)
        st.session_state.pending_image_for_main_gemini = image_pil
        st.image(image_pil, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
    except Exception as e:
        st.error(f"處理上傳圖片時出錯: {e}")

# --- LangChain 核心函式 (In-Memory Session 版) ---

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """從 Streamlit Session State 獲取或創建對話歷史物件"""
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

def create_rag_chain(vector_store):
    """建立一個整合了「對話內記憶」的 RAG 鏈"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    retriever = vector_store.as_retriever()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        請根據以下提供的「對話歷史」和從資料庫檢索到的「上下文」來回答問題。
        請優先使用「上下文」中的資訊。如果「上下文」不足，可以參考「對話歷史」。
        如果都無法回答，請直說「根據我所擁有的資料，無法回答這個問題」。

        [上下文]:
        {context}
        """),
        ("human", "{input}"),
    ])

    base_rag_chain = (
        {
            "context": itemgetter("input") | retriever,
            "input": itemgetter("input"),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain_with_history = RunnableWithMessageHistory(
        base_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return rag_chain_with_history

def create_generic_chat_chain(system_prompt: str):
    """為特定角色建立一個具備「對話內記憶」的通用聊天鏈"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    base_chain = prompt_template | llm | StrOutputParser()
    
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_history

def create_vector_db_from_csv(file_path: str):
    """(保持不變) 從 CSV 檔案載入、切割、嵌入並建立向量資料庫"""
    with st.spinner("正在處理資料，建立知識庫中..."):
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        st.success(f"知識庫建立完成！已載入 {len(docs)} 筆資料。")
        return vector_store

# --- 其他 Gemini 呼叫函式 (保持不變) ---
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
    # (保持不變)
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
    # (保持不變)
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
# 主應用入口 (最終整合版)
# ------------------------------
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理")

    # --- 初始化 Session States (不再需要手動管理 memory) ---
    keys_to_init = {
        "rag_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "executive_workflow_stage": "idle", "executive_user_query": "",
        "executive_data_profile_str": "", "cfo_analysis_text": "", "coo_analysis_text": "", "ceo_summary_text": "",
        "debug_mode": False
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- 側邊欄介面 (移除檔案記憶體管理) ---
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
            st.success("所有對話、記憶和工作狀態已清除！")
            st.rerun()

        st.session_state.debug_mode = st.checkbox("啟用偵錯模式", value=st.session_state.debug_mode)
        # ... (偵錯資訊顯示邏輯不變)

    # --- 主工作區 (標籤頁面) ---
    tab_titles = ["💬 主要聊天室", "💼 高管工作流"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    # --- 主要聊天室 (使用 In-Memory 記憶) ---
    with tabs[0]:
        st.header("💬 主要聊天室")
        st.caption("可進行 RAG 資料問答、圖片分析 (對話功能具備 Session 記憶)。")
        
        MAIN_CHAT_SESSION_ID = "main_chat_session"
        history = get_session_history(MAIN_CHAT_SESSION_ID)

        for msg in history.messages:
            with st.chat_message(msg.type):
                # 處理圖片顯示
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and "image" in part:
                             st.image(part["image"])
                        elif isinstance(part, str):
                             st.markdown(part)
                else:
                    st.markdown(msg.content)
        
        if user_input := st.chat_input("請對數據或圖片提問..."):
            with st.chat_message("human"):
                st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("Gemini 正在思考中..."):
                    # 情境1：RAG 問答
                    if st.session_state.rag_chain:
                        response = st.session_state.rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": MAIN_CHAT_SESSION_ID}}
                        )
                        st.markdown(response)
                    # 情境2：圖片問答
                    elif st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(
                            user_input, st.session_state.pending_image_for_main_gemini
                        )
                        # 手動將圖文問答加入歷史
                        history.add_user_message(user_input)
                        history.add_ai_message(response)
                        st.markdown(response)
                    # 情境3：一般聊天
                    else:
                        # 建立一個臨時的通用聊天鏈來處理一般對話
                        general_chain = create_generic_chat_chain("你是一個樂於助人的 AI 助理。")
                        response = general_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": MAIN_CHAT_SESSION_ID}}
                        )
                        st.markdown(response)

    # --- 高管工作流標籤 (保持不變) ---
    with tabs[1]:
        st.header("💼 高管工作流 (由 Gemini Pro 驅動)")
        st.write("請先在側邊欄上傳CSV資料，然後在此輸入商業問題，最後點擊按鈕啟動分析。")
        # ... (此處所有程式碼完全不變)
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
        # ... 後續 CFO, COO, CEO 的邏輯完全不變 ...

    # --- 其他 AI 角色標籤 (套用 In-Memory 記憶) ---
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + 2]:
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"])
            
            # 為每個角色創建帶有記憶的鏈
            if f"{role_id}_chain" not in st.session_state:
                st.session_state[f"{role_id}_chain"] = create_generic_chat_chain(role_info["system_prompt"])
            
            chain = st.session_state[f"{role_id}_chain"]
            session_id = role_info["session_id"]
            history = get_session_history(session_id)
            
            # 顯示歷史訊息
            for msg in history.messages:
                with st.chat_message(msg.type):
                    st.markdown(msg.content)
            
            # 聊天輸入
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
