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
import time
import matplotlib.font_manager as fm
import matplotlib
import numpy as np

# --- 基礎 API 和資料處理套件 ---
import google.generativeai as genai
from openai import OpenAI
import faiss

# --- 只在 RAG 建立/檢索時使用的 LangChain 元件 ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

# --- 初始化與常數定義 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

ROLE_DEFINITIONS = {
    "summarizer": { "name": "📝 摘要專家", "system_prompt": "你是一位專業的摘要專家...", "session_id": "summarizer_chat" },
    "creative_writer": { "name": "✍️ 創意作家", "system_prompt": "你是一位充滿想像力的創意作家...", "session_id": "creative_writer_chat" }
}

# --- 基礎輔助函數 ---
def save_uploaded_file(uploaded_file):
    # ... (此函式保持不變)
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    # ... (此函式保持不變)
    try:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.pending_image_for_main_gemini = Image.open(file_path)
        st.image(st.session_state.pending_image_for_main_gemini, caption="圖片已上傳...", use_container_width=True)
    except Exception as e: st.error(f"處理上傳圖片時出錯: {e}")

# --- 混合架構 RAG 核心函式 ---
@st.cache_resource
def create_lc_retriever(file_path: str, openai_api_key: str):
    """(使用 LangChain) 從 CSV 建立一個僅用於「檢索」的工具"""
    with st.status("正在使用 LangChain 建立知識庫...", expanded=True) as status:
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
            # 我們只返回一個可以接收問題並返回文件的檢索器
            return vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            st.error(f"建立知識庫過程中發生錯誤: {e}")
            status.update(label="建立失敗", state="error")
            return None

# --- Gemini API 相關函式 (手動控制) ---
def get_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    # 將我們的 history 格式轉換為 gemini 的格式
    gemini_history = []
    for msg in history:
        # Gemini API 期望的 role 是 'user' 和 'model'
        role = "user" if msg["role"] == "human" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = client.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt)
    return response.text

# --- 主應用入口 ---
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理 (混合架構版)")

    # --- 初始化 Session States ---
    if "retriever_chain" not in st.session_state: st.session_state.retriever_chain = None
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "last_uploaded_filename" not in st.session_state: st.session_state.last_uploaded_filename = None
    if "chat_histories" not in st.session_state: st.session_state.chat_histories = {}
    # ... 其他需要的 session state ...

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ API 金鑰設定")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        st.text_input("請輸入您的 OpenAI API Key", type="password", key="openai_api_key_input")
        st.caption("Gemini 用於聊天，OpenAI 用於 RAG 資料嵌入。")
        st.divider()

        st.subheader("📁 資料問答 (RAG)")
        uploaded_file = st.file_uploader("上傳 CSV 以啟用 RAG 問答功能", type=["csv"])
        
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    st.error("請在側邊欄設定您的 OpenAI API Key！")
                else:
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    file_path = save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_file_path = file_path
                    # 建立並儲存 LangChain 檢索器
                    st.session_state.retriever_chain = create_lc_retriever(file_path, openai_api_key)
        
        if st.session_state.retriever_chain: st.success("✅ RAG 檢索功能已啟用！")
        
        st.divider()
        if st.button("🗑️ 清除所有對話與資料"):
            st.session_state.clear()
            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

    # --- 主工作區 (標籤頁面) ---
    tab_titles = ["💬 主要聊天室"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    # --- API Key 檢查 ---
    gemini_api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.warning("請在側邊欄輸入您的 Google Gemini API Key 以啟動聊天功能。")
        st.stop()
    gemini_client = get_gemini_client(gemini_api_key)

    # --- 主要聊天室 (混合模式) ---
    with tabs[0]:
        st.header("💬 主要聊天室")
        session_id = "main_chat"
        if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
        
        # 顯示歷史訊息
        for msg in st.session_state.chat_histories[session_id]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if user_input := st.chat_input("請對數據提問或開始對話..."):
            st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
            with st.chat_message("human"): st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("正在思考中..."):
                    response = ""
                    # --- 混合邏輯開始 ---
                    # 如果 RAG 檢索器已啟用，則執行 RAG 流程
                    if st.session_state.retriever_chain:
                        # 1. 使用 LangChain 檢索上下文
                        retrieved_docs = st.session_state.retriever_chain.invoke(user_input)
                        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                        
                        # 2. 手動組合 Prompt
                        prompt = f"""
                        請根據以下提供的「上下文」來回答問題。請只使用上下文中的資訊。

                        [上下文]:
                        {context}

                        [問題]:
                        {user_input}

                        [回答]:
                        """
                        # 3. 手動呼叫 Gemini API
                        response = gemini_client.generate_content(prompt).text
                    # 否則，執行一般聊天
                    else:
                        history = st.session_state.chat_histories[session_id][:-1]
                        response = get_gemini_response_with_history(gemini_client, history, user_input)
                    
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    # --- 其他 AI 角色標籤 (手動模式) ---
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + 1]: # 注意索引從 1 開始
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"])
            session_id = role_info["session_id"]
            if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []

            # 顯示歷史訊息
            for msg in st.session_state.chat_histories[session_id]:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            if user_input := st.chat_input(f"與 {role_info['name']} 對話..."):
                st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
                with st.chat_message("human"): st.markdown(user_input)
                
                with st.chat_message("ai"):
                    with st.spinner("正在生成回應..."):
                        # 為角色專家加上 system_prompt
                        client_with_prompt = get_gemini_client(gemini_api_key)
                        client_with_prompt.system_instruction = role_info["system_prompt"]
                        
                        history = st.session_state.chat_histories[session_id][:-1]
                        response = get_gemini_response_with_history(client_with_prompt, history, user_input)
                        
                        st.markdown(response)
                        st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()
