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

# --- 初始化與常數定義 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

ROLE_DEFINITIONS = {
    "summarizer": { "name": "📝 摘要專家", "system_prompt": "你是一位專業的摘要專家...", "session_id": "summarizer_chat" },
    "creative_writer": { "name": "✍️ 創意作家", "system_prompt": "你是一位充滿想像力的創意作家...", "session_id": "creative_writer_chat" }
}

# --- 基礎輔助函數 (幾乎不變) ---
def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return file_path

def add_user_image_to_main_chat(uploaded_file):
    try:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.pending_image_for_main_gemini = Image.open(file_path)
        st.image(st.session_state.pending_image_for_main_gemini, caption="圖片已上傳...", use_container_width=True)
    except Exception as e: st.error(f"處理上傳圖片時出錯: {e}")

# --- 從零打造 RAG 核心函式 ---

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """簡單的文本切割函式"""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def get_openai_embeddings(texts: list, client: OpenAI, model="text-embedding-3-small"):
    """使用 OpenAI API 獲取嵌入向量"""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

@st.cache_resource
def create_knowledge_base(file_path: str, openai_api_key: str):
    """從 CSV 檔案建立一個 FAISS 向量知識庫 (無 LangChain)"""
    with st.status("正在建立知識庫...", expanded=True) as status:
        try:
            status.update(label="步驟 1/3：正在讀取與切割文件...")
            df = pd.read_csv(file_path, encoding='utf-8')
            df.dropna(how='all', inplace=True)
            df.fillna("", inplace=True)
            # 將所有欄位合併為一個長文本，並為每行建立一個 document
            documents = df.apply(lambda row: ' | '.join(row.astype(str)), axis=1).tolist()
            
            all_chunks = []
            for doc in documents:
                all_chunks.extend(split_text(doc))
            status.update(label=f"步驟 1/3 完成！已切割成 {len(all_chunks)} 個文本區塊。")

            status.update(label="步驟 2/3：正在呼叫 OpenAI API 生成向量嵌入...")
            client = OpenAI(api_key=openai_api_key)
            embeddings = get_openai_embeddings(all_chunks, client)
            status.update(label="步驟 2/3 完成！向量嵌入已生成。")

            status.update(label="步驟 3/3：正在建立 FAISS 索引...")
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            status.update(label="知識庫已就緒！", state="complete", expanded=False)

            return {"index": index, "chunks": all_chunks, "client": client}
        except Exception as e:
            st.error(f"建立知識庫過程中發生錯誤: {e}")
            status.update(label="建立失敗", state="error")
            return None

def get_rag_response(query: str, kb: dict, gemini_client):
    """執行 RAG 查詢"""
    # 1. 檢索 (Retrieve)
    client = kb["client"]
    query_embedding = get_openai_embeddings([query], client)[0]
    D, I = kb["index"].search(np.array([query_embedding]).astype('float32'), k=5)
    retrieved_chunks = [kb["chunks"][i] for i in I[0]]
    context = "\n---\n".join(retrieved_chunks)

    # 2. 增強 (Augment) & 3. 生成 (Generate)
    prompt = f"""
    請根據以下提供的「上下文」來回答問題。請只使用上下文中的資訊。

    [上下文]:
    {context}

    [問題]:
    {query}

    [回答]:
    """
    response = gemini_client.generate_content(prompt)
    return response.text

# --- Gemini API 相關函式 ---
def get_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    # 將我們的 history 格式轉換為 gemini 的格式
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "human" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = client.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt)
    return response.text

# --- 主應用入口 ---
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理 (無 LangChain 版)")

    # --- 初始化 Session States ---
    if "knowledge_base" not in st.session_state: st.session_state.knowledge_base = None
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "last_uploaded_filename" not in st.session_state: st.session_state.last_uploaded_filename = None
    if "pending_image_for_main_gemini" not in st.session_state: st.session_state.pending_image_for_main_gemini = None
    if "chat_histories" not in st.session_state: st.session_state.chat_histories = {}

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ API 金鑰設定")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        st.text_input("請輸入您的 OpenAI API Key", type="password", key="openai_api_key_input")
        st.caption("本應用使用 Gemini 進行聊天，使用 OpenAI 進行資料嵌入。")
        st.divider()

        st.subheader("📁 資料問答 (RAG)")
        uploaded_file = st.file_uploader("上傳 CSV 以啟用 RAG 問答功能", type=["csv"])
        
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    st.error("請在側邊欄或 Secrets 中設定您的 OpenAI API Key！")
                else:
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    file_path = save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.knowledge_base = create_knowledge_base(file_path, openai_api_key)
        
        if st.session_state.knowledge_base: st.success("✅ RAG 問答功能已啟用！")
        
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

    # --- 主要聊天室 ---
    with tabs[0]:
        st.header("💬 主要聊天室")
        session_id = "main_chat"
        if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
        
        # 顯示歷史訊息
        for msg in st.session_state.chat_histories[session_id]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if user_input := st.chat_input("請對數據或圖片提問..."):
            st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
            with st.chat_message("human"): st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("正在思考中..."):
                    response = ""
                    # RAG 問答
                    if st.session_state.knowledge_base:
                        response = get_rag_response(user_input, st.session_state.knowledge_base, gemini_client)
                    # 一般聊天
                    else:
                        history = st.session_state.chat_histories[session_id][:-1] # 不包含當前問題
                        response = get_gemini_response_with_history(gemini_client, history, user_input)
                    
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    # --- 其他 AI 角色標籤 ---
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + 2]:
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
