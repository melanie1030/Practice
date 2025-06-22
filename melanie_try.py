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
# --- RAG 相關套件暫時保留，但不一定會被使用 (2025/06/22) ---
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

# --- RAG 核心函式 - 暫時保留程式碼以備未來恢復 (2025/06/22) ---
@st.cache_resource
def create_lc_retriever(file_path: str, openai_api_key: str):
    # 此函數為 RAG 功能的核心，目前已暫停呼叫。
    # 未來若要恢復 RAG，只需在側邊欄重新啟用對此函數的呼叫即可。
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
            return vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            st.error(f"建立知識庫過程中發生嚴重錯誤: {e}")
            status.update(label="建立失敗", state="error")
            return None

# --- Gemini API 相關函式 (手動控制) ---
def get_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

def get_gemini_response_with_history(client, history, user_prompt):
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "human" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
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
    except Exception as e:
        st.error(f"Gemini 圖片分析請求失敗: {e}")
        return f"錯誤: {e}"

def get_gemini_executive_analysis(api_key, executive_role_name, full_prompt):
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
    st.title("✨ Gemini 多功能 AI 助理 ")

    # --- 初始化 Session States ---
    keys_to_init = {
        "retriever_chain": None, "uploaded_file_path": None, "last_uploaded_filename": None,
        "pending_image_for_main_gemini": None, "chat_histories": {},
        "executive_workflow_stage": "idle", "executive_user_query": "",
        "executive_data_profile_str": "", "executive_rag_context": "", "cfo_analysis_text": "",
        "coo_analysis_text": "", "ceo_summary_text": "",
        "sp_workflow_stage": "idle", "sp_user_query": "",
        "sp_final_report": ""
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ API 金鑰設定")
        st.text_input("請輸入您的 Google Gemini API Key", type="password", key="gemini_api_key_input")
        
        # --- RAG 相關 UI 暫時停用 (2025/06/22) ---
        # st.text_input("請輸入您的 OpenAI API Key", type="password", key="openai_api_key_input")
        # st.caption("Gemini 用於聊天，OpenAI 用於 RAG 資料嵌入。")
        st.caption("Gemini 用於所有 AI 分析功能。")
        st.divider()

        # --- RAG 功能修改：僅保留檔案上傳與路徑保存，移除 RAG 建立 (2025/06/22) ---
        st.subheader("📁 上傳資料進行分析")
        st.info("RAG 知識庫功能已暫時停用以節省成本。目前僅支援上傳 CSV 進行資料摘要分析。")
        uploaded_file = st.file_uploader("上傳 CSV 以啟用工作流分析", type=["csv"])
        
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename"):
                st.session_state.last_uploaded_filename = uploaded_file.name
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                st.success(f"檔案 '{uploaded_file.name}' 上傳成功！")
                # 以下為停用的 RAG 建立過程
                # openai_api_key = st.session_state.get("openai_api_key_input") or os.environ.get("OPENAI_API_KEY")
                # if not openai_api_key:
                #     st.error("請在側邊欄設定您的 OpenAI API Key！")
                # else:
                #     st.session_state.retriever_chain = create_lc_retriever(file_path, openai_api_key)
        
        # if st.session_state.retriever_chain: st.success("✅ RAG 檢索功能已啟用！")

        st.subheader("🖼️ 圖片分析")
        uploaded_image = st.file_uploader("上傳圖片進行分析", type=["png", "jpg", "jpeg"])
        if uploaded_image: add_user_image_to_main_chat(uploaded_image)
        
        st.divider()
        if st.button("🗑️ 清除所有對話與資料"):
            api_keys = {
                'gemini_api_key_input': st.session_state.get('gemini_api_key_input'),
                # 'openai_api_key_input': st.session_state.get('openai_api_key_input')
            }
            st.session_state.clear()
            for key, value in api_keys.items():
                if value: st.session_state[key] = value

            st.cache_resource.clear()
            st.success("所有對話、Session 記憶和快取已清除！")
            st.rerun()

    # --- 主工作區 (標籤頁面) ---
    tab_titles = ["💬 主要聊天室", "💼 階段式工作流", "🧩 單一提示整合工作流"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    # --- API Key 檢查 ---
    gemini_api_key = st.session_state.get("gemini_api_key_input") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.warning("請在側邊欄輸入您的 Google Gemini API Key 以啟動聊天功能。")
        st.stop()
    gemini_client = get_gemini_client(gemini_api_key)

    # --- 主要聊天室 (tabs[0]) ---
    with tabs[0]:
        st.header("💬 主要聊天室")
        st.caption("可進行圖片分析或一般對話。RAG 資料問答已暫停。")
        session_id = "main_chat"
        if session_id not in st.session_state.chat_histories: st.session_state.chat_histories[session_id] = []
        
        for msg in st.session_state.chat_histories[session_id]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if user_input := st.chat_input("請對圖片提問或開始對話..."):
            st.session_state.chat_histories[session_id].append({"role": "human", "content": user_input})
            with st.chat_message("human"): st.markdown(user_input)
            
            with st.chat_message("ai"):
                with st.spinner("正在思考中..."):
                    response = ""
                    # --- RAG 邏輯暫時停用 (2025/06/22) ---
                    # if st.session_state.retriever_chain:
                    #     retrieved_docs = st.session_state.retriever_chain.invoke(user_input)
                    #     context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                    #     prompt = f"請根據上下文回答問題。\n[上下文]:\n{context}\n\n[問題]:\n{user_input}\n\n[回答]:"
                    #     response = gemini_client.generate_content(prompt).text
                    if st.session_state.pending_image_for_main_gemini:
                        response = get_gemini_response_for_image(gemini_api_key, user_input, st.session_state.pending_image_for_main_gemini)
                    else:
                        history = st.session_state.chat_histories[session_id][:-1]
                        response = get_gemini_response_with_history(gemini_client, history, user_input)
                    
                    st.markdown(response)
                    st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

    # --- 階段式高管工作流 (tabs[1]) ---
    with tabs[1]:
        st.header("💼 階段式工作流")
        st.info("**方法說明**：此流程模擬三位獨立的專家。每一步都是一次獨立的 API 請求。分析僅基於上傳 CSV 的**統計摘要**。")
        
        st.session_state.executive_user_query = st.text_area(
            "請輸入商業問題以啟動分析:", value=st.session_state.get("executive_user_query", ""), height=100, key="original_workflow_query"
        )
        # --- RAG 依賴移除 (2025/06/22) ---
        can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
        
        if st.button("🚀 啟動階段式分析", disabled=not can_start, key="exec_flow_button"):
            st.session_state.executive_workflow_stage = "cfo_analysis_pending"
            st.session_state.cfo_analysis_text = ""
            st.session_state.coo_analysis_text = ""
            st.session_state.ceo_summary_text = ""
            st.rerun()

        if st.session_state.executive_workflow_stage == "cfo_analysis_pending":
            with st.spinner("CFO 正在獨立分析..."):
                df = pd.read_csv(st.session_state.uploaded_file_path)
                data_profile = generate_data_profile(df)
                st.session_state.executive_data_profile_str = data_profile
                
                # --- RAG 相關程式碼停用 (2025/06/22) ---
                # retriever = st.session_state.retriever_chain
                query = st.session_state.executive_user_query
                # retrieved_docs = retriever.invoke(query)
                # rag_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                # st.session_state.executive_rag_context = rag_context

                # --- Prompt 修改，移除 RAG 內容 (2025/06/22) ---
                cfo_prompt = f"作為財務長(CFO)，請基於你的專業知識，並嚴格參考以下提供的「統計摘要」，為商業問題提供財務角度的簡潔分析。\n\n[商業問題]:\n{query}\n\n[統計摘要]:\n{data_profile}"
                response = get_gemini_executive_analysis(gemini_api_key, "CFO", cfo_prompt)
                st.session_state.cfo_analysis_text = response
                st.session_state.executive_workflow_stage = "coo_analysis_pending"
                st.rerun()
        
        if st.session_state.get('executive_data_profile_str'):
            with st.expander("查看統計摘要"): st.text(st.session_state.executive_data_profile_str)
        # --- RAG 相關 UI 停用 (2025/06/22) ---
        # if st.session_state.get('executive_rag_context'):
        #     with st.expander("查看 RAG 檢索出的相關資料"): st.markdown(st.session_state.executive_rag_context)

        if st.session_state.cfo_analysis_text:
            st.subheader("📊 財務長 (CFO) 分析")
            st.markdown(st.session_state.cfo_analysis_text)
        
        if st.session_state.executive_workflow_stage == "coo_analysis_pending":
            with st.spinner("COO 正在基於 CFO 報告進行分析..."):
                coo_prompt = f"作為營運長(COO)，請基於商業問題、統計摘要以及下方 CFO 的分析，提供營運層面的策略與潛在風險。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[統計摘要]:\n{st.session_state.executive_data_profile_str}"
                response = get_gemini_executive_analysis(gemini_api_key, "COO", coo_prompt)
                st.session_state.coo_analysis_text = response
                st.session_state.executive_workflow_stage = "ceo_summary_pending"
                st.rerun()

        if st.session_state.coo_analysis_text:
            st.subheader("🏭 營運長 (COO) 分析")
            st.markdown(st.session_state.coo_analysis_text)

        if st.session_state.executive_workflow_stage == "ceo_summary_pending":
            with st.spinner("CEO 正在整合所有報告進行總結..."):
                ceo_prompt = f"作為執行長(CEO)，請整合所有資訊，包含原始問題、CFO 的財務分析、COO 的營運分析，提供一個高層次的、可執行的決策總結與明確的行動建議。\n\n[商業問題]:\n{st.session_state.executive_user_query}\n\n[CFO 的財務分析]:\n{st.session_state.cfo_analysis_text}\n\n[COO 的營運分析]:\n{st.session_state.coo_analysis_text}"
                response = get_gemini_executive_analysis(gemini_api_key, "CEO", ceo_prompt)
                st.session_state.ceo_summary_text = response
                st.session_state.executive_workflow_stage = "completed"
                st.rerun()

        if st.session_state.ceo_summary_text:
            st.subheader("👑 執行長 (CEO) 最終決策")
            st.markdown(st.session_state.ceo_summary_text)

    # --- 單一提示整合工作流 (tabs[2]) ---
    with tabs[2]:
        st.header("🧩 單一提示整合工作流")
        st.info("**方法說明**：此流程模擬一個全能的 AI 高管團隊。分析僅基於上傳 CSV 的**統計摘要**。")
        
        st.session_state.sp_user_query = st.text_area(
             "請輸入商業問題以啟動分析:", value=st.session_state.get("sp_user_query", ""), height=100, key="sp_workflow_query"
        )
        # --- RAG 依賴移除 (2025/06/22) ---
        can_start_sp = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("sp_user_query"))

        if st.button("🚀 啟動單一提示整合分析", disabled=not can_start_sp, key="sp_flow_button"):
            st.session_state.sp_workflow_stage = "running"
            st.session_state.sp_final_report = ""
            
            with st.spinner("AI 高管團隊正在進行全面分析..."):
                df = pd.read_csv(st.session_state.uploaded_file_path)
                data_profile = generate_data_profile(df)
                query = st.session_state.sp_user_query
                
                # --- Prompt 修改，移除 RAG 內容 (2025/06/22) ---
                mega_prompt = f"""
你是一個頂尖的 AI 商業分析團隊，能夠在一次思考中扮演多個高管角色。你的任務是針對給定的商業問題和數據統計摘要，生成一份包含三個部分的完整分析報告。

請嚴格按照以下結構和要求進行輸出，使用 Markdown 標題來區分每個部分：
---
### 📊 財務長 (CFO) 分析
在此部分，請完全以財務長的角度思考。專注於財務指標、成本效益、投資回報率、毛利率、潛在的財務風險等。你的分析必須完全基於提供的統計摘要。

### 🏭 營運長 (COO) 分析
在此部分，轉換為營運長的角色。你需要思考，在CFO會提出的財務考量下，營運上是否可行？分析潛在的流程、供應鏈、人力資源或執行風險。

### 👑 執行長 (CEO) 最終決策
在此部分，作為CEO，請綜合上述的財務(CFO)和營運(COO)分析。不要重複細節，而是提供一個高層次的戰略總結。最終，給出一個明確、果斷的**決策**，並列出 2-3 個最重要的**後續行動建議**。
---
現在，請根據以下資訊開始分析：

**[商業問題]:**
{query}

**[資料統計摘要]:**
{data_profile}
"""
                response = get_gemini_executive_analysis(gemini_api_key, "IntegratedTeam", mega_prompt)
                st.session_state.sp_final_report = response
                st.session_state.sp_workflow_stage = "completed"
                st.rerun()

        if st.session_state.sp_workflow_stage == "completed" and st.session_state.sp_final_report:
            st.subheader("📈 AI 高管團隊整合報告")
            st.markdown(st.session_state.sp_final_report)

    # --- 其他 AI 角色標籤 (tabs[3] 和 tabs[4]) ---
    role_tab_offset = 3 
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
                        client_with_prompt = get_gemini_client(gemini_api_key)
                        try: client_with_prompt.system_instruction = role_info["system_prompt"]
                        except: pass
                        history = st.session_state.chat_histories[session_id][:-1]
                        final_prompt = f"{role_info['system_prompt']}\n\n{user_input}"
                        response = get_gemini_response_with_history(client_with_prompt, history, final_prompt)
                        st.markdown(response)
                        st.session_state.chat_histories[session_id].append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()
