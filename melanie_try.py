import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO, StringIO
from PIL import Image
import google.generativeai as genai
import time
import matplotlib.font_manager as fm
import matplotlib
import sys

# --- LangChain and Gemini Imports ---
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Role & Workflow Definitions (Now simplified and Gemini-focused) ---
ROLE_DEFINITIONS = {
    "summarizer": {
        "name": "📝 摘要專家",
        "system_prompt": "你是一位專業的摘要專家。你的任務是將提供的任何文本或對話，濃縮成清晰、簡潔的繁體中文摘要。專注於要點和關鍵結論。",
        "messages_key": "summarizer_messages",
        "chat_session_key": "summarizer_chat_session",
    },
    "creative_writer": {
        "name": "✍️ 創意作家",
        "system_prompt": "你是一位充滿想像力的創意作家。你的任務是幫助使用者完成創意寫作，例如寫故事、詩歌、劇本或腦力激盪，全部使用繁體中文。",
        "messages_key": "creative_writer_messages",
        "chat_session_key": "creative_writer_chat_session",
    }
}

EXECUTIVE_ROLE_IDS = {
    "CFO": "cfo_exec",
    "COO": "coo_exec",
    "CEO": "ceo_exec",
}


# --- 中文字型設定 ---
try:
    font_path = "./fonts/msjh.ttc"
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字型載入失敗: {e}")
    st.warning(f"中文字型載入失敗，圖表中的中文可能無法正常顯示。請確認字型路徑 '{font_path}' 是否正確。")

# --- 初始化設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MAX_MESSAGES_PER_STREAM = 12

# --- 基礎輔助函數 ---
def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_logs" not in st.session_state: st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"**LOG ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_errors" not in st.session_state: st.session_state.debug_errors = []
        st.session_state.debug_errors.append(f"**ERROR ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG ERROR: {msg}")

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def append_message_to_stream(message_stream_key, role, content):
    if message_stream_key not in st.session_state: st.session_state[message_stream_key] = []
    st.session_state[message_stream_key].append({"role": role, "content": content})
    if len(st.session_state[message_stream_key]) > MAX_MESSAGES_PER_STREAM:
        st.session_state[message_stream_key] = st.session_state[message_stream_key][-MAX_MESSAGES_PER_STREAM:]

def add_user_image_to_main_chat(uploaded_file):
    try:
        file_path = save_uploaded_file(uploaded_file)
        image_pil = Image.open(file_path)
        # For a Gemini-only app, we always handle it this way
        st.session_state.pending_image_for_main_gemini = image_pil
        st.image(image_pil, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
        debug_log(f"圖片已暫存，待與文字一同發送 (Gemini): {file_path}.")
    except Exception as e:
        st.error(f"處理上傳圖片時出錯: {e}")
        debug_error(f"Error in add_user_image_to_main_chat: {e}, Traceback: {traceback.format_exc()}")

# --- Gemini Pandas Agent 核心函數 ---
def create_pandas_agent(file_path: str):
    gemini_api_key = st.session_state.get("gemini_api_key_input") or st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("建立 Gemini 資料分析代理需要 API Key。")
        return None
    try:
        df = pd.read_csv(file_path)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_api_key, convert_system_message_to_human=True)
        agent = create_pandas_dataframe_agent(llm, df, verbose=st.session_state.get("debug_mode", False), handle_parsing_errors=True, allow_dangerous_code=True)
        return agent
    except Exception as e:
        st.error(f"建立資料分析代理時發生錯誤: {e}")
        return None

def query_pandas_agent(agent, query: str):
    if not agent: return "錯誤：資料分析代理未初始化。"
    prompt = f"請針對以下問題進行分析，並用繁體中文回答：\n問題: \"{query}\""
    try:
        response = agent.invoke({"input": prompt})
        return response.get("output", "代理沒有提供有效的輸出。")
    except Exception as e:
        st.error(f"代理在處理您的請求時發生錯誤: {e}")
        return f"代理執行時出錯: {e}"

# --- Gemini 通用聊天函數 ---
def get_gemini_response_main_chat(user_prompt, image_pil=None):
    api_key = st.session_state.get("gemini_api_key_input") or st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        content_parts = []
        if image_pil:
            content_parts.append(image_pil)
        content_parts.append(user_prompt)
        response = model.generate_content(content_parts)
        # Clear the pending image after use
        if "pending_image_for_main_gemini" in st.session_state:
            st.session_state.pending_image_for_main_gemini = None
        return response.text
    except Exception as e:
        st.error(f"Gemini API 請求失敗: {e}")
        return f"錯誤: {e}"

def get_gemini_response_for_generic_role(role_id, user_input_text):
    api_key = st.session_state.get("gemini_api_key_input") or st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "錯誤：未設定 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        role_info = ROLE_DEFINITIONS[role_id]
        model = genai.GenerativeModel("gemini-1.5-flash-latest", system_instruction=role_info["system_prompt"])
        response = model.generate_content(user_input_text)
        return response.text
    except Exception as e:
        st.error(f"'{role_info['name']}' 角色執行時出錯: {e}")
        return f"錯誤: {e}"

# --- Gemini 高管工作流函數 ---
def get_gemini_executive_analysis(executive_role_name, full_prompt):
    api_key = st.session_state.get("gemini_api_key_input") or st.secrets.get("GEMINI_API_KEY")
    if not api_key: return f"錯誤：高管工作流 ({executive_role_name}) 未能獲取 Gemini API Key。"
    try:
        genai.configure(api_key=api_key)
        # Use a more powerful model for executive reasoning
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"高管分析 ({executive_role_name}) 失敗: {e}")
        return f"錯誤: {e}"

# --- 資料摘要函數 ---
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
# 主應用入口 (最終 Gemini 整合版)
# ------------------------------
def main():
    st.set_page_config(page_title="Gemini Multi-Function Bot", page_icon="✨", layout="wide")
    st.title("✨ Gemini 多功能 AI 助理")

    # --- 初始化所有需要的 Session States ---
    # 主聊天室與代理
    if "messages" not in st.session_state: st.session_state.messages = []
    if "pandas_agent" not in st.session_state: st.session_state.pandas_agent = None
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "last_uploaded_filename" not in st.session_state: st.session_state.last_uploaded_filename = None
    if "pending_image_for_main_gemini" not in st.session_state: st.session_state.pending_image_for_main_gemini = None
    
    # 高管工作流
    if "executive_workflow_stage" not in st.session_state: st.session_state.executive_workflow_stage = "idle"
    if "executive_user_query" not in st.session_state: st.session_state.executive_user_query = ""
    if "executive_data_profile_str" not in st.session_state: st.session_state.executive_data_profile_str = ""
    if "cfo_analysis_text" not in st.session_state: st.session_state.cfo_analysis_text = ""
    if "coo_analysis_text" not in st.session_state: st.session_state.coo_analysis_text = ""
    if "ceo_summary_text" not in st.session_state: st.session_state.ceo_summary_text = ""
    for exec_id_key in ["cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages"]:
        if exec_id_key not in st.session_state: st.session_state[exec_id_key] = []

    # 其他角色
    for role_id, role_info in ROLE_DEFINITIONS.items():
        if role_info["messages_key"] not in st.session_state: st.session_state[role_info["messages_key"]] = []
        if role_info["chat_session_key"] not in st.session_state: st.session_state[role_info["chat_session_key"]] = None

    # 除錯模式
    if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
    if "debug_logs" not in st.session_state: st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state: st.session_state.debug_errors = []

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # --- Gemini API Key 輸入 ---
        st.text_input(
            "請輸入您的 Google Gemini API Key",
            value=st.session_state.get("gemini_api_key_input", ""),
            type="password",
            key="gemini_api_key_input"
        )
        st.caption("優先使用此處輸入的金鑰，若為空則嘗試從雲端 Secrets 或 .env 檔案載入。")
        
        st.divider()

        # --- 功能區 ---
        st.subheader("📁 資料分析")
        uploaded_file = st.file_uploader("上傳 CSV 以啟用資料分析代理", type=["csv"])
        if uploaded_file:
            if uploaded_file.name != st.session_state.get("last_uploaded_filename") or not st.session_state.get("pandas_agent"):
                st.session_state.last_uploaded_filename = uploaded_file.name
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = file_path
                with st.spinner("正在初始化 Gemini 資料分析代理..."):
                    st.session_state.pandas_agent = create_pandas_agent(file_path)
        
        if st.session_state.get("pandas_agent"):
            st.success("✅ 資料分析代理已啟用！")

        st.subheader("🖼️ 圖片分析")
        uploaded_image = st.file_uploader("上傳圖片進行分析", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            add_user_image_to_main_chat(uploaded_image)
        
        st.divider()

        # --- 清除按鈕與偵錯 ---
        if st.button("🗑️ 清除對話與資料"):
            # 只清除對話和工作狀態，保留 API Key
            keys_to_clear = [
                "messages", "pandas_agent", "uploaded_file_path", "last_uploaded_filename",
                "pending_image_for_main_gemini", "executive_workflow_stage", "executive_user_query",
                "executive_data_profile_str", "cfo_analysis_text", "coo_analysis_text",
                "ceo_summary_text", "cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages",
                "debug_logs", "debug_errors"
            ]
            # 清除所有角色的對話紀錄
            for role_info in ROLE_DEFINITIONS.values():
                keys_to_clear.append(role_info["messages_key"])
                keys_to_clear.append(role_info["chat_session_key"])

            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state.pop(key)
            st.success("所有對話和工作狀態已清除！")
            st.rerun()

        st.session_state.debug_mode = st.checkbox("啟用偵錯模式", value=st.session_state.get("debug_mode", False))
        if st.session_state.debug_mode:
            with st.expander("🛠️ 偵錯資訊"):
                st.json(st.session_state.get("debug_logs", []))
                st.json(st.session_state.get("debug_errors", []))

    # --- 主工作區 (標籤頁面) ---
    tab_titles = ["💬 主要聊天室", "💼 高管工作流"] + [role["name"] for role in ROLE_DEFINITIONS.values()]
    tabs = st.tabs(tab_titles)

    # 主要聊天室標籤
    with tabs[0]:
        st.header("💬 主要聊天室 (分析數據與圖片)")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], list): # 處理圖片
                    for item in msg["content"]: st.image(item)
                else:
                    st.markdown(msg["content"])
        
        if st.session_state.get("pending_image_for_main_gemini"):
            st.chat_message("user").image(st.session_state.pending_image_for_main_gemini)

        if user_input := st.chat_input("請對數據或圖片提問..."):
            append_message_to_stream("messages", "user", user_input)
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_prompt = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Gemini 正在思考中..."):
                    response = ""
                    if st.session_state.get("pandas_agent"):
                        response = query_pandas_agent(st.session_state.pandas_agent, last_prompt)
                    else:
                        pending_image = st.session_state.get("pending_image_for_main_gemini")
                        response = get_gemini_response_main_chat(last_prompt, pending_image)
                    st.markdown(response)
                    append_message_to_stream("messages", "assistant", response)

    # 高管工作流標籤
    with tabs[1]:
        st.header("💼 高管工作流 (由 Gemini Pro 驅動)")
        st.write("請先在側邊欄上傳CSV資料，然後在此輸入商業問題，最後點擊按鈕啟動分析。")
        st.session_state.executive_user_query = st.text_area(
            "請輸入商業問題以啟動分析:", 
            value=st.session_state.get("executive_user_query", ""), 
            height=100
        )
        can_start = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
        
        if st.button("🚀 啟動/重啟高管分析", disabled=not can_start):
             # 重置狀態以重新開始
            st.session_state.executive_workflow_stage = "data_profiling_pending"
            st.session_state.executive_data_profile_str = ""
            st.session_state.cfo_analysis_text = ""
            st.session_state.coo_analysis_text = ""
            st.session_state.ceo_summary_text = ""
            st.rerun()
        
        # --- 工作流狀態機 ---
        if st.session_state.executive_workflow_stage == "data_profiling_pending":
            with st.spinner("正在生成資料摘要..."):
                df = pd.read_csv(st.session_state.uploaded_file_path)
                st.session_state.executive_data_profile_str = generate_data_profile(df)
                st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                st.rerun()

        if st.session_state.executive_data_profile_str:
            with st.expander("查看資料摘要"):
                st.text(st.session_state.executive_data_profile_str)

        # --- CFO 分析階段 ---
        if st.session_state.executive_workflow_stage == "cfo_analysis_pending":
            with st.spinner("CFO 正在分析... (Gemini Pro)"):
                cfo_prompt = f"""作為財務長(CFO)，請基於以下商業問題和資料摘要，提供財務角度的簡潔分析，包括成本、營收、利潤等潛在影響。

                商業問題: {st.session_state.executive_user_query}
                
                資料摘要:
                {st.session_state.executive_data_profile_str}
                """
                response = get_gemini_executive_analysis("CFO", cfo_prompt)
                st.session_state.cfo_analysis_text = response
                st.session_state.executive_workflow_stage = "coo_analysis_pending"
                st.rerun()
        
        if st.session_state.cfo_analysis_text:
            st.subheader("📊 財務長 (CFO) 分析")
            st.markdown(st.session_state.cfo_analysis_text)
        
        # --- COO 分析階段 (已補全) ---
        if st.session_state.executive_workflow_stage == "coo_analysis_pending":
            with st.spinner("COO 正在分析... (Gemini Pro)"):
                coo_prompt = f"""作為營運長(COO)，請基於以下商業問題、資料摘要和財務長(CFO)的分析，提供營運和執行層面的策略與潛在風險。請保持簡潔有力。

                商業問題: {st.session_state.executive_user_query}
                
                資料摘要:
                {st.session_state.executive_data_profile_str}

                CFO 的財務分析:
                {st.session_state.cfo_analysis_text}
                """
                response = get_gemini_executive_analysis("COO", coo_prompt)
                st.session_state.coo_analysis_text = response
                st.session_state.executive_workflow_stage = "ceo_summary_pending" # 更新狀態到下一步
                st.rerun()

        if st.session_state.coo_analysis_text:
            st.subheader("🏭 營運長 (COO) 分析")
            st.markdown(st.session_state.coo_analysis_text)

        # --- CEO 總結階段 (已補全) ---
        if st.session_state.executive_workflow_stage == "ceo_summary_pending":
            with st.spinner("CEO 正在進行最終總結... (Gemini Pro)"):
                ceo_prompt = f"""作為執行長(CEO)，請整合以下所有資訊（原始商業問題、資料摘要、CFO的財務分析、COO的營運分析），提供一個高層次的決策總結與明確的行動建議。

                商業問題: {st.session_state.executive_user_query}

                資料摘要:
                {st.session_state.executive_data_profile_str}

                CFO 的財務分析:
                {st.session_state.cfo_analysis_text}

                COO 的營運分析:
                {st.session_state.coo_analysis_text}
                """
                response = get_gemini_executive_analysis("CEO", ceo_prompt)
                st.session_state.ceo_summary_text = response
                st.session_state.executive_workflow_stage = "completed" # 標記工作流完成
                st.rerun()

        if st.session_state.ceo_summary_text:
            st.subheader("👑 執行長 (CEO) 最終決策")
            st.markdown(st.session_state.ceo_summary_text)


    # 其他 AI 角色標籤
    for i, (role_id, role_info) in enumerate(ROLE_DEFINITIONS.items()):
        with tabs[i + 2]:
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"])
            message_key = role_info["messages_key"]
            for msg in st.session_state[message_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if user_input := st.chat_input(f"與 {role_info['name']} 對話..."):
                append_message_to_stream(message_key, "user", user_input)
                # 直接在輸入後處理回應，避免需要兩次rerun
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("正在生成回應..."):
                        response = get_gemini_response_for_generic_role(role_id, user_input)
                        st.markdown(response)
                        append_message_to_stream(message_key, "assistant", response)


if __name__ == "__main__":
    main()
