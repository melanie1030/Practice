import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
from io import StringIO
import google.generativeai as genai
import time
import matplotlib.font_manager as fm
import matplotlib
import sys

# --- LangChain and Gemini Imports ---
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 指定中文字型 ---
try:
    font_path = "./fonts/msjh.ttc"
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字型載入失敗，圖表可能無法正常顯示中文: {e}")

# --- 初始化設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MAX_MESSAGES_PER_STREAM = 20 # 可以適當增加對話歷史長度

# --- 基礎輔助函數 ---
def debug_log(msg):
    """在 session state 和控制台中記錄偵錯訊息。"""
    if st.session_state.get("debug_mode", False):
        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"**LOG ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    """在 session state 和控制台中記錄錯誤訊息。"""
    if st.session_state.get("debug_mode", False):
        if "debug_errors" not in st.session_state:
            st.session_state.debug_errors = []
        st.session_state.debug_errors.append(f"**ERROR ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG ERROR: {msg}")

def save_uploaded_file(uploaded_file):
    """將上傳的檔案儲存到本地，並返回檔案路徑。"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def append_message_to_stream(role, content):
    """將訊息附加到主對話流中，並管理歷史長度。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES_PER_STREAM:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_PER_STREAM:]

# --- Gemini Pandas Agent 核心函數 ---
def create_pandas_agent(file_path: str):
    """
    建立由 Google Gemini 驅動的 LangChain Pandas DataFrame Agent。
    API Key 的獲取順序為：側邊欄輸入 > Streamlit Secrets。
    """
    debug_log(f"準備為檔案 '{file_path}' 建立 Gemini Pandas Agent。")
    
    # --- Gemini API Key 的優先級邏輯 ---
    gemini_api_key = st.session_state.get("gemini_api_key_input")
    if not gemini_api_key:
        try:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")
            if gemini_api_key:
                debug_log("從 Streamlit Secrets 備用載入 Gemini API Key。")
        except Exception:
            pass # 如果 Secrets 不存在則忽略
            
    if not gemini_api_key:
        st.error("建立 Gemini 資料分析代理需要 API Key。請在側邊欄輸入或在應用的 Secrets 中設定。")
        return None

    try:
        df = pd.read_csv(file_path)

        # 初始化 Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            google_api_key=gemini_api_key,
            convert_system_message_to_human=True 
        )
        
        # 建立 Pandas Agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=st.session_state.get("debug_mode", False),
            handle_parsing_errors=True,
            allow_dangerous_code=True # 在受信任的環境中執行
        )
        
        debug_log("由 Gemini 驅動的 Pandas Agent 已成功建立。")
        return agent
        
    except FileNotFoundError:
        st.error(f"檔案未找到：{file_path}")
        return None
    except Exception as e:
        st.error(f"建立資料分析代理時發生錯誤: {e}")
        debug_error(f"Pandas Agent creation failed: {e}, Traceback: {traceback.format_exc()}")
        return None

def query_pandas_agent(agent, query: str):
    """
    使用給定的問題查詢 Pandas Agent。
    """
    if not agent:
        return "錯誤：資料分析代理尚未初始化。請先上傳 CSV 檔案。"
    
    prompt = f"""
    請針對以下問題進行分析，並用繁體中文回答。
    你的回答應該清晰、易於理解，如果需要，請直接給出計算結果或結論。
    
    問題: "{query}"
    """
    debug_log(f"正在用問題查詢 Pandas Agent: '{query}'")
    try:
        # 使用標準的 .invoke() 方法呼叫代理
        response = agent.invoke({"input": prompt})
        result = response.get("output", "代理沒有提供有效的輸出。")
        debug_log(f"Pandas Agent 的原始回應: {response}")
        return result
    except Exception as e:
        error_message = f"代理在處理您的請求時發生錯誤: {e}"
        debug_error(f"Pandas Agent invocation error: {e}, Traceback: {traceback.format_exc()}")
        st.error(error_message)
        return error_message

# ------------------------------
# 主應用入口 (最終完整版)
# ------------------------------
def main():
    st.set_page_config(
        page_title="Gemini CSV 資料分析助理",
        page_icon="🤖",
        layout="centered"
    )
    st.title("🤖 Gemini CSV 資料分析助理")

    # --- 初始化簡化後的 Session States ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pandas_agent" not in st.session_state:
        st.session_state.pandas_agent = None
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state:
        st.session_state.debug_errors = []

    # --- 側邊欄介面 ---
    with st.sidebar:
        st.header("⚙️ 設定")
        st.caption("請先提供您的 API Key 並上傳 CSV 檔案。")

        # --- MODIFIED SECTION: Removed the problematic if block ---
        # The st.text_input widget with a key will automatically manage the session state.
        st.text_input(
            "請輸入您的 Google Gemini API Key",
            value=st.session_state.get("gemini_api_key_input", ""),
            type="password",
            key="gemini_api_key_input"
        )
        # No need for: if gemini_api_key_input: st.session_state.gemini_api_key_input = ...

        # CSV 檔案上傳器
        uploaded_file = st.file_uploader(
            "上傳您的 CSV 檔案",
            type=["csv"],
            key="main_csv_uploader_sidebar"
        )
        
        # 檔案上傳後的處理邏輯
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path != st.session_state.get("uploaded_file_path") or not st.session_state.get("pandas_agent"):
                st.session_state.uploaded_file_path = file_path
                with st.spinner("正在初始化資料分析代理..."):
                    st.session_state.pandas_agent = create_pandas_agent(file_path)
        
        st.divider()

        # 清除按鈕與偵錯工具
        if st.button("🗑️ 清除對話與資料"):
            # A list of keys to clear from session state
            keys_to_clear = [
                "messages", "pandas_agent", "uploaded_file_path", 
                "debug_logs", "debug_errors", "gemini_api_key_input" # Also clear the key
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state.pop(key)
            st.success("所有對話和資料都已清除！")
            st.rerun()

        st.session_state.debug_mode = st.checkbox("啟用偵錯模式", value=st.session_state.get("debug_mode", False))
        if st.session_state.debug_mode:
            with st.expander("🛠️ 偵錯資訊", expanded=False):
                st.write("除錯日誌:")
                st.json(st.session_state.get("debug_logs", []))
                st.write("錯誤日誌:")
                st.json(st.session_state.get("debug_errors", []))


if __name__ == "__main__":
    main()
