import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
import time
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os
from streamlit_ace import st_ace
import traceback

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key) if api_key else None

def execute_code(code):
    """Execute the given Python code and capture output."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "Code executed successfully. Output: " + str(exec_globals.get("output", "(No output returned)"))
    except Exception as e:
        return f"Error executing code:\n{traceback.format_exc()}"

def main():
    # 設定 Streamlit 頁面標題與配置
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas")

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        # 初始化 LangChain 與記憶
        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = initialize_client(api_key)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ 請輸入 API Key 以初始化聊天機器人。")
                return

        # 清除記憶
        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

        # 顯示記憶狀態
        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        # 上傳 CSV
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

    # 儲存對話訊息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示先前生成的訊息（角色對話）
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "code" in message:
                st.code(message["code"], language="python")

    # 使用者輸入
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # 記錄使用者訊息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 產生回覆
        with st.spinner("Thinking..."):
            try:
                if csv_data is not None:
                    # 取得 CSV 欄位名稱
                    csv_columns = ", ".join(csv_data.columns)
                    # 使用雙大括號以顯示大括號字面量
                    prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "根據 {csv_columns} 的數據分析，這是我的觀察：{{{{分析內容}}}}",
    "code": ""
}}
Based on the request: {user_input}.
Available columns: {csv_columns}.
"""
                else:
                    prompt = f"請全部以繁體中文回答此問題：{user_input}"

                # 呼叫 LangChain
                response = st.session_state.conversation.run(prompt)

                # 為了 debug，可以先檢查原始輸出
                st.write("Model raw response:", response)

                # 嘗試轉成 JSON 格式
                try:
                    response_json = json.loads(response)
                except Exception as e:
                    st.error(f"json.loads parsing error: {e}")
                    # 如果解析失敗，fallback 為最簡單的格式
                    response_json = {"content": response, "code": ""}

                # 顯示回覆的文字內容
                content = response_json.get("content", "這是我的分析：")
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.write(content)

                # 如果有程式碼，則顯示
                code = response_json.get("code", "")
                if code:
                    st.session_state.messages.append({"role": "assistant", "code": code})
                    with st.chat_message("assistant"):
                        st.code(code, language="python")

                    # 顯示可編輯的 ACE Editor
                    st.write("### 🖋️ Edit and Execute Code")
                    edited_code = st_ace(value=code, language="python", theme="monokai", height=300)

                    # 執行按鈕
                    if st.button("▶️ Execute Code"):
                        result = execute_code(edited_code)
                        st.write("### Execution Result")
                        st.text(result)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
