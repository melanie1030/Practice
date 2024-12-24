import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from streamlit_ace import st_ace

# --- Initialize and Settings ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"


def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key) if api_key else None


def save_uploaded_file(uploaded_file):
    """保存上傳的檔案到指定目錄，並返回檔案路徑"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # Debug: 顯示即將寫入的路徑
    st.write(f"DEBUG: saving file to {file_path}")
    print(f"[DEBUG] Saving file to: {file_path}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Debug: 列出該目錄下的檔案
    st.write(f"DEBUG: files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    print(f"[DEBUG] Files in {UPLOAD_DIR}:", os.listdir(UPLOAD_DIR))

    return file_path


def execute_code(code, global_vars=None):
    """Execute the given Python code and capture output."""
    try:
        exec_globals = global_vars if global_vars else {}
        # Debug: 顯示要執行的程式碼
        st.write("DEBUG: Ready to exec the following code:")
        st.code(code, language="python")

        print("[DEBUG] Exec code with global_vars:", list(exec_globals.keys()))
        exec(code, exec_globals)
        return "Code executed successfully. Output: " + str(exec_globals.get("output", "(No output returned)"))
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        print("[DEBUG] Execution error:", error_msg)
        return error_msg


def extract_json_block(response: str) -> str:
    """
    從模型回傳的字串中，找出 JSON 物件部分
    （例如模型用三反引號 ```json ... ``` 包起來）
    """
    pattern = r'```(?:json)?(.*)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        # 只取三反引號之間的內容
        json_str = match.group(1).strip()
        return json_str
    else:
        # 如果沒找到，就回傳原字串
        return response.strip()


def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug Logs)")

    # 如果尚未在 session_state 建立變數，先初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "editor_location" not in st.session_state:
        st.session_state.editor_location = "Main"  # 預設編輯器顯示在主區
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None

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
            st.session_state.ace_code = ""
            st.session_state.uploaded_file_path = None
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
            # 保存檔案並記錄路徑
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            # Debug: 確認是否拿到路徑
            st.write("DEBUG: st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)
            print("[DEBUG] st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)

            # 讀取檔案到 DataFrame
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                print("[DEBUG] Error reading CSV:", e)

        # 編輯器顯示位置
        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location

    # ===================== 主區：顯示對話、接收輸入、聊天功能 =====================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "code" in message:
                st.code(message["code"], language="python")

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # 記錄使用者訊息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 產生回覆
        with st.spinner("Thinking..."):
            try:
                # Debug: 檢查已上傳檔案路徑
                st.write("DEBUG: Currently st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)
                print("[DEBUG] Currently st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)

                if st.session_state.uploaded_file_path is not None:
                    # CSV 檔案的欄位名稱
                    try:
                        df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                        csv_columns = ", ".join(df_temp.columns)
                    except Exception as e:
                        csv_columns = "無法讀取欄位"
                        st.error(f"Error reading columns: {e}")
                        print("[DEBUG] Error reading columns:", e)
                else:
                    csv_columns = "無上傳檔案"

                # 準備 prompt
                prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "根據 {csv_columns} 的數據分析，這是我的觀察：{{{{分析內容}}}}",
    "code": "生成一些使用matplotlib來生成分析圖表的python code"
}}
Based on the request: {user_input}.
Available columns: {csv_columns}.
"""

                # 如果沒有上傳檔案，就改成全繁體
                if csv_columns == "無上傳檔案":
                    prompt = f"請全部以繁體中文回答此問題：{user_input}"

                st.write("DEBUG: Prompt used =>", prompt)
                print("[DEBUG] Prompt used =>", prompt)

                raw_response = st.session_state.conversation.run(prompt)
                st.write("Model raw response:", raw_response)
                print("[DEBUG] Model raw response =>", raw_response)

                # 擷取三反引號中的 JSON 區塊
                json_str = extract_json_block(raw_response)
                try:
                    response_json = json.loads(json_str)
                except Exception as e:
                    st.error(f"json.loads parsing error: {e}")
                    print("[DEBUG] json.loads parsing error:", e)
                    response_json = {"content": json_str, "code": ""}

                # 顯示回覆的文字內容
                content = response_json.get("content", "這是我的分析：")
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.write(content)

                # 如果有程式碼，則顯示並更新到 ace_code
                code = response_json.get("code", "")
                if code:
                    st.session_state.messages.append({"role": "assistant", "code": code})
                    with st.chat_message("assistant"):
                        st.code(code, language="python")
                    st.session_state.ace_code = code

            except Exception as e:
                st.error(f"An error occurred: {e}")
                print("[DEBUG] An error occurred:", e)

    # ===================== 根據 editor_location 決定編輯器要放在哪裡 =====================
    # Debug: 顯示目前 editor_location
    st.write("DEBUG: editor_location =", st.session_state.editor_location)
    print("[DEBUG] editor_location =", st.session_state.editor_location)

    # Debug: 顯示目前 uploaded_file_path
    st.write("DEBUG: final st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)
    print("[DEBUG] final st.session_state.uploaded_file_path =", st.session_state.uploaded_file_path)

    if st.session_state.editor_location == "Main":
        # 放在主區底部，用 expander 收合
        with st.expander("🖋️ Persistent Code Editor (Main)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_main"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code

            if st.button("▶️ Execute Code", key="execute_code_main"):
                # 在全局變數中注入檔案路徑
                global_vars = {"uploaded_file_path": st.session_state.uploaded_file_path}
                st.write("DEBUG: executing code with uploaded_file_path =", st.session_state.uploaded_file_path)
                print("[DEBUG] executing code with uploaded_file_path =", st.session_state.uploaded_file_path)

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)

    else:
        # 放在側邊欄，用 expander 收合
        with st.sidebar.expander("🖋️ Persistent Code Editor (Sidebar)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_sidebar"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code

            if st.button("▶️ Execute Code", key="execute_code_sidebar"):
                global_vars = {"uploaded_file_path": st.session_state.uploaded_file_path}
                st.write("DEBUG: executing code with uploaded_file_path =", st.session_state.uploaded_file_path)
                print("[DEBUG] executing code with uploaded_file_path =", st.session_state.uploaded_file_path)

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)


if __name__ == "__main__":
    main()
