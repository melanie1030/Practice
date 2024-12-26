import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from streamlit_ace import st_ace

# --- Initialize and Settings ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"

# 可自由新增/刪除你想要的模型名稱
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k"
]


def debug_log(msg):
    """只有在 debug_mode == True 時才顯示 debug 訊息."""
    if st.session_state.get("debug_mode", False):
        st.write(msg)
        print(msg)


def debug_error(msg):
    """只有在 debug_mode == True 時才顯示 error 訊息."""
    if st.session_state.get("debug_mode", False):
        st.error(msg)
        print(msg)


def initialize_client(api_key, model_name):
    """Initialize OpenAI client with the provided API key and model."""
    return ChatOpenAI(
        model=model_name,
        temperature=0.5,
        openai_api_key=api_key
    ) if api_key else None


def save_uploaded_file(uploaded_file):
    """保存上傳的檔案到指定目錄，並返回檔案路徑"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # Debug: 顯示即將寫入的路徑
    debug_log(f"DEBUG: saving file to {file_path}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Debug: 列出該目錄下的檔案
    debug_log(f"DEBUG: files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")

    return file_path


def execute_code(code, global_vars=None):
    """Execute the given Python code and capture output."""
    try:
        exec_globals = global_vars if global_vars else {}
        # Debug: 顯示要執行的程式碼
        debug_log("DEBUG: Ready to exec the following code:")
        if st.session_state.get("debug_mode", False):
            st.code(code, language="python")

        debug_log("[DEBUG] Exec code with global_vars: " + str(list(exec_globals.keys())))
        exec(code, exec_globals)
        return "Code executed successfully. Output: " + str(exec_globals.get("output", "(No output returned)"))
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log("[DEBUG] Execution error: " + error_msg)
        # 只有 debug_mode 時才顯示完整錯誤
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            # 若 debug_mode == False，就不顯示詳細錯誤
            return "Error executing code (hidden in non-debug mode)."


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
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        # 新增：選擇要使用的模型
        selected_model = st.selectbox("選擇模型:", OPENAI_MODELS, index=0)

        # 新增：Debug Mode
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)

        # 初始化 LangChain 與記憶
        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = initialize_client(api_key, selected_model)
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
            st.session_state.uploaded_image_path = None
            st.session_state.image_base64 = None
            st.success("Memory cleared!")

        # 顯示記憶狀態
        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        # ===================== CSV 上傳 =====================
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None

        if uploaded_file:
            # 保存檔案並記錄路徑
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            debug_log(f"DEBUG: st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")

            # 讀取檔案到 DataFrame
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
            except Exception as e:
                # 只有 debug_mode==True 才顯示詳細錯誤
                if st.session_state.debug_mode:
                    st.error(f"Error reading CSV: {e}")
                debug_log(f"[DEBUG] Error reading CSV: {e}")

        # ===================== 圖片上傳功能 =====================
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            # 保存圖片並記錄路徑
            st.session_state.uploaded_image_path = save_uploaded_file(uploaded_image)
            debug_log(f"DEBUG: st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

            # 顯示圖片預覽
            st.image(st.session_state.uploaded_image_path, caption="Uploaded Image Preview", use_column_width=True)

            # 將圖片轉成 base64 編碼（若你想要在 prompt 中傳遞給 GPT）
            try:
                with open(st.session_state.uploaded_image_path, "rb") as f:
                    img_bytes = f.read()
                st.session_state.image_base64 = base64.b64encode(img_bytes).decode("utf-8")
                debug_log("DEBUG: Image has been converted to base64.")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error converting image to base64: {e}")
                debug_log(f"[DEBUG] Error converting image to base64: {e}")

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
                debug_log(f"DEBUG: Currently st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: Currently st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

                # 準備 CSV 欄位資訊（若有上傳）
                if st.session_state.uploaded_file_path is not None:
                    try:
                        df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                        csv_columns = ", ".join(df_temp.columns)
                    except Exception as e:
                        csv_columns = "無法讀取欄位"
                        if st.session_state.debug_mode:
                            st.error(f"Error reading columns: {e}")
                        debug_log(f"[DEBUG] Error reading columns: {e}")
                else:
                    csv_columns = "無上傳檔案"

                prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "這是我的觀察：{{{{分析內容}}}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# 讀取 CSV 檔案 (請直接使用 st.session_state.uploaded_file_path 變數)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# 在這裡加入你要的繪圖或分析邏輯\\n\\n# 例如使用 st.pyplot() 來顯示圖表:\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)\\n"
}}
Important:
1) Must use st.session_state.uploaded_file_path as the CSV path (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)

Based on the request: {user_input}.
Available columns: {csv_columns}.
"""

                if st.session_state.image_base64:
                    prompt += "\nHere is the image data in base64 format:\n"
                    prompt += st.session_state.image_base64[:300] + "..."

                if csv_columns == "無上傳檔案":
                    prompt = f"請全部以繁體中文回答此問題：{user_input}"

                debug_log(f"DEBUG: Prompt used => {prompt}")

                raw_response = st.session_state.conversation.run(prompt)
                if st.session_state.debug_mode:
                    st.write("Model raw response:", raw_response)
                debug_log(f"[DEBUG] Model raw response => {raw_response}")

                # 擷取三反引號中的 JSON 區塊
                json_str = extract_json_block(raw_response)
                try:
                    response_json = json.loads(json_str)
                except Exception as e:
                    # 若 debug_mode=False，就不顯示這個 error
                    debug_log(f"json.loads parsing error: {e}")
                    # 若要在 debug 模式下顯示此錯誤：
                    debug_error(f"json.loads parsing error: {e}")
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
                # 只有 debug_mode==True 時顯示詳細錯誤
                if st.session_state.debug_mode:
                    st.error(f"An error occurred: {e}")
                debug_log(f"[DEBUG] An error occurred: {e}")

    # ===================== 根據 editor_location 決定編輯器要放在哪裡 =====================
    debug_log(f"DEBUG: editor_location = {st.session_state.editor_location}")
    debug_log(f"DEBUG: final st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
    debug_log(f"DEBUG: final st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

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
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"DEBUG: executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

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
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"DEBUG: executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)


if __name__ == "__main__":
    main()
