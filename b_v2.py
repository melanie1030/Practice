import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO
import openai  # 正確導入 OpenAI 客戶端
from PIL import Image
from streamlit_ace import st_ace
import time

# --- Initialization and Settings ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4-turbo",  # 使用更穩定的模型
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o"
]

MAX_MESSAGES = 10  # 限制訊息歷史數量

def initialize_client(api_key):
    """使用提供的 API 密鑰初始化 OpenAI 客戶端。"""
    if api_key:
        openai.api_key = api_key
        return True
    return False

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
        print(msg)

def save_uploaded_file(uploaded_file):
    """保存上傳的文件到指定目錄。"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image):
    """將圖片轉換為 Base64 編碼。"""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # 使用 PNG 格式保持一致性
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message(role, content):
    """
    添加一條訊息，並確保總訊息數量不超過 MAX_MESSAGES。
    """
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # 移除最舊的訊息，保留系統提示
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES - 1):]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(uploaded_file):
    """
    使用 Markdown 圖片語法將圖片添加到訊息歷史中，並保存圖片文件。
    """
    try:
        # 打開上傳的圖片
        image = Image.open(uploaded_file)
        img_base64 = load_image_base64(image)
        if img_base64:
            # 使用 Markdown 圖片語法嵌入圖片
            image_markdown = f"![image](data:image/png;base64,{img_base64})"
            append_message("user", image_markdown)  # 將圖片訊息添加到訊息歷史
            st.session_state.image_base64 = img_base64  # 更新 image_base64
            st.session_state.uploaded_image_path = save_uploaded_file(uploaded_file)  # 保存圖片檔案路徑
            st.success("圖片上傳成功！")
            debug_log("Image uploaded and added to messages.")
        else:
            debug_error("無法將圖片轉換為 base64。")
    except Exception as e:
        debug_error(f"處理上傳圖片時出錯: {e}")

def reset_session_messages():
    """清除會話中的所有訊息。"""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
        st.success("Memory cleared!")
        debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
    """
    執行用戶提供的 Python 代碼，並返回執行結果。
    """
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.session_state.debug_logs.append(f"```python\n{code}\n```")

        debug_log(f"Executing code with global_vars: {list(exec_globals.keys())}")
        exec(code, exec_globals)
        output = exec_globals.get("output", "(No output returned)")
        debug_log(f"Execution output: {output}")
        return f"Code executed successfully. Output: {output}"
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log(f"Execution error: {error_msg}")
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode)."

def extract_json_block(response: str) -> str:
    """
    從 API 回應中提取 JSON 區塊。
    """
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

def get_llm_response(model_params, max_retries=3):
    """
    與 OpenAI API 同步獲取回應，包含重試邏輯。
    """
    retries = 0
    wait_time = 5  # 初始等待時間 5 秒

    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model_params.get("model", "gpt-4-turbo"),
                messages=st.session_state.messages,
                temperature=model_params.get("temperature", 0.3),
                max_tokens=model_params.get("max_tokens", 4096),
                stream=False  # 禁用串流
            )
            # 提取完整的回應內容
            response_content = response.choices[0].message['content'].strip()
            debug_log(f"Full assistant response: {response_content}")
            return response_content

        except Exception as e:
            if 'rate_limit_exceeded' in str(e).lower() or '429' in str(e):
                debug_error(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2  # 指數回退
            else:
                debug_error(f"Error getting response: {e}")
                st.error(f"An error occurred while getting the response: {e}")
                return ""

    st.error("Max retries exceeded. Please try again later.")
    return ""

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug & Deep Analysis)")

    # 初始化 session state 變數
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "editor_location" not in st.session_state:
        st.session_state.editor_location = "Main"
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state:
        st.session_state.deep_analysis_mode = False
    if "second_response" not in st.session_state:
        st.session_state.second_response = ""
    if "third_response" not in st.session_state:
        st.session_state.third_response = ""
    if "deep_analysis_image" not in st.session_state:
        st.session_state.deep_analysis_image = None
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state:
        st.session_state.debug_errors = []

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API密鑰", value=default_api_key, type="password", key="api_key_input")

        selected_model = st.selectbox("Select Model:", OPENAI_MODELS, index=0)

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode", value=False)

        if "memory" not in st.session_state:
            st.session_state.memory = []

        if "conversation_initialized" not in st.session_state:
            if api_key:
                # 初始化 OpenAI 客戶端
                client_initialized = initialize_client(api_key)
                if client_initialized:
                    st.session_state.conversation_initialized = True
                    st.session_state.messages = []  # 使用空訊息歷史初始化
                    debug_log("Conversation initialized with empty message history.")
                else:
                    st.warning("Failed to initialize OpenAI client.")
            else:
                st.warning("⬅️ Please enter your API Key to initialize the chatbot.")

        if st.session_state.debug_mode:
            debug_log(f"Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory = []
            st.session_state.messages = []
            st.session_state.ace_code = ""
            st.session_state.uploaded_file_path = None
            st.session_state.uploaded_image_path = None
            st.session_state.image_base64 = None
            st.session_state.deep_analysis_mode = False
            st.session_state.second_response = ""
            st.session_state.third_response = ""
            st.session_state.deep_analysis_image = None
            st.session_state.debug_logs = []
            st.session_state.debug_errors = []
            st.success("Memory cleared!")
            debug_log("Memory has been cleared.")

        st.subheader("🧠 Memory State")
        if st.session_state.messages:
            memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.text_area("Current Memory", value=memory_content, height=200)
            debug_log(f"Current memory content: {memory_content}")
        else:
            st.text_area("Current Memory", value="No messages yet.", height=200)
            debug_log("No messages in memory.")

        # --- CSV Upload ---
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
                debug_log(f"CSV Data Columns: {list(csv_data.columns)}")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading CSV: {e}")
                debug_log(f"Error reading CSV: {e}")

        # --- Image Upload ---
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_image:
            add_user_image(uploaded_image)

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location
        debug_log(f"Editor location set to: {st.session_state.editor_location}")

        # --- 調試區塊移動到側邊欄 ---
        with st.expander("🛠️ 調試與會話信息", expanded=False):
            if st.session_state.debug_mode:
                st.subheader("調試日志")
                if st.session_state.debug_logs:
                    debug_logs_combined = "\n".join(st.session_state.debug_logs)
                    st.text_area("Debug Logs", value=debug_logs_combined, height=200)
                else:
                    st.write("沒有調試日志。")

                st.subheader("調試錯誤")
                if st.session_state.debug_errors:
                    debug_errors_combined = "\n".join(st.session_state.debug_errors)
                    st.text_area("Debug Errors", value=debug_errors_combined, height=200)
                else:
                    st.write("沒有調試錯誤。")

            st.subheader("會話信息 (messages.json)")
            if "messages" in st.session_state:
                messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
                st.text_area("messages.json", value=messages_json, height=300)

                # 添加下載按鈕
                st.download_button(
                    label="📥 下載 messages.json",
                    data=messages_json,
                    file_name="messages.json",
                    mime="application/json"
                )
            else:
                st.write("沒有找到 messages。")

    # --- Display Message History ---
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], str):
                    # 檢查是否包含 Markdown 圖片語法
                    image_match = re.search(r'!\[.*?\]\((.*?)\)', message["content"])
                    if image_match:
                        image_url = image_match.group(1)
                        st.image(image_url, caption="📷 上傳的圖片", use_column_width=True)
                        debug_log(f"Displaying image from {message['role']}: {image_url}")
                    else:
                        # 檢查是否包含 Python 代碼塊
                        code_match = re.search(r'```python\s*(.*?)\s*```', message["content"], re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            st.code(code, language="python")
                            debug_log(f"Displaying code from {message['role']}: {code}")
                        else:
                            # 顯示普通文本
                            st.write(message["content"])
                            debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")
                else:
                    # 如果 content 不是字符串，顯示原始內容
                    st.write(message["content"])
                    debug_log(f"Displaying non-string message {idx} from {message['role']}: {message['content']}")

    # --- User Input ---
        user_input = st.chat_input("Hi! Ask me anything...")
        if user_input:
            append_message("user", user_input)
            with st.chat_message("user"):
                st.write(user_input)
                debug_log(f"User input added to messages: {user_input}")

            with st.spinner("Thinking..."):
                try:
                    # 確保 OpenAI 客戶端已初始化
                    if not initialize_client(api_key):
                        raise ValueError("OpenAI API Key is not provided or invalid.")

                    debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
                    debug_log(f"Uploaded image path: {st.session_state.uploaded_image_path}")

                    # --- Ensure system prompt is added only once ---
                    if not any(msg["role"] == "system" for msg in st.session_state.messages):
                        system_prompt = "You are an assistant that helps with data analysis."
                        append_message("system", system_prompt)
                        debug_log("System prompt added to messages.")

                    # --- Decide which prompt to use ---
                    if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                        # 圖片已上傳，直接使用用戶輸入
                        prompt = user_input
                        debug_log("User input with image data already appended.")
                    else:
                        # 沒有圖片上傳，使用複雜的 JSON 邏輯
                        if st.session_state.uploaded_file_path is not None:
                            try:
                                df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                                csv_columns = ", ".join(df_temp.columns)
                                debug_log(f"CSV columns: {csv_columns}")
                            except Exception as e:
                                csv_columns = "Unable to read columns"
                                if st.session_state.debug_mode:
                                    st.error(f"Error reading columns: {e}")
                                debug_log(f"Error reading columns: {e}")
                        else:
                            csv_columns = "No file uploaded"
                            debug_log("No CSV file uploaded.")

                        if st.session_state.uploaded_file_path is not None and csv_columns != "No file uploaded":
                            prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "Here are my observations: {{analysis}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# Read CSV file (use st.session_state.uploaded_file_path variable)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# Add your plotting or analysis logic here\\n\\n# For example, to display a plot using st.pyplot():\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)"
}}
Important:
1) Must use st.session_state.uploaded_file_path as the CSV path (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)

Based on the request: {user_input}.
Available columns: {csv_columns}.
"""
                            debug_log("Prompt constructed for CSV input with JSON response.")
                            append_message("system", prompt)
                            debug_log("System prompt appended to messages.")
                        else:
                            prompt = f"Please answer this question entirely in Traditional Chinese: {user_input}"
                            debug_log("Prompt constructed for plain text input.")
                            append_message("system", prompt)
                            debug_log("Plain text system prompt appended to messages.")

                    # Make the API request and get the response
                    model_params = {
                        "model": selected_model,
                        "temperature": 0.5,
                        "max_tokens": 4096
                    }
                    response_content = get_llm_response(model_params)
                    debug_log(f"Full assistant response: {response_content}")

                    if response_content:
                        # Append assistant message
                        append_message("assistant", response_content)
                        with st.chat_message("assistant"):
                            st.write(response_content)  # 避免二次顯示
                            debug_log(f"Assistant response added to messages: {response_content}")

                        # Extract JSON and code
                        json_str = extract_json_block(response_content)
                        try:
                            response_json = json.loads(json_str)
                            debug_log("JSON parsing successful.")
                        except Exception as e:
                            debug_log(f"json.loads parsing error: {e}")
                            debug_error(f"json.loads parsing error: {e}")
                            response_json = {"content": json_str, "code": ""}
                            debug_log("Fallback to raw response for content.")

                        content = response_json.get("content", "Here is my analysis:")
                        append_message("assistant", content)

                        code = response_json.get("code", "")
                        if code:
                            code_block = f"```python\n{code}\n```"
                            append_message("assistant", code_block)
                            with st.chat_message("assistant"):
                                st.code(code, language="python")
                            st.session_state.ace_code = code
                            debug_log("ace_code updated with new code.")

                        # --- If deep analysis mode is checked & code is present -> execute code and re-analyze chart ---
                        if st.session_state.deep_analysis_mode and code:
                            st.write("### [Deep Analysis] Automatically executing the generated code and sending the chart for analysis...")
                            debug_log("Deep analysis mode activated.")

                            global_vars = {
                                "uploaded_file_path": st.session_state.uploaded_file_path,
                                "uploaded_image_path": st.session_state.uploaded_image_path,
                            }
                            exec_result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                            st.write("#### Execution Result")
                            st.text(exec_result)
                            debug_log(f"Execution result: {exec_result}")

                            # 確保生成的圖表已經被創建
                            fig = plt.gcf()
                            buf = BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                            st.session_state.deep_analysis_image = chart_base64
                            debug_log("Chart has been converted to base64.")

                            # 使用 Markdown 圖片語法將圖表添加到訊息歷史中
                            image_markdown = f"![image](data:image/png;base64,{chart_base64})"
                            append_message("user", image_markdown)
                            debug_log("Deep analysis image added to messages.")

                            # Make the API request for deep analysis
                            second_raw_response = get_llm_response(model_params)
                            debug_log(f"Deep analysis response: {second_raw_response}")

                            if second_raw_response:
                                # Append assistant response
                                append_message("assistant", second_raw_response)
                                st.session_state.second_response = second_raw_response
                                with st.chat_message("assistant"):
                                    st.write(second_raw_response)
                                    debug_log(f"Deep analysis response added to messages: {second_raw_response}")

                                # Prepare final summary prompt
                                prompt_3 = f"""
First response content: {content}
Second response chart analysis content: {second_raw_response}

Please help me summarize the above two responses and provide additional suggestions or insights.
"""
                                debug_log(f"Final Summary Prompt: {prompt_3}")

                                # Append prompt_3 to messages
                                append_message("user", prompt_3)
                                debug_log("Final summary prompt appended to messages.")

                                # Make the API request for final summary
                                third_raw_response = get_llm_response(model_params)
                                debug_log(f"Final summary response: {third_raw_response}")

                                if third_raw_response:
                                    # Append assistant response
                                    append_message("assistant", third_raw_response)
                                    st.session_state.third_response = third_raw_response
                                    with st.chat_message("assistant"):
                                        st.write(third_raw_response)
                                        debug_log(f"Final summary response added to messages: {third_raw_response}")

                                    # Display the chart
                                    st.write("#### [Deep Analysis] Chart:")
                                    try:
                                        img_data = base64.b64decode(st.session_state.deep_analysis_image)
                                        st.image(img_data, caption="Chart generated from deep analysis", use_column_width=True)
                                        debug_log("Deep analysis chart displayed.")
                                    except Exception as e:
                                        if st.session_state.debug_mode:
                                            st.error(f"Error displaying chart: {e}")
                                        debug_log(f"Error displaying chart: {e}")

                except Exception as e:
                    if st.session_state.debug_mode:
                        st.error(f"An error occurred: {e}")
                    debug_log(f"An error occurred: {e}")

    # --- Persistent Code Editor ---
        if "ace_code" not in st.session_state:
            st.session_state.ace_code = ""

        if st.session_state.editor_location == "Main":
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
                    debug_log("ace_code updated from main editor.")

                if st.button("▶️ Execute Code", key="execute_code_main"):
                    global_vars = {
                        "uploaded_file_path": st.session_state.uploaded_file_path,
                        "uploaded_image_path": st.session_state.uploaded_image_path,
                    }
                    debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                    debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                    result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                    st.write("### Execution Result")
                    st.text(result)
                    debug_log(f"Code execution result: {result}")

        else:
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
                    debug_log("ace_code updated from sidebar editor.")

                if st.button("▶️ Execute Code", key="execute_code_sidebar"):
                    global_vars = {
                        "uploaded_file_path": st.session_state.uploaded_file_path,
                        "uploaded_image_path": st.session_state.uploaded_image_path,
                    }
                    debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                    debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                    result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                    st.write("### Execution Result")
                    st.text(result)
                    debug_log(f"Code execution result: {result}")

    if __name__ == "__main__":
        main()
