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
from openai import OpenAI
from PIL import Image
import google.generativeai as genai  # 新增Gemini依赖
from streamlit_ace import st_ace
import time

# --- 初始化设置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
LLM_MODELS = [  # 修改后的模型列表
    "gpt-4o",
    "gpt-3.5-turbo-16k",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-2.0-flash"
]

MAX_MESSAGES = 10  # Limit message history

def initialize_client(api_key):
    return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        st.write(msg)
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
        print(msg)

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image):
    """Convert image to Base64 encoding."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # Use PNG for consistency
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message(role, content):
    """Append a message and ensure the total number of messages does not exceed MAX_MESSAGES."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Remove the oldest messages except the system prompt
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES - 1):]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(uploaded_file):
    """添加用戶圖片消息到session state"""
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        use_base64 = "gpt" in current_model  
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path
        
        if use_base64:
            # 為OpenAI模型生成Base64 URL
            image = Image.open(file_path)
            image_base64 = load_image_base64(image)
            image_url = f"data:image/{file_path.split('.')[-1]};base64,{image_base64}"
        else:
            image_url = file_path
        
        image_msg = {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "auto"
            }
        }
        
        if use_base64:
            append_message("user", [image_msg])
        else:
            return image_url
        debug_log(f"圖片消息已添加：{image_url[:50]}...")
        
        st.session_state.image_base64 = image_base64 if use_base64 else None
        st.rerun()
        
    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        st.error("圖片處理異常，請檢查日誌")

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
        st.success("Memory cleared!")
        debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
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
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

# ------------------------------
# 舊版本的 OpenAI 與 Gemini 回覆方法保留（供參考）
# ------------------------------

def get_gemini_response(model_params, max_retries=3):
    """
    整合新版 Gemini 請求方法，支援先讀取圖片 (generate_content) 再進行完整對話 (send_message)。
    流程：
      1) 偵測是否有圖片 (最後一則 user 訊息)
      2) 如果有，先用 generate_content() 取得一段回覆並加到 messages
      3) 最後用 send_message() 把整個 messages 發送給 Gemini，取得最終回覆
    """
    api_key = st.session_state.get("gemini_api_key_input", "")
    debug_log(f"gemini api key: {api_key}")
    if not api_key:
        st.error("未設定 Gemini API 金鑰")
        return ""
    genai.configure(api_key=api_key)
    model_name = model_params.get("model", "gemini-1.5-flash")
    debug_log(f"gemini model: {model_name}")
    model = genai.GenerativeModel(model_name)
    st.session_state.gemini_chat = model.start_chat(history=[])
    debug_log("Gemini chat session created.")

    if st.session_state.uploaded_image_path:
        debug_log("Detected user message with image, using generate_content() first...")
        gen_content = True
    else:
        gen_content = False

    last_user_msg_with_image = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    last_user_msg_with_image = msg
                    break
        if last_user_msg_with_image:
            break

    if gen_content:
        debug_log("Detected user message with image, using generate_content() first... by gen_content")
        text_parts = []
        image_data = st.session_state.uploaded_image_path
        debug_log("Using generate_content() first... by gen_content")
        try:
            debug_log("entering generate_content()")
            retries = 0
            while retries < max_retries:
                try:
                    debug_log("entering generate_content() try block")
                    imageee = genai.upload_file(path=image_data, display_name="Image")
                    debug_log(f"imageee: {imageee}")
                    response_gc = model.generate_content(["請你繁體中文解讀圖片", imageee])
                    debug_log(f"response_gc: {response_gc.text}")
                    generate_content_reply = response_gc.text
                    debug_log(f"Gemini generate_content reply: {generate_content_reply}")
                    append_message("assistant", generate_content_reply)
                    break
                except genai.GenerationError as e:
                    debug_error(f"generate_content() 失敗: {e}")
                    retries += 1
                    time.sleep(5 * retries)
                except Exception as e:
                    debug_error(f"generate_content() 其他錯誤: {e}")
                    return "generate_content Error"
        except Exception as e:
            debug_error(f"generate_content() 其他錯誤: {e}")

    converted_history = []
    for msg in st.session_state.messages:
        role = msg.get("role")
        parts = []
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append(f"[Image included, base64 size={len(item['image_url']['url'])} chars]")
                else:
                    parts.append(str(item))
        else:
            parts.append(str(msg["content"]))
        converted_history.append({"role": role, "parts": parts})
    converted_history_json = json.dumps(converted_history, ensure_ascii=False)
    debug_log(f"converted history (json) => {converted_history_json}")
    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            debug_log("Calling send_message with entire conversation...")
            response_sm = st.session_state.gemini_chat.send_message(converted_history_json)
            final_reply = response_sm.text.strip()
            debug_log(f"Gemini send_message final reply => {final_reply}")
            return final_reply
        except genai.GenerationError as e:
            debug_error(f"send_message() 生成錯誤: {e}")
            st.warning(f"Gemini 生成錯誤，{wait_time}秒後重試...")
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2
        except Exception as e:
            debug_error(f"send_message() API請求異常: {e}")
            st.error(f"Gemini API請求異常: {e}")
            return ""
    return "請求失敗次數過多，請稍後重試"

def get_openai_response(client, model_params, max_retries=3):
    """处理OpenAI API请求"""
    retries = 0
    wait_time = 5
    model_name = model_params.get("model", "gpt-4-turbo")
    while retries < max_retries:
        try:
            request_params = {
                "model": model_name,
                "messages": st.session_state.messages,
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096),
                "stream": False
            }
            if any(msg.get("content") and isinstance(msg["content"], list) for msg in st.session_state.messages):
                request_params["max_tokens"] = 4096
                debug_log("Detected multimodal input, adjusting max_tokens")
            response = client.chat.completions.create(**request_params)
            response_content = response.choices[0].message.content.strip()
            debug_log(f"OpenAI原始响应：\n{response_content}")
            return response_content
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                debug_error(f"速率限制错误（尝试 {retries+1}/{max_retries}）：{e}")
                st.warning(f"请求过于频繁，{wait_time}秒后重试...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower():
                debug_error(f"API密钥无效：{e}")
                st.error("OpenAI API密钥无效，请检查后重试")
                return ""
            else:
                debug_error(f"OpenAI请求异常：{str(e)}")
                st.error(f"请求发生错误：{str(e)}")
                return ""
    debug_error(f"超过最大重试次数（{max_retries}次）")
    st.error("请求失败次数过多，请稍后再试")
    return ""

def get_llm_response(client, model_params, max_retries=3):
    """獲取LLM模型回覆（支持OpenAI和Gemini）"""
    model_name = model_params.get("model", "gpt-4-turbo")
    debug_log(f"starting to get llm response...{model_name}")
    if "gpt" in model_name:
        debug_log("GPT")
        return get_openai_response(client, model_params, max_retries)
    elif "gemini" in model_name:
        debug_log("Gemini")
        return get_gemini_response(model_params=model_params, max_retries=max_retries)
    else:
        st.error(f"不支持的模型類型: {model_name}")
        return ""

# ------------------------------
# 新增二模型交叉驗證函數
# ------------------------------

def get_cross_validated_response(client, model_params_openai, model_params_gemini, max_retries=3):
    """
    簡單的二模型交叉驗證：
    1. 調用 OpenAI 模型 (例如 GPT-4-turbo) 獲取回答。
    2. 調用 Gemini 模型 (例如 gemini-1.5-flash) 獲取回答。
    3. 將兩者結果返回，方便後續做比對或投票決策。
    """
    response_openai = get_openai_response(client, model_params_openai, max_retries)
    response_gemini = get_gemini_response(model_params_gemini, max_retries)
    final_response = {
        "openai_response": response_openai,
        "gemini_response": response_gemini
    }
    return final_response

# ------------------------------
# 主應用入口
# ------------------------------

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug & Deep Analysis)")

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
    if "thinking_protocol" not in st.session_state:
        st.session_state.thinking_protocol = None
    if "gemini_ai_chat" not in st.session_state:
        st.session_state.gemini_ai_chat = None
    if "gemini_ai_history" not in st.session_state: 
        st.session_state.gemini_ai_history = []

    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input("OpenAI API Key", value=default_openai_key, type="password")
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input("Gemini API Key", 
                                       value=default_gemini_key, 
                                       type="password",
                                       key="gemini_api_key")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            st.session_state["gemini_api_key_input"] = gemini_api_key 

        selected_model = st.selectbox(
            "選擇模型", 
            LLM_MODELS, 
            index=0, 
            key="selected_model"
        )
        
        if "selected_model" in st.session_state:
            current_model = st.session_state.selected_model.lower()
            if "gemini" in current_model:
                gemini_key = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_api_key")
                if not gemini_key:
                    st.error("使用Gemini模型需在下方輸入API金鑰 🔑")
                    st.stop()
            elif "gpt" in current_model:
                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
                if not openai_key:
                    st.error("使用OpenAI模型需在下方輸入API金鑰 🔑")
                    st.stop()

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode", value=False)

        if "memory" not in st.session_state:
            st.session_state.memory = []

        if "conversation_initialized" not in st.session_state:
            openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key_input")
            if openai_api_key or gemini_api_key:
                client = initialize_client(openai_api_key)
                st.session_state.conversation_initialized = True
                st.session_state.messages = []
                debug_log("Conversation initialized with OpenAI client.")
            else:
                st.warning("⬅️ 請在側邊欄輸入OpenAI API金鑰以初始化聊天機器人")

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
            st.session_state.thinking_protocol = None
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

        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_image:
            st.session_state.uploaded_image = add_user_image(uploaded_image)
            debug_log(f"Uploaded image path: {st.session_state.uploaded_image}")

        st.subheader("🧠 Upload Thinking Protocol")
        uploaded_thinking_protocol = st.file_uploader("Choose a thinking_protocol.md file:", type=["md"], key="thinking_protocol_uploader")
        if uploaded_thinking_protocol:
            try:
                thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
                st.session_state.thinking_protocol = thinking_protocol_content
                append_message("user", thinking_protocol_content)
                st.success("Thinking Protocol uploaded successfully!")
                debug_log("Thinking Protocol uploaded and added to messages.")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading Thinking Protocol: {e}")
                debug_log(f"Error reading Thinking Protocol: {e}")

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location
        debug_log(f"Editor location set to: {st.session_state.editor_location}")

        with st.expander("🛠️ 调试与会话信息", expanded=False):
            if st.session_state.debug_mode:
                st.subheader("调试日志")
                if st.session_state.debug_logs:
                    debug_logs_combined = "\n".join(st.session_state.debug_logs)
                    st.text_area("Debug Logs", value=debug_logs_combined, height=200)
                else:
                    st.write("没有调试日志。")
                st.subheader("调试错误")
                if st.session_state.debug_errors:
                    debug_errors_combined = "\n".join(st.session_state.debug_errors)
                    st.text_area("Debug Errors", value=debug_errors_combined, height=200)
                else:
                    st.write("没有调试错误。")
            st.subheader("会话信息 (messages.json)")
            if "messages" in st.session_state:
                messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
                st.text_area("messages.json", value=messages_json, height=300)
                st.download_button(
                    label="📥 下载 messages.json",
                    data=messages_json,
                    file_name="messages.json",
                    mime="application/json"
                )
                st.markdown("---")
                if st.button("📄 显示原始消息"):
                    st.subheader("🔍 原始消息内容")
                    st.json(st.session_state.messages)
            else:
                st.write("没有找到 messages。")

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        st.image(image_url, caption="📷 上傳的圖片", use_column_width=True)
                        debug_log(f"Displaying image from {message['role']}: {image_url}")
                    else:
                        st.write(item)
                        debug_log(f"Displaying non-image content from {message['role']}: {item}")
            elif isinstance(message["content"], str) and "```python" in message["content"]:
                code_match = re.search(r'```python\s*(.*?)\s*```', message["content"], re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    st.code(code, language="python")
                    debug_log(f"Displaying code from {message['role']}: {code}")
                else:
                    st.write(message["content"])
                    debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")
            else:
                st.write(message["content"])
                debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        append_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)
            debug_log(f"User input added to messages: {user_input}")

        with st.spinner("Thinking..."):
            try:
                if openai_api_key or gemini_api_key:
                    client = initialize_client(openai_api_key)
                else:
                    raise ValueError("OpenAI API Key is not provided.")
                debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
                debug_log(f"Uploaded image path: {st.session_state.uploaded_image_path}")

                if not any(msg["role"] == "system" for msg in st.session_state.messages):
                    system_prompt = "You are an assistant that helps with data analysis."
                    append_message("system", system_prompt)
                    debug_log("System prompt added to messages.")

                if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                    prompt = user_input
                    debug_log("User input with image data already appended.")
                else:
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
    "content": "這是我的觀察跟分析: {{analysis}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# Read CSV file (use st.session_state.uploaded_file_path variable)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# Add your plotting or analysis logic here\\n\\n# For example, to display a plot using st.pyplot():\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)"
}}
Important:
1) 必須使用 st.session_state.uploaded_file_path 作為 CSV 路徑 (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)

Based on the request: {user_input}.
Available columns: {csv_columns}.
然後請使用繁體中文回應
"""
                        debug_log("Prompt constructed for CSV input with JSON response.")
                        append_message("system", prompt)
                        debug_log("System prompt appended to messages.")
                    else:
                        prompt = f"Please answer this question entirely in Traditional Chinese: {user_input}"
                        debug_log("Prompt constructed for plain text input.")
                        append_message("system", prompt)
                        debug_log("Plain text system prompt appended to messages.")

                model_params = {
                    "model": selected_model,
                    "temperature": 0.5,
                    "max_tokens": 4096
                }
                response_content = get_llm_response(client, model_params)
                debug_log(f"Full assistant response: {response_content}")

                if response_content:
                    append_message("assistant", response_content)
                    with st.chat_message("assistant"):
                        st.write(response_content)
                        debug_log(f"Assistant response added to messages: {response_content}")

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

                    if st.session_state.deep_analysis_mode and code:
                        st.write("### [Deep Analysis] Automatically executing the generated code and sending the chart to GPT-4o for analysis...")
                        debug_log("Deep analysis mode activated.")
                        global_vars = {
                            "uploaded_file_path": st.session_state.uploaded_file_path,
                            "uploaded_image_path": st.session_state.uploaded_image_path,
                        }
                        exec_result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                        st.write("#### Execution Result")
                        st.text(exec_result)
                        debug_log(f"Execution result: {exec_result}")
                        fig = plt.gcf()
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        st.session_state.deep_analysis_image = chart_base64
                        debug_log("Chart has been converted to base64.")
                        prompt_2 = f"""基於圖片給我更多資訊"""
                        debug_log(f"Deep Analysis Prompt: {prompt_2}")
                        append_message("user", prompt_2)
                        debug_log("Deep analysis prompt appended to messages.")
                        image_content = [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{chart_base64}"}
                        }]
                        append_message("user", image_content)
                        second_raw_response = get_llm_response(client, model_params)
                        debug_log(f"Deep analysis response: {second_raw_response}")
                        if second_raw_response:
                            append_message("assistant", second_raw_response)
                            st.session_state.second_response = second_raw_response
                            with st.chat_message("assistant"):
                                st.write(second_raw_response)
                                debug_log(f"Deep analysis response added to messages: {second_raw_response}")
                            prompt_3 = f"""
First response content: {content}
Second response chart analysis content: {second_raw_response}

請把前兩次的分析內容做分析總結，有數據的話就顯示得漂亮一點，主要是需要讓使用者感到很厲害。並且以繁體中文作為回答用的語言。
另外需要解釋傳給妳的圖表，以一個沒有資料科學背景的小白解釋我所傳的圖表。還有根據第二次的圖表分析得出的結論，直接預測之後的走向，例如:"之後這個數值的走向會呈現向上的趨勢"等...
不要跟使用者說甚麼妳可以使用RFM分析，交叉分析之類的方法。我需要妳直接預測之後的走向，比如往上還是往下。
"""
                            debug_log(f"Final Summary Prompt: {prompt_3}")
                            append_message("user", prompt_3)
                            debug_log("Final summary prompt appended to messages.")
                            third_raw_response = get_llm_response(client, model_params)
                            debug_log(f"Final summary response: {third_raw_response}")
                            if third_raw_response:
                                append_message("assistant", third_raw_response)
                                st.session_state.third_response = third_raw_response
                                with st.chat_message("assistant"):
                                    st.write(third_raw_response)
                                    debug_log(f"Final summary response added to messages: {third_raw_response}")
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

    # 新增：二模型交叉驗證按鈕
    if st.button("二模型交叉驗證"):
        if openai_api_key:
            client = initialize_client(openai_api_key)
        else:
            st.error("OpenAI API Key is required for cross validation.")
            st.stop()
    
        # 設定兩個模型的參數（可根據需要調整）
        model_params_openai = {
            "model": "gpt-4-turbo",
            "temperature": 0.5,
            "max_tokens": 4096
        }
        model_params_gemini = {
            "model": "gemini-1.5-flash",
            "temperature": 0.5,
            "max_tokens": 4096
        }
        cross_validated_response = get_cross_validated_response(client, model_params_openai, model_params_gemini)
        st.write("### OpenAI 回答")
        st.write(cross_validated_response["openai_response"])
        st.write("### Gemini 回答")
        st.write(cross_validated_response["gemini_response"])
    
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
