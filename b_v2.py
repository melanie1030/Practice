import streamlit as st
import json  # 導入 json 模塊
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO
from openai import OpenAI  # 確保這是正確的 OpenAI 類
from PIL import Image
from streamlit_ace import st_ace
import time

# --- 初始化與設置 ---
dotenv.load_dotenv()

# 定義全域變數
OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o"  # Retain if necessary
]

MAX_MESSAGES = 10  # Limit message history

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.write(f"**DEBUG LOG:** {msg}")
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.error(f"**DEBUG ERROR:** {msg}")
        print(msg)

def load_image_base64(image, max_size=(800, 800)):
    """Resize image if necessary and convert to Base64 encoding."""
    try:
        image.thumbnail(max_size, Image.ANTIALIAS)  # Resize to max_size
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

def add_user_image(image):
    """Add an image message to the session state as a Markdown string."""
    img_base64 = load_image_base64(image)
    if img_base64:
        image_markdown = f"![Uploaded Image](data:image/png;base64,{img_base64})"
        append_message("user", image_markdown)
        st.success("圖像已上傳!")
    else:
        debug_error("Failed to convert image to base64.")

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
        st.success("對話已清除！")
        debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.code(code, language="python")

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

def stream_llm_response(client, model_params, max_retries=3):
    """Stream responses from the LLM model with retry logic."""
    retries = 0
    wait_time = 5  # Start with 5 seconds

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_params.get("model", "gpt-4-turbo"),
                messages=st.session_state.messages,
                temperature=model_params.get("temperature", 0.3),
                max_tokens=model_params.get("max_tokens", 4096),
                stream=True
            )
            response_content = ""
            assistant_placeholder = st.empty()  # Create a placeholder for assistant's response

            for chunk in response:
                chunk_text = chunk.choices[0].delta.get('content', '')
                if chunk_text:
                    response_content += chunk_text
                    # Update the assistant's response in the placeholder
                    assistant_placeholder.markdown(response_content)
                    debug_log(f"Received chunk: {chunk_text[:100]}...")
            return response_content

        except Exception as e:
            if 'rate_limit_exceeded' in str(e).lower() or '429' in str(e):
                debug_error(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2  # Exponential backoff
            else:
                debug_error(f"Error streaming response: {e}")
                st.error(f"An error occurred while streaming the response: {e}")
                return ""
    
    st.error("Max retries exceeded. Please try again later.")
    return ""

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="聊天機器人 + 數據分析", page_icon="🤖", layout="wide")
    st.title("🤖 聊天機器人 + 📊 數據分析 + 🧠 記憶 + 🖋️ 編輯器 (含調試與深度分析)")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state:
        st.session_state.deep_analysis_mode = False

    with st.sidebar:
        st.subheader("🔐 輸入您的API密鑰")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API密鑰", value=default_api_key, type="password")

        selected_model = st.selectbox("選擇模型:", OPENAI_MODELS, index=0)

        st.session_state.debug_mode = st.checkbox("調試模式", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("深度分析模式", value=False)

        if not api_key:
            st.warning("⬅️ 請輸入您的API密鑰以繼續...")
            st.stop()

        client = initialize_client(api_key)
        if not client:
            st.error("無法初始化 OpenAI 客戶端。請檢查您的 API 密鑰。")
            st.stop()

        if st.session_state.debug_mode:
            debug_log(f"目前使用的模型: {selected_model}")

        # Image Upload and Capture
        st.subheader("📂 上傳圖像或拍照")
        
        # Image Upload
        uploaded_img = st.file_uploader("選擇一張圖片:", type=["png", "jpg", "jpeg"], key="uploaded_img")
        if uploaded_img:
            img = Image.open(uploaded_img)
            add_user_image(img)
            st.success("圖像已上傳!")

        # Camera Capture
        camera_img = st.camera_input("拍照", key="camera_img")
        if camera_img:
            img = Image.open(camera_img)
            add_user_image(img)
            st.success("拍照已成功!")

        # Reset Conversation
        st.button("🗑️ 清除對話", on_click=reset_session_messages)

    # --- Display Message History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            else:
                # Fallback for unexpected content types
                st.write(message["content"])
                debug_error("Unexpected content type in messages.")

    # --- User Input ---
    prompt = st.chat_input("嗨！問我任何問題...")
    if prompt:
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
            debug_log(f"用戶輸入: {prompt}")

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, {
                "model": selected_model,
                "temperature": st.slider("溫度", 0.0, 2.0, 0.3),
                "max_tokens": 4096
            }))

    # --- Persistent Code Editor ---
    if st.session_state.deep_analysis_mode:
        st.subheader("🖋️ 持久化編輯器 (深度分析模式)")
        with st.expander("編輯生成的代碼", expanded=True):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code
                debug_log("ace_code 已更新自編輯器。")

            if st.button("▶️ 執行代碼"):
                global_vars = {
                    "uploaded_file_path": st.session_state.get("uploaded_file_path"),
                    "uploaded_image_path": st.session_state.get("uploaded_image_path"),
                }
                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### 執行結果")
                st.text(result)
                debug_log(f"代碼執行結果: {result}")

    # --- 添加調試區塊 ---
    with st.expander("🛠️ 調試: 查看 session_state.messages", expanded=False):
        if "messages" in st.session_state:
            messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
            st.text_area("messages.json", value=messages_json, height=300)
        else:
            st.write("沒有找到 messages。")

# 程式入口點
if __name__ == "__main__":
    main()
