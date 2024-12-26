import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
import io

from PIL import Image  # 新增圖片處理用
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from streamlit_ace import st_ace

# --- Initialize and Settings ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4o",  # 假設可解析圖片的實驗模型
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k"
]

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.write(msg)
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.error(msg)
        print(msg)

def initialize_client(api_key, model_name):
    return ChatOpenAI(
        model=model_name,
        temperature=0.5,
        openai_api_key=api_key
    ) if api_key else None

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    debug_log(f"DEBUG: saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    debug_log(f"DEBUG: files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

# --- 新增：Base64 與 PIL 處理 ---
def load_image_base64(image_path):
    """Convert an image to Base64 encoding using BytesIO + PIL."""
    img = Image.open(image_path)
    buffer = io.BytesIO()
    fmt = img.format if img.format else "PNG"
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_uploaded_image(image_file):
    """Save uploaded image to the specified directory and return its path."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())
    return file_path

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("DEBUG: Ready to exec the following code:")
        if st.session_state.get("debug_mode", False):
            st.code(code, language="python")

        debug_log("[DEBUG] Exec code with global_vars: " + str(list(exec_globals.keys())))
        exec(code, exec_globals)
        return "Code executed successfully. Output: " + str(exec_globals.get("output", "(No output returned)"))
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log("[DEBUG] Execution error: " + error_msg)
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode)."

def extract_json_block(response: str) -> str:
    pattern = r'```(?:json)?(.*)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json_str
    else:
        return response.strip()

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug & Deep Analysis)")

    # 初始化狀態
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    # 側邊欄設置
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        selected_model = st.selectbox("選擇模型:", OPENAI_MODELS, index=0)

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)

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

        if st.session_state.debug_mode:
            debug_log(f"DEBUG: Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.ace_code = ""
            st.session_state.uploaded_file_path = None
            st.session_state.uploaded_image_path = None
            st.session_state.image_base64 = None
            st.success("Memory cleared!")

    # --- 圖片上傳與處理 ---
    st.subheader("🖼️ Upload or Capture an Image")

    # 圖像上傳
    uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        file_path = save_uploaded_image(uploaded_image)
        st.session_state.uploaded_image_path = file_path

        # 使用 PIL 和 Base64 進行處理
        st.session_state.image_base64 = load_image_base64(file_path)
        st.image(file_path, caption="Uploaded Image Preview", use_column_width=True)
        st.success("Image uploaded and processed successfully!")
        debug_log(f"Image uploaded and saved to: {file_path}")

    # 拍照
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        file_path = save_uploaded_image(camera_image)
        st.session_state.uploaded_image_path = file_path

        # 使用 PIL 和 Base64 進行處理
        st.session_state.image_base64 = load_image_base64(file_path)
        st.image(file_path, caption="Captured Image Preview", use_column_width=True)
        st.success("Photo captured and processed successfully!")
        debug_log(f"Photo captured and saved to: {file_path}")

    # --- 顯示歷史訊息 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "code" in message:
                st.code(message["code"], language="python")

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                prompt = f"User input: {user_input}\nBase64 Image: {st.session_state.image_base64}" if st.session_state.image_base64 else user_input
                raw_response = st.session_state.conversation.run(prompt)
                response_json = extract_json_block(raw_response)

                st.session_state.messages.append({"role": "assistant", "content": response_json})
                with st.chat_message("assistant"):
                    st.write(response_json)
            except Exception as e:
                debug_error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
