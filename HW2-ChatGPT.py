import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random

# --- 初始化與設置 ---
dotenv.load_dotenv()

# 定義全域變數
OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]

# --- 輔助函數 ---

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def load_image_base64(image):
    """Convert an image to Base64 encoding."""
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def file_to_base64(file_path):
    """Convert a file to Base64 encoding."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")

def add_user_message(content, message_type="text"):
    """Add a user message to the session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": [{"type": message_type, "text": content}]})

# --- Chatbot Main Functionality ---

def stream_llm_response(client, model_params):
    """Stream responses from the LLM model."""
    for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-4o"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=4096,
            stream=True):
        chunk_text = chunk.choices[0].delta.content or ""
        yield chunk_text

def main():
    # --- 頁面配置 ---
    st.set_page_config(page_title="聊天機器人", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>OpenAI 聊天機器人</i> </h1>""")

    # --- 側邊欄設置 ---
    with st.sidebar:
        st.subheader("🔐 輸入您的API密鑰")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API密鑰", value=default_api_key, type="password")

        client = initialize_client(api_key)
        if not api_key or not client:
            st.warning("⬅️ 請輸入API密鑰以繼續...")
            return

        # 模型設置
        model = st.selectbox("選擇模型:", OPENAI_MODELS)
        model_params = {"model": model, "temperature": st.slider("溫度", 0.0, 2.0, 0.3)}

        # 重置對話
        st.button("🗑️ 清除對話", on_click=reset_session_messages)

    # --- 聊天窗口與訊息 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Iterate over content to handle multiple types
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content.get("text", ""))
                elif content["type"] == "image_url":
                    st.image(content["image_url"].get("url", ""))
                elif content["type"] == "video_file":
                    st.video(content.get("video_file", ""))
                elif content["type"] == "audio_file":
                    st.audio(content.get("audio_file", ""))

    # --- 用戶輸入 ---
    prompt = st.chat_input("嗨！問我任何問題...")
    if prompt:
        add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

# 程式入口點
if __name__ == "__main__":
    main()

