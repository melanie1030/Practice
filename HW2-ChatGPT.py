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
from openai import OpenAI  # Custom OpenAI class as per your requirement
from PIL import Image
from streamlit_ace import st_ace

# --- 初始化與設置 ---
dotenv.load_dotenv()

# 定義全域變數
OPENAI_MODELS = ["gpt-4o"]

# --- 輔助函數 ---

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def load_image_base64(image):
    """Convert an image to Base64 encoding."""
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def add_user_image(image):
    """Add an image message to the session state."""
    img_base64 = load_image_base64(image)
    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]
    })

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")

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

        # 圖像上傳和拍照功能
        st.write("### 上傳圖像或拍照")
        
        # 圖像上傳
        uploaded_img = st.file_uploader("選擇一張圖片:", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img)
            add_user_image(img)
            st.success("圖像已上傳!")

        # 相機拍照
        camera_img = st.camera_input("拍照")
        if camera_img:
            img = Image.open(camera_img)
            add_user_image(img)
            st.success("拍照已成功!")

        # 重置對話
        st.button("🗑️ 清除對話", on_click=reset_session_messages)

    # --- 聊天窗口與訊息 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content.get("text", ""))
                elif content["type"] == "image_url":
                    st.image(content["image_url"].get("url", ""))

    # --- 用戶輸入 ---
    prompt = st.chat_input("嗨！問我任何問題...")
    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

# 程式入口點
if __name__ == "__main__":
    main()
