import streamlit as st  # 導入Streamlit用於建立Web應用界面
from openai import OpenAI  # 導入OpenAI API客戶端
import dotenv  # 用於加載環境變量
import os
from PIL import Image  # 用於圖像處理
from audio_recorder_streamlit import audio_recorder  # 用於錄音功能
import base64  # 用於Base64編碼/解碼
from io import BytesIO  # 用於處理二進制數據
import random  # 用於生成隨機數

# 加載環境變量（比如API密鑰）
dotenv.load_dotenv()

# 定義可用的OpenAI模型列表
openai_models = [
    "gpt-4o", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k", 
    "gpt-4", 
    "gpt-4-32k",
]

# 用於查詢和串流LLM（大型語言模型）回應的函數
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        client = OpenAI(api_key=api_key)  # 創建OpenAI客戶端實例
        # 使用串流模式獲取AI回應
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

# 將圖像轉換為Base64編碼的函數
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# 將文件轉換為Base64編碼的函數
def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

# 將Base64編碼轉換回圖像的函數
def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

# 主函數
def main():
    # --- 頁面配置 ---
    st.set_page_config(
        page_title="Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- 頁面標題 ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The Openai Chatbot</i> </h1>""")

    # --- 側邊欄設置 ---
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            # 獲取預設的OpenAI API密鑰（如果有的話）
            default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
            with st.popover("🔐 API_KEY"):
                openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", 
                                             value=default_openai_api_key, 
                                             type="password")
#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
    # --- 主要內容 ---
    # 檢查用戶是否輸入了API密鑰
    if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    
    else:
        client = OpenAI(api_key=openai_api_key)  # 創建OpenAI客戶端

        # 初始化聊天記錄
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 顯示之前的聊天記錄
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # --- 側邊欄模型選項和輸入 ---
        with st.sidebar:
            st.divider()
            
            # 顯示可用的模型列表
            available_models = [] + (openai_models if openai_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = None
            if model.startswith("gpt"): model_type = "openai"
            
            # 模型參數設置
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            # 音頻回應選項
            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", 
                                           ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            # 設置模型參數
            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            # 重置對話功能
            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button("🗑️", on_click=reset_conversation)

            st.divider()

            # --- 圖像上傳功能 ---
            if model in ["gpt-4o", "gpt-4-turbo"]:
                st.write(f"### **image{' or a video file' if model_type=='google' else ''}:**")

                # 添加圖像到訊息列表的函數
                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # 保存視頻文件
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            # 處理圖像文件
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                # 圖像上傳介面
                cols_img = st.columns(2)
                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

            # --- 音頻上傳功能 ---
            st.write("#")
            st.write(f"### **🎤{' Speech To Text' if model_type == 'openai' else ''}:**")

            # 音頻處理相關變量初始化
            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            # 音頻錄製介面
            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395")
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                if model_type != "google":
                    # 使用OpenAI的Whisper模型進行語音轉文字
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=("audio.wav", speech_input),
                    )
                    audio_prompt = transcript.text
                elif model_type == "google":
                    # 保存音頻文件
                    audio_id = random.randint(100000, 999999)
                    with open(f"audio_{audio_id}.wav", "wb") as f:
                        f.write(speech_input)
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "audio_file",
                                "audio_file": f"audio_{audio_id}.wav",
                            }]
                        }
                    )
                    audio_file_added = True

        # --- 聊天輸入處理 ---
        if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_f