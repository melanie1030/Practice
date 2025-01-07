import streamlit as st
import openai
import dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import base64
from io import BytesIO
from PIL import Image

# --- 初始化與設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4", 
    "gpt-3.5-turbo",
]

# --- 輔助函數 ---

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return its path."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def execute_code(code, global_vars=None):
    """Execute Python code dynamically."""
    try:
        exec_globals = global_vars if global_vars else {}
        exec(code, exec_globals)
        return "Code executed successfully."
    except Exception as e:
        return f"Error executing code:\n{traceback.format_exc()}"

def stream_llm_response(model, messages, temperature=0.3, max_tokens=1000):
    """Stream responses from the OpenAI ChatCompletion API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in response:
            chunk_text = chunk.choices[0].delta.get("content", "")
            yield chunk_text
    except Exception as e:
        yield f"Error: {e}"

def reset_session_state():
    """Reset session state variables."""
    for key in [
        "messages", "uploaded_file_path", "uploaded_image_path", 
        "image_base64", "debug_mode", "deep_analysis_mode", 
        "ace_code"
    ]:
        st.session_state.pop(key, None)

def debug_log(message):
    """Log debug messages when debug mode is enabled."""
    if st.session_state.get("debug_mode", False):
        st.write(message)
        print(message)

# --- 主功能 ---
def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + Data Analysis + Deep Analysis + Debug Mode")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state:
        st.session_state.deep_analysis_mode = False
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""

    # 側邊欄
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("⬅️ Please enter your API Key to continue.")
            return
        openai.api_key = api_key

        # 模型選擇
        model = st.selectbox("Select Model:", OPENAI_MODELS)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

        # Debug模式
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode")
        st.session_state.deep_analysis_mode = st.checkbox("Enable Deep Analysis Mode")

        # 重置對話
        if st.button("🗑️ Clear Conversation"):
            reset_session_state()

        # CSV上傳
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### CSV Preview")
                st.dataframe(csv_data)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        # 圖片上傳
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            st.session_state.uploaded_image_path = save_uploaded_file(uploaded_image)
            st.image(st.session_state.uploaded_image_path, caption="Uploaded Image", use_column_width=True)

    # 顯示聊天歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用戶輸入
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # 將用戶消息添加到對話歷史
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant 回應
        with st.chat_message("assistant"):
            response_text = ""
            for chunk in stream_llm_response(model, st.session_state.messages, temperature=temperature):
                response_text += chunk
                st.markdown(response_text)

            # 儲存回應到聊天歷史
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # 深度分析模式
        if st.session_state.deep_analysis_mode and st.session_state.uploaded_file_path:
            with st.spinner("Performing deep analysis..."):
                try:
                    # CSV深度分析處理
                    csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                    fig, ax = plt.subplots()
                    csv_data.hist(ax=ax)  # 簡單的示例
                    st.pyplot(fig)

                    # 提取圖像並進行Base64編碼
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                    st.image(buf, caption="Generated Analysis Chart")

                    # 發送圖像數據進行分析
                    messages = st.session_state.messages + [{
                        "role": "user",
                        "content": f"Here is a chart: data:image/png;base64,{img_base64}"
                    }]
                    analysis_response = ""
                    for chunk in stream_llm_response(model, messages, temperature=temperature):
                        analysis_response += chunk
                        st.markdown(analysis_response)

                except Exception as e:
                    st.error(f"Error in deep analysis: {e}")

# --- 主程式 ---
if __name__ == "__main__":
    main()
