import streamlit as st
import requests
import base64  # 如果需要处理图像文件

# Title and description for the Streamlit app
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!!歡迎您問我答~")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]

# Custom CSS for chat bubble styles
st.markdown("""
    <!-- 您的 CSS 样式保持不变 -->
""", unsafe_allow_html=True)

# Create a placeholder for chat messages
chat_placeholder = st.empty()

# Function to render messages
def render_messages():
    # 您的渲染消息函数保持不变
    pass  # 为了简洁，这里省略具体内容

# Display chat history with avatars
render_messages()

# Input box for the user's question at the bottom of the screen
user_input = st.chat_input("輸入訊息：")

# File uploader for uploading files or images
uploaded_file = st.file_uploader("上傳檔案或圖片：", type=["txt", "pdf", "png", "jpg", "jpeg"])

# Your API key (read securely from Streamlit secrets)
api_key = st.secrets["api_key"]
api_url = "https://free.gpt.ge"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# When the user submits a message or uploads a file
if user_input or uploaded_file:
    if user_input:
        # Add the user's input to the session state messages
        st.session_state["messages"].append({"role": "user", "content": user_input})

    if uploaded_file:
        # Process the uploaded file
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        if uploaded_file.type.startswith('text'):
            # For text files
            file_content = uploaded_file.getvalue().decode("utf-8")
            # Add the file content to the conversation
            st.session_state["messages"].append({"role": "user", "content": f"上傳的文件內容：\n{file_content}"})
        elif uploaded_file.type in ["application/pdf"]:
            # For PDF files
            file_bytes = uploaded_file.read()
            # 这里您可能需要使用 PDF 解析库来读取内容，例如 PyPDF2
            st.session_state["messages"].append({"role": "user", "content": f"上傳了一個 PDF 文件：{uploaded_file.name}"})
        elif uploaded_file.type.startswith('image'):
            # For image files
            img_bytes = uploaded_file.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            # Add the image content to the conversation (可能需要根据您的 API 能力调整)
            st.session_state["messages"].append({"role": "user", "content": f"上傳的圖片內容（Base64編碼）：{img_base64}"})
        else:
            st.error("不支持的文件類型")

    # Re-render messages to display the user's message immediately
    render_messages()

    # Prepare the payload for the API request
    data = {
        "model": "gpt-4o-mini",  # 使用您的 gpt-4o-mini 模型
        "messages": st.session_state["messages"]
    }

    # Show a spinner while waiting for the AI's response
    with st.spinner("AI 正在回應..."):
        # Send the API request
        response = requests.post(api_url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']

            # Add the AI's response to the session state messages
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.status_code}, {response.text}")

    # Re-render messages to include the AI's response
    render_messages()
