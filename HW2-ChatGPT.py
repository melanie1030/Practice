import streamlit as st
import requests
import pandas as pd
import json

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
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
    }
    .ai-bubble {
        background-color: #E8E8E8;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
    }
    .user-container, .ai-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .user-container img, .ai-container img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    .user-container {
        justify-content: flex-end;
    }
    .user-container img {
        margin-left: 10px;
    }
    .ai-container {
        justify-content: flex-start;
    }
    .ai-container img {
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Create a placeholder for chat messages
chat_placeholder = st.empty()

# Function to render messages
def render_messages():
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            if message["role"] == "system":
                continue  # Skip system messages
            elif message["role"] == "user":
                st.markdown(f"""
                <div class="user-container">
                    <div class="user-bubble">
                        {message['content']}
                    </div>
                    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEh6XGT5Hz9MpAiyfTHlBczavuUjyTBza9zWdzYmoifglj0p1lsylcTEScnpSa-Youh7YXw-ssgO-mMQmw-DBz4NeesioQPTe8beOH_QS-A4JMnfZAGP-01gxPQrS-pPEnrnJxbdVnWguhCC/s1600/pose_pien_uruuru_woman.png" alt="User">
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="ai-container">
                    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjCHBgyqLrwRdbSM72R9PutXIqxbI9yR5UzXWC0TYIYVlKgHH5TzkaHijRkdxQMRSJx8upcecs2RGHYW7gVOSQPH-LUrPUg3esbqx5-7Q04BPJWD-DdzTealzGBQehfXpDeLxYe29MjQQgo/s1600/megane_hikaru_woman.png" alt="AI">
                    <div class="ai-bubble">
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Display chat history with avatars
render_messages()

# Input box for the user's question at the bottom of the screen
user_input = st.chat_input("輸入訊息：")

# File uploader for uploading files or images
uploaded_file = st.file_uploader("上傳檔案或圖片：", type=["txt", "pdf", "png", "jpg", "jpeg", "csv", "json"])

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
        # Get file details
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        file_type = uploaded_file.type

        # Process the uploaded file based on its type
        if file_type.startswith('text'):
            # Text files
            if file_type == 'text/plain':
                # Handle plain text files
                file_content = uploaded_file.getvalue().decode("utf-8")
                st.session_state["messages"].append({"role": "user", "content": f"上傳的文本文件內容：\n{file_content}"})
            elif file_type == 'text/csv':
                # Handle CSV files
                try:
                    df = pd.read_csv(uploaded_file)
                    csv_content = df.to_csv(index=False)
                    st.session_state["messages"].append({"role": "user", "content": f"上傳的 CSV 文件內容：\n{csv_content}"})
                except Exception as e:
                    st.error(f"無法讀取 CSV 文件：{e}")
            elif file_type == 'application/json':
                # Handle JSON files
                try:
                    json_content = json.load(uploaded_file)
                    json_str = json.dumps(json_content, ensure_ascii=False, indent=2)
                    st.session_state["messages"].append({"role": "user", "content": f"上傳的 JSON 文件內容：\n{json_str}"})
                except Exception as e:
                    st.error(f"無法讀取 JSON 文件：{e}")
            else:
                st.error("不支持的文本文件類型")
        elif file_type in ["application/pdf"]:
            # Handle PDF files
            st.session_state["messages"].append({"role": "user", "content": f"上傳了一個 PDF 文件：{uploaded_file.name}"})
            # You may need to use a PDF parsing library like PyPDF2 to extract content
        elif file_type.startswith('image'):
            # Handle image files
            img_bytes = uploaded_file.getvalue()
            # Optionally display the image
            st.image(img_bytes, caption=uploaded_file.name)
            st.session_state["messages"].append({"role": "user", "content": f"上傳了一張圖片：{uploaded_file.name}"})
            # Note: Sending image data to the model may require special handling
        elif file_type == 'application/json':
            # Handle JSON files
            try:
                json_content = json.load(uploaded_file)
                json_str = json.dumps(json_content, ensure_ascii=False, indent=2)
                st.session_state["messages"].append({"role": "user", "content": f"上傳的 JSON 文件內容：\n{json_str}"})
            except Exception as e:
                st.error(f"無法讀取 JSON 文件：{e}")
        else:
            st.error("不支持的文件類型")

    # Re-render messages to display the user's message immediately
    render_messages()

    # Prepare the payload for the API request
    data = {
        "model": "gpt-4o-mini",
        "messages": st.session_state["messages"]
    }

    # Show a spinner while waiting for the AI's response
    with st.spinner("AI 正在回應..."):
        # Send the API request
        response = requests.post(api_url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json['choices'][0]['message']
