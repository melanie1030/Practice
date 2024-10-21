import streamlit as st
import requests
import pandas as pd

# Title and description for the Streamlit app
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~")

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

# Layout with input box, file uploader, and send button
col1, col2 = st.columns([4, 1])

# User input section with chat input
with col1:
    user_input = st.chat_input("輸入訊息：")

# File uploader section for CSV files
with col2:
    uploaded_file = st.file_uploader("上傳檔案", type=["csv", "txt", "pdf", "jpg", "png", "jpeg"])

# Your API key (read securely from Streamlit secrets)
api_key = st.secrets["api_key"]
api_url = "https://free.gpt.ge"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Handle user input
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_messages()

    data = {"model": "gpt-40-mini", "messages": st.session_state["messages"]}

    with st.spinner("AI 正在回應..."):
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.status_code}, {response.text}")

    render_messages()

# Handle file upload
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        # 讀取並顯示 CSV 檔案內容
        df = pd.read_csv(uploaded_file)
        st.session_state["messages"].append(
            {"role": "user", "content": f"已上傳 CSV 檔案：{uploaded_file.name}，以下是內容：\n{df.to_string(index=False)}"}
        )
    else:
        # 處理其他檔案類型
        file_info = f"已上傳檔案：{uploaded_file.name} (大小：{uploaded_file.size} bytes)"
        st.session_state["messages"].append({"role": "user", "content": file_info})

    render_messages()
