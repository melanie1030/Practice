from openai import OpenAI
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

# Custom CSS for chat bubble styles (same as before)...

# Function to render messages
def render_messages():
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            if message["role"] == "system":
                continue
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
chat_placeholder = st.empty()
render_messages()

# File uploader section (above the input box)
uploaded_file = st.file_uploader("上傳檔案", type=["csv", "xlsx", "txt", "pdf", "jpg", "png", "jpeg"])

# Handle file upload
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state["messages"].append(
                {"role": "user", "content": f"已上傳 CSV 檔案：{uploaded_file.name}，以下是內容：\n{df.to_string(index=False)}"}
            )
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            st.session_state["messages"].append(
                {"role": "user", "content": f"已上傳 Excel 檔案：{uploaded_file.name}，以下是內容：\n{df.to_string(index=False)}"}
            )
        else:
            file_info = f"已上傳檔案：{uploaded_file.name} (大小：{uploaded_file.size} bytes)"
            st.session_state["messages"].append({"role": "user", "content": file_info})

    except Exception as e:
        st.error(f"無法讀取檔案：{e}")

    render_messages()

# Layout with input box and send button (below the file uploader)
user_input = st.chat_input("輸入訊息：")

# API key and URL configuration
api_key = st.secrets["api_key"]
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Handle user input
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_messages()

    data = {"model": "gpt-4O", "messages": st.session_state["messages"]}

    with st.spinner("AI 正在回應..."):
        try:
            response = requests.post(headers=headers, json=data)
            response.raise_for_status()

            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            st.error(f"HTTP 錯誤: {e}")
        except ValueError:
            st.error("回應不是有效的 JSON 格式")
            st.write("伺服器回應內容：", response.text)

    render_messages()
