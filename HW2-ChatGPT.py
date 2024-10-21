import streamlit as st
import pandas as pd
from openai import OpenAI

# 初始化 OpenAI 客戶端
OPENAI_API_KEY = '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'
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit 標題和副標題
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~")

# 初始化聊天記錄
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]

# 渲染訊息的函數
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

# 聊天訊息的占位符
chat_placeholder = st.empty()
render_messages()

# 上傳檔案區塊
uploaded_file = st.file_uploader("上傳檔案", type=["csv", "xlsx", "txt", "pdf", "jpg", "png", "jpeg"])

# 檔案處理邏輯
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

# 輸入區域（在檔案上傳區塊下方）
user_input = st.chat_input("輸入訊息：")

# 使用者輸入的處理
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_messages()

    # 呼叫 OpenAI API 生成回應
    with st.spinner("AI 正在回應..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4.0",  # 改為使用 GPT-4.0 模型
                messages=st.session_state["messages"]
            )
            answer = response.choices[0].message.content
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"API 錯誤：{e}")

    render_messages()
