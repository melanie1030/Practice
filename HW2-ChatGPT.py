import streamlit as st
from openai import OpenAI
client = OpenAI()

# 使用您的 OpenAI API 金鑰
openai.api_key = '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'

# Streamlit App 標題
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~")

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]

def render_messages():
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            if message["role"] == "system":
                continue
            elif message["role"] == "user":
                st.markdown(f"**使用者:** {message['content']}")
            elif message["role"] == "assistant":
                st.markdown(f"**AI:** {message['content']}")

chat_placeholder = st.empty()
render_messages()

user_input = st.chat_input("輸入訊息：")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_messages()

    with st.spinner("AI 正在回應..."):
        try:
            # 使用 openai 庫發送請求
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state["messages"]
            )

            # 從回應中取得內容
            full_response = response['choices'][0]['message']['content']
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

            with st.chat_message("assistant"):
                st.markdown(full_response)

        except Exception as e:
            st.error(f"發生錯誤：{e}")

    render_messages()
