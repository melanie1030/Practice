import streamlit as st
import requests
import json

# 使用您的 OpenAI API 金鑰
OPENAI_API_KEY = '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'

# Streamlit App 標題
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~")

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
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
            # 建立 headers 和 data
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }

            data = {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": st.session_state["messages"],
            }

            # 發送 POST 請求到 OpenAI API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

            # 檢查請求是否成功
            if response.status_code == 200:
                full_response = response.json()["choices"][0]["message"]["content"]
                st.session_state["messages"].append({"role": "assistant", "content": full_response})

                with st.chat_message("assistant"):
                    st.markdown(full_response)
            else:
                st.error(f"請求失敗：{response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"發生錯誤：{e}")

    render_messages()
