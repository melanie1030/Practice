import streamlit as st
import pandas as pd
import subprocess
import json

# 使用您的 OpenAI API 金鑰
OPENAI_API_KEY = "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"

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
            # 建立 JSON 資料
            messages_json = json.dumps(st.session_state["messages"])

            # 構建 curl 命令
            curl_command = [
                "curl", "https://api.openai.com/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {OPENAI_API_KEY}",
                "-d", json.dumps({
                    "model": "gpt-3.5-turbo",
                    "messages": st.session_state["messages"]
                })
            ]

            # 使用 subprocess 執行 curl 命令
            result = subprocess.run(curl_command, capture_output=True, text=True)

            # 解析回應
            response = json.loads(result.stdout)

            # 顯示完整的 JSON 回應
            st.markdown("### 完整回應 JSON：")
            st.json(response)

            # 從回應中擷取 AI 的訊息內容
            full_response = response['choices'][0]['message']['content']

            # 儲存並顯示 AI 回應
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

            with st.chat_message("assistant"):
                st.markdown(full_response)

        except Exception as e:
            st.error(f"發生錯誤：{e}")

    render_messages()
