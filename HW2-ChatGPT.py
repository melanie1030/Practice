import streamlit as st
import requests
import json

# Use Streamlit secrets for API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit App Title
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~今天是11/1")

# Initialize chat history
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
            # API request payload
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": st.session_state["messages"]
            }

            # Set up headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            # Send request using requests.post
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            # Parse JSON response
            response_json = response.json()

            # Display full JSON response for debugging
            st.markdown("### 完整回應 JSON：")
            st.json(response_json)

            # Extract and display assistant's response
            if 'choices' in response_json:
                full_response = response_json['choices'][0]['message']['content']
                st.session_state["messages"].append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    st.markdown(full_response)
            else:
                st.error(f"Unexpected response structure: {response_json}")

        except Exception as e:
            st.error(f"發生錯誤：{e}")

    render_messages()
