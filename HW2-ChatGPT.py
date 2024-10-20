import streamlit as st
import requests

# Title and description for the Streamlit app
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!!歡迎您問我答~")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.write(f"**請問：** {message['content']}")
    else:
        st.success(f"AI 回答：{message['content']}")

# Input box for the user's question
user_input = st.text_input("輸入訊息:")

# Your API key (read securely from Streamlit secrets)
api_key = st.secrets["api_key"]
api_url = "https://api.chatanywhere.tech/v1/chat/completions"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# When the user submits a message
if st.button("送出"):
    if user_input:
        # Add the user's input to the session state messages
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Prepare the payload for the API request
        data = {
            "model": "gpt-3.5-turbo",
            "messages": st.session_state["messages"]
        }

        # Send the API request
        response = requests.post(api_url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']

            # Add the AI's response to the session state messages
            st.session_state["messages"].append({"role": "assistant", "content": answer})

            # Display the updated conversation
            st.success(f"AI 回答：{answer}")
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    else:
        st.warning("請先輸入訊息！")
