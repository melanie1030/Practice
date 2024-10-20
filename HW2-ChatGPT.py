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
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #E8E8E8;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
        float: left;
        clear: both;
    }
    .user-container, .ai-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
        clear: both;
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
        order: 2;
    }
    .ai-container img {
        margin-right: 10px;
        order: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Display chat history with avatars
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-container">
            <div class="user-bubble">
                {message['content']}
            </div>
            <img src="https://i.imgur.com/7S7oETi.png" alt="User">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-container">
            <img src="https://i.imgur.com/nGF1K8f.png" alt="AI">
            <div class="ai-bubble">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Input box for the user's question at the bottom of the screen
user_input = st.chat_input("輸入訊息：")

# Your API key (read securely from Streamlit secrets)
api_key = st.secrets["api_key"]
api_url = "https://api.chatanywhere.tech/v1/chat/completions"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# When the user submits a message
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
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
