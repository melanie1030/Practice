import streamlit as st
import requests

# Title and description for the Streamlit app
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!!歡迎您問我答~")

# Initialize session state for conversation history and response state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]
if "waiting_for_response" not in st.session_state:
    st.session_state["waiting_for_response"] = False

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

# Function to render messages
def render_messages():
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue  # Skip system messages
        elif message["role"] == "user":
            st.markdown(f"""
            <div class="user-container">
                <div class="user-bubble">
                    {message['content']}
                </div>
                <img src="https://i.imgur.com/7S7oETi.png" alt="User">
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="ai-container">
                <img src="https://i.imgur.com/nGF1K8f.png" alt="AI">
                <div class="ai-bubble">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Display chat history with avatars
render_messages()

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

# Function to send API request and get AI response
def get_ai_response():
    data = {
        "model": "gpt-3.5-turbo",
        "messages": st.session_state["messages"]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        response_json = response.json()
        answer = response_json['choices'][0]['message']['content']
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP Error: {err}")
    except Exception as err:
        st.error(f"Error: {err}")
    finally:
        st.session_state["waiting_for_response"] = False

# Handle user input and manage response state
if user_input and not st.session_state["waiting_for_response"]:
    # Add the user's input to the session state messages
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["waiting_for_response"] = True
    # Rerun the app to display the user's message immediately
    st.experimental_rerun()

if st.session_state.get("waiting_for_response"):
    with st.spinner("AI 正在回應..."):
        get_ai_response()
    # Rerun to display the AI's response
    st.experimental_rerun()

# Re-render messages after updates
render_messages()
