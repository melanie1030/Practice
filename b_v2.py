import streamlit as st
import pandas as pd
import json
import time
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os
import pygwalker as pyg

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key) if api_key else None

def generate_chart_with_pygwalker(response, csv_data):
    """Generate a Pygwalker chart based on GPT's response and save it in session_state."""
    try:
        # Render interactive chart using Pygwalker
        st_pygwalker = pyg.walk(csv_data)

        # Convert to HTML and return for rendering
        return st_pygwalker.to_html()
    except Exception as e:
        st.error(f"Failed to generate the chart with Pygwalker: {e}")
        return None

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory")

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = initialize_client(api_key)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ Please enter the API key to initialize the chatbot.")
                return

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previously generated messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "code" in message:
                st.code(message["code"], language="python")
            if "chart" in message:
                st.components.v1.html(message["chart"], height=600, scrolling=True)

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
                    Please respond with a JSON object in the format:
                    {{
                        "chart_type": "pygwalker", 
                        "content": "根據 {csv_columns} 的數據分析，這是我的觀察：{{分析內容}}"
                    }}
                    Based on the request: {user_input}.
                    Available columns: {csv_columns}.
                    """
                else:
                    prompt = f"請全部以繁體中文回答此問題：{user_input}"

                response = st.session_state.conversation.run(prompt)
                response_json = json.loads(response)

                # Display response content
                content = response_json.get("content", "這是我的分析：")
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.write(content)

                # Generate Pygwalker chart
                if csv_data is not None:
                    chart_html = generate_chart_with_pygwalker(response_json, csv_data)
                    if chart_html:
                        st.session_state.messages.append({"role": "assistant", "chart": chart_html})
                        with st.chat_message("assistant"):
                            st.components.v1.html(chart_html, height=600, scrolling=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
