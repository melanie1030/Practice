import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
from PIL import Image
import time
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    system_prompt = """
    You are an expert data analyst with years of experience in statistical analysis and data visualization.
    Always provide in-depth insights, explain correlations clearly, and suggest actionable insights.
    """
    return ChatOpenAI(model="gpt-4-turbo", temperature=0.7, max_tokens=1500, openai_api_key=api_key, system=system_prompt) if api_key else None

def generate_chart(chart_type, x_column, y_column, csv_data):
    """Generate a chart based on user input."""
    try:
        plt.figure(figsize=(10, 6))
        if chart_type == "line":
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
        elif chart_type == "bar":
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
        elif chart_type == "scatter":
            plt.scatter(csv_data[x_column], csv_data[y_column], alpha=0.7, edgecolors='b')
        elif chart_type == "box":
            plt.boxplot(csv_data[y_column], vert=True, patch_artist=True)
            plt.xticks([1], [y_column])
        plt.title(f"{y_column} vs {x_column} ({chart_type.capitalize()} Chart)", fontsize=16)
        plt.xlabel(x_column if chart_type != "box" else "", fontsize=14)
        plt.ylabel(y_column, fontsize=14)
        plt.grid(True)

        # Save the chart to a buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 🔢 Data Analysis + 🧠 Memory")

    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
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
                st.image(message["chart"], caption="Generated Chart", use_container_width=True)

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
                    summary = csv_data.describe().to_dict()
                    prompt = f"""
                    Based on the uploaded CSV file with columns: {csv_columns},
                    and the following statistical summary: {summary},
                    provide an in-depth analysis including:
                    1. Key statistical insights.
                    2. Correlations between features.
                    3. Identification of outliers.
                    4. Possible business applications or actionable insights.
                    """
                else:
                    prompt = f"Please respond to this question in Traditional Chinese: {user_input}"

                response = st.session_state.conversation.run(prompt)
                response_json = json.loads(response)

                # Display response content
                content = response_json.get("content", "Here is my analysis:")
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.write(content)

                # Generate chart
                if csv_data is not None:
                    chart_buf = generate_chart(
                        response_json.get("chart_type", "line"),
                        response_json.get("x_column", csv_data.columns[0]),
                        response_json.get("y_column", csv_data.columns[1]),
                        csv_data
                    )
                    if chart_buf:
                        st.session_state.messages.append({"role": "assistant", "chart": chart_buf})
                        with st.chat_message("assistant"):
                            st.image(chart_buf, caption="Generated Chart", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
