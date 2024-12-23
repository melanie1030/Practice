import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
from PIL import Image
from fpdf import FPDF
from datetime import datetime
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os
import time

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def display_code_line_by_line_in_block(code_snippet):
    """Display code one line at a time within a single code block."""
    displayed_code = ""
    code_lines = code_snippet.split("\n")
    code_placeholder = st.empty()
    for line in code_lines:
        if line.strip():
            displayed_code += line + "\n"
            code_placeholder.code(displayed_code, language="python")
            time.sleep(0.2)

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response."""
    try:
        chart_type = response.get("chart_type", "line")
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])

        # 新增功能：逐行顯示程式碼在同一區塊內
        code_snippet = f"""
        plt.figure(figsize=(10, 6))
        if chart_type == "line":
            plt.plot(csv_data['{x_column}'], csv_data['{y_column}'], marker='o')
        elif chart_type == "bar":
            plt.bar(csv_data['{x_column}'], csv_data['{y_column}'], color='skyblue')
        elif chart_type == "scatter":
            plt.scatter(csv_data['{x_column}'], csv_data['{y_column}'], alpha=0.7, edgecolors='b')
        elif chart_type == "box":
            plt.boxplot(csv_data['{y_column}'], vert=True, patch_artist=True)
            plt.xticks([1], ['{y_column}'])

        plt.title('{y_column} vs {x_column} ({chart_type.capitalize()} Chart)')
        plt.xlabel('{x_column}' if chart_type != 'box' else '')
        plt.ylabel('{y_column}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        """
        display_code_line_by_line_in_block(code_snippet)

        plt.figure(figsize=(10, 6))

        if chart_type == "line":
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
        elif chart_type == "bar":
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
        elif chart_type == "scatter":
            plt.scatter(csv_data[x_column], csv_data[y_column], alpha=0.7, edgecolors='b')
        elif chart_type == "box":
            if y_column in csv_data.columns:
                plt.boxplot(csv_data[y_column], vert=True, patch_artist=True)
                plt.xticks([1], [y_column])
            else:
                raise ValueError("Boxplot requires a valid column for Y-axis.")

        plt.title(f"{y_column} vs {x_column} ({chart_type.capitalize()} Chart)", fontsize=16)
        plt.xlabel(x_column if chart_type != "box" else "", fontsize=14)
        plt.ylabel(y_column, fontsize=14)
        plt.grid(True)

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
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory")

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = ChatOpenAI(model="gpt-4o", temperature=0.5, openai_api_key=api_key)
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

        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            img_bytes = BytesIO(uploaded_image.read())
            st.session_state.messages.append({"role": "user", "image": img_bytes.getvalue()})
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "image" in message:
                img = Image.open(BytesIO(message["image"]))
                st.image(img, caption="Uploaded Image", use_container_width=True)

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
                    Please respond with a JSON object in the format:
                    {{
                        "chart_type": "line", 
                        "x_column": "{csv_data.columns[0]}", 
                        "y_column": "{csv_data.columns[1]}",
                        "content": "根據 {csv_data.columns[0]} 和 {csv_data.columns[1]} 的數據分析，這是我的觀察：{{分析內容}}"
                    }}
                    Based on the request: {user_input}.
                    Available columns: {csv_columns}.
                    """
                else:
                    prompt = f"請全部以繁體中文回答此問題：{user_input}"

                response = st.session_state.conversation.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})

                with st.chat_message("assistant"):
                    try:
                        response_json = json.loads(response)
                        text_feedback = response_json.get("content", "這是我的分析：")
                        st.write(text_feedback)
                        chart_buf = generate_image_from_gpt_response(response_json, csv_data)
                        if chart_buf:
                            st.image(chart_buf, caption="Generated Chart", use_container_width=True)
                            st.download_button(
                                label="Download Chart",
                                data=chart_buf,
                                file_name="generated_chart.png",
                                mime="image/png"
                            )
                    except (json.JSONDecodeError, TypeError):
                        st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
