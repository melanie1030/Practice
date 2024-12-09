import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import dotenv
import os
from io import BytesIO
import json
from PIL import Image
from datetime import datetime
from langchain.memory import ConversationBufferMemory  # 新增
from langchain.llms import OpenAI as LangChainOpenAI  # 新增
from langchain.chains import ConversationChain  # 新增

# --- Initialize and Settings ---
dotenv.load_dotenv()

# Define global variables
OPENAI_MODELS = ["gpt-4-turbo", "gpt-3.5-turbo"]

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return LangChainOpenAI(temperature=0.7, openai_api_key=api_key) if api_key else None

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response."""
    try:
        chart_type = response.get("chart_type", "line")  # Default to line chart
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])

        plt.figure(figsize=(10, 6))

        if chart_type == "line":
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
        elif chart_type == "bar":
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
        elif chart_type == "scatter":
            plt.scatter(csv_data[x_column], csv_data[y_column], alpha=0.7, edgecolors='b')
        elif chart_type == "box":  # 新增盒鬚圖支持
            if y_column in csv_data.columns:
                plt.boxplot(csv_data[y_column], vert=True, patch_artist=True)
                plt.xticks([1], [y_column])  # 只顯示 y 軸欄位名稱
            else:
                raise ValueError("Boxplot requires a valid column for Y-axis.")

        plt.title(f"{y_column} vs {x_column} ({chart_type.capitalize()} Chart)", fontsize=16)
        plt.xlabel(x_column if chart_type != "box" else "", fontsize=14)
        plt.ylabel(y_column, fontsize=14)
        plt.grid(True)

        # Save chart to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

def save_chat_to_json(messages):
    """Save chat history as a JSON file."""
    try:
        chat_json = json.dumps(messages, ensure_ascii=False, indent=4)
        json_bytes = BytesIO(chat_json.encode("utf-8"))
        json_bytes.seek(0)
        file_name = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            label="Download Chat History",
            data=json_bytes,
            file_name=file_name,
            mime="application/json",
        )
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + Memory + Data Analysis")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        if not api_key:
            st.warning("⬅️ Please enter the API key to continue...")
            return

        # Initialize memory and conversation chain
        memory = ConversationBufferMemory()
        client = initialize_client(api_key)
        conversation = ConversationChain(llm=client, memory=memory)

        # Upload CSV
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

        # Upload Image
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Download Chat History
        st.subheader("💾 Export Chat History")
        if st.button("Save Chat History"):
            save_chat_to_json(memory.chat_memory.messages)

    # --- Chat Interface ---
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # Process user input through conversation chain
        with st.spinner("Thinking..."):
            try:
                response = conversation.run(input=user_input)
                st.chat_message("user").write(user_input)
                st.chat_message("assistant").write(response)

                # Generate chart if CSV is uploaded
                if csv_data is not None:
                    parsed_response = json.loads(response)  # Validate JSON format
                    chart_buf = generate_image_from_gpt_response(parsed_response, csv_data)
                    if chart_buf:
                        st.image(chart_buf, caption="Generated Chart", use_column_width=True)
                        st.download_button(
                            label="Download Chart",
                            data=chart_buf,
                            file_name="generated_chart.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
