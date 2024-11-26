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

dotenv.load_dotenv()

OPENAI_MODELS = ["gpt-4-turbo", "gpt-3.5-turbo"]

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

# --- Helper Functions ---
def plot_line_chart(data, x_column, y_column):
    """生成折線圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_bar_chart(data, x_column, y_column):
    """生成柱狀圖"""
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color='skyblue')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.tight_layout()
    return plt

def plot_scatter_chart(data, x_column, y_column):
    """生成散點圖"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], color='purple')
    plt.title(f"{y_column} vs {x_column} (Scatter Plot)", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_pie_chart(data, column):
    """生成餅圖"""
    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f"Distribution of {column}", fontsize=16)
    plt.ylabel('')
    plt.tight_layout()
    return plt

def generate_summary(data):
    """生成數據摘要"""
    summary = data.describe(include='all').transpose()
    return summary

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response."""
    try:
        chart_type = response.get("chart_type", "line")  # Default to line chart
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])
        fig = None

        if chart_type == "line":
            fig = plot_line_chart(csv_data, x_column, y_column)
        elif chart_type == "bar":
            fig = plot_bar_chart(csv_data, x_column, y_column)
        elif chart_type == "scatter":
            fig = plot_scatter_chart(csv_data, x_column, y_column)
        elif chart_type == "pie":
            fig = plot_pie_chart(csv_data, x_column)

        # Save chart to buffer
        buf = BytesIO()
        fig.savefig(buf, format="png")
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

# --- Main Function ---
def main():
    st.set_page_config(page_title="Chatbot with Data & Images", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🖼️ Image Upload")

    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        client = initialize_client(api_key)
        if not api_key or not client:
            st.warning("⬅️ Please enter the API key to continue...")
            return

        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

        st.subheader("💾 Export Chat History")
        if st.button("Save Chat History"):
            save_chat_to_json(st.session_state.messages)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            try:
                prompt = user_input
                if csv_data is not None:
                    prompt = (
                        f"You are a data analyst. Analyze the following dataset and provide insights or "
                        f"chart suggestions. Dataset preview:\n{csv_data.head(5).to_json()}.\n"
                        f"User question: {user_input}"
                    )

                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                if hasattr(response, "choices") and len(response.choices) > 0:
                    gpt_reply = response.choices[0].message.content
                else:
                    raise ValueError("Invalid GPT response structure.")

                st.session_state.messages.append({"role": "assistant", "content": gpt_reply})
                with st.chat_message("assistant"):
                    st.write(gpt_reply)

                if csv_data is not None:
                    parsed_response = json.loads(gpt_reply)
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
