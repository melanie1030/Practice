import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import dotenv
import os
from io import BytesIO
import json
from datetime import datetime

# --- Initialize and Settings ---
dotenv.load_dotenv()

# Define global variables
OPENAI_MODELS = ["gpt-4-turbo", "gpt-3.5-turbo"]


def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None


def generate_image_from_gpt_response(response, csv_data):
    """根据 GPT 的回复生成图表."""
    try:
        # 解析 GPT 回复中的关键词生成图表
        chart_type = response.get("chart_type", "line")  # 默认折线图
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])

        if chart_type == "line":
            plt.figure(figsize=(10, 6))
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
            plt.title(f"{y_column} vs {x_column}", fontsize=16)
            plt.xlabel(x_column, fontsize=14)
            plt.ylabel(y_column, fontsize=14)
            plt.grid(True)
        elif chart_type == "bar":
            plt.figure(figsize=(10, 6))
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
            plt.title(f"{y_column} vs {x_column}", fontsize=16)
            plt.xlabel(x_column, fontsize=14)
            plt.ylabel(y_column, fontsize=14)

        # 保存图表并返回
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"生成图片失败: {e}")
        return None


def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot + CSV", page_icon="🤖", layout="centered")
    st.title("📊 Chatbot with CSV Analysis")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        client = initialize_client(api_key)
        if not api_key or not client:
            st.warning("⬅️ Please enter the API key to continue...")
            return

        # Upload CSV
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call GPT
        with st.spinner("GPT is thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": user_input}]
                )

                # Debug: Output full GPT response
                st.write("### Debug: Full GPT Response")
                st.json(response)

                # Extract GPT reply
                if "choices" in response and response.choices:
                    gpt_reply = response.choices[0].message.get("content", "No content available.")
                else:
                    raise ValueError("Invalid response structure: 'choices' not found.")

                # Append GPT response to chat
                st.session_state.messages.append({"role": "assistant", "content": gpt_reply})
                with st.chat_message("assistant"):
                    st.write(gpt_reply)

                # Generate chart if CSV is uploaded
                if csv_data is not None:
                    with st.spinner("Generating chart from GPT response..."):
                        try:
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
                        except json.JSONDecodeError:
                            st.error("Failed to parse GPT response. Ensure the response is in JSON format.")

            except Exception as e:
                # Output error details
                st.error(f"An error occurred while processing GPT response: {e}")


if __name__ == "__main__":
    main()
