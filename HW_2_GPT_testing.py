import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os
from io import BytesIO
import json
from datetime import datetime

# --- Initialize OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-proj-JKTU3wx5jqUcf4RWBg0hlmKWxnqsPKaIo1K15myDybNb5iyzbHzbzGmVt5L1aN1fCE4B2c70YTT3BlbkFJKqK2r0Oh_-d5gJ8faKeQxxFT3B0lwIPNjABpQrsSf4MQoRdAIL67v_LdpOIagVWu1RDmMZWRgA")  # 替换为你的 API 密钥

def generate_image_from_json(chart_data, csv_data):
    """Generate a chart based on chart_data JSON and CSV."""
    try:
        chart_type = chart_data.get("chart_type", "line")  # Default to line chart
        x_column = chart_data.get("x_column", csv_data.columns[0])
        y_column = chart_data.get("y_column", csv_data.columns[1])

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

        plt.tight_layout()

        # Save chart to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None


def parse_gpt_response(response_content):
    """Parse GPT response to extract chart data."""
    try:
        response_json = json.loads(response_content)
        return response_json
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response. Ensure it contains valid JSON.")
        return None


def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot with Strict JSON Output", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot with Strict JSON Output for Chart Generation")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔐 API Key Configuration")
        st.write("Ensure your OpenAI API key is correctly configured in the environment or code.")

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

    # User input
    user_input = st.chat_input("Describe the chart you need (e.g., bar chart of sales over time):")
    if user_input:
        # Add user input to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Immediately display user input
        with st.chat_message("user"):
            st.write(user_input)

        # Call GPT and display its response
        with st.spinner("Generating JSON response..."):
            try:
                # GPT Prompt to enforce JSON-only response
                prompt = f"""
                Please respond only with a JSON object in the following format:
                {{
                    "chart_type": "bar",
                    "x_column": "Date",
                    "y_column": "Sales"
                }}
                Based on this user request: {user_input}.
                Do not include any additional text or explanation.
                """
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0
                )

                gpt_response = response["choices"][0]["text"].strip()

                # Parse GPT response
                chart_data = parse_gpt_response(gpt_response)

                if chart_data and csv_data is not None:
                    with st.spinner("Generating chart from JSON..."):
                        chart_buf = generate_image_from_json(chart_data, csv_data)
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
