import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os
from io import BytesIO
import json
from PIL import Image
from datetime import datetime

# --- Initialize OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY", "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")  # 替换为你的 API 密钥

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
    """Parse GPT response to extract message and chart data."""
    try:
        response_json = json.loads(response_content)
        return response_json.get("message"), response_json.get("chart_data")
    except json.JSONDecodeError:
        return response_content, None


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
    st.set_page_config(page_title="Chatbot with Charts", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Chart Generation")

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

        # Download Chat History
        st.subheader("💾 Export Chat History")
        if st.button("Save Chat History"):
            chat_history = st.session_state.get("messages", [])
            save_chat_to_json(chat_history)

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # Add user input to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Immediately display user input
        with st.chat_message("user"):
            st.write(user_input)

        # Call GPT and display its response
        with st.spinner("Thinking..."):
            try:
                # Modify prompt for JSON + general response
                prompt = f"""
                Respond to the following user query in two parts:
                1. Provide a helpful answer to the user's question.
                2. If relevant, include chart details in JSON format for generating a chart. 
                The JSON format should look like this:
                {{
                    "chart_type": "bar",
                    "x_column": "Date",
                    "y_column": "Sales"
                }}
                
                User query: {user_input}
                """
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0.7
                )

                gpt_response = response["choices"][0]["text"]

                # Parse GPT response for general message and chart data
                message, chart_data = parse_gpt_response(gpt_response)

                # Add GPT response to conversation history
                st.session_state.messages.append({"role": "assistant", "content": message})

                # Display GPT response
                with st.chat_message("assistant"):
                    st.write(message)

                # Generate chart if chart_data is provided
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
