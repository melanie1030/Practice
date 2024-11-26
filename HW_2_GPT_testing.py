import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import dotenv
import os
from io import BytesIO
import json
from PIL import Image
from datetime import datetime

# --- Initialize and Settings ---
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        st.error("Failed to parse GPT response. Ensure it contains valid JSON.")
        return response_content, None


def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot with Charts", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Chart Generation")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        if not api_key:
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
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}]
                )

                gpt_response = response["choices"][0]["message"]["content"]

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
