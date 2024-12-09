import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import ConversationChain
import dotenv
import os
from io import BytesIO
import json
import re
from PIL import Image
from datetime import datetime

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize LangChain's OpenAI client with the provided API key."""
    return LangChainOpenAI(temperature=0.7, openai_api_key=api_key) if api_key else None

def extract_json(response_text):
    """Extract JSON from GPT response."""
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No valid JSON found in the response.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")

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

        # Save chart to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

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

        memory = ConversationBufferMemory()
        client = initialize_client(api_key)
        conversation = ConversationChain(llm=client, memory=memory)

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call GPT
        with st.spinner("Thinking..."):
            try:
                response = conversation.run(input=user_input)
                try:
                    parsed_response = extract_json(response)
                except ValueError:
                    st.error("Failed to parse GPT response as JSON.")
                    st.text_area("Raw GPT Response", response)
                    return

                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(parsed_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
