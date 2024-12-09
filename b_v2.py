import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os

# --- Initialize and Settings ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response."""
    try:
        chart_type = response.get("chart_type", "line")
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

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

def save_to_pdf(conversation, chart_image=None):
    """Save conversation and chart as a PDF."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add title
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Conversation and Analysis", ln=True, align="C")
        pdf.ln(10)

        # Add conversation
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Conversation History:", ln=True, align="L")
        pdf.ln(5)
        for message in conversation:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            pdf.multi_cell(0, 10, txt=f"{role}: {content}")
            pdf.ln(5)

        # Add chart
        if chart_image:
            pdf.add_page()
            pdf.cell(200, 10, txt="Generated Chart:", ln=True, align="L")
            pdf.ln(5)
            pdf.image(chart_image, x=10, y=None, w=180)

        # Save as file
        file_name = "conversation_and_chart.pdf"
        pdf.output(file_name)
        return file_name
    except Exception as e:
        st.error(f"Failed to save PDF: {e}")
        return None

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory")
    
    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ Please enter the API key to initialize the chatbot.")
                return

        # Memory management buttons
        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

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
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner("Thinking..."):
            try:
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
                    Please respond with a JSON object in the format:
                    {{"chart_type": "line", "x_column": "{csv_data.columns[0]}", "y_column": "{csv_data.columns[1]}"}}
                    Based on the request: {user_input}.
                    Available columns: {csv_columns}.
                    """
                else:
                    prompt = user_input

                response = st.session_state.conversation.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})

                if csv_data is not None:
                    parsed_response = json.loads(response)
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

    if st.sidebar.button("📄 Save as PDF"):
        try:
            chart_image = None
            if "chart_buf" in locals() and chart_buf:
                chart_image_path = "temp_chart.png"
                with open(chart_image_path, "wb") as f:
                    f.write(chart_buf.getvalue())
                chart_image = chart_image_path

            pdf_file = save_to_pdf(st.session_state.messages, chart_image)
            if pdf_file:
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                st.sidebar.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=pdf_file,
                    mime="application/pdf"
                )
                st.success("PDF saved successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
