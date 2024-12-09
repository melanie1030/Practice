import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import json
from PIL import Image
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


# --- Helper Functions ---
def process_data(file):
    """Load and preprocess the uploaded ERP data."""
    try:
        if file.name.endswith(".csv"):
            data = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            data = pd.read_excel(file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None


def analyze_data(data):
    """Perform basic analysis on the ERP data."""
    analysis = {}
    if "sales" in data.columns:
        analysis["total_sales"] = data["sales"].sum()
        analysis["average_sales"] = data["sales"].mean()
    if "inventory" in data.columns:
        analysis["total_inventory"] = data["inventory"].sum()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        analysis["date_range"] = f"{data['date'].min()} - {data['date'].max()}"
    return analysis


def generate_report(data, analysis):
    """Generate a PDF report with analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="ERP Data Analysis Report", ln=True, align="C")

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Report Generated: {datetime.now()}", ln=True)

    pdf.ln(10)
    for key, value in analysis.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Save to a buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


def generate_chart(data, column_x, column_y, chart_type="line"):
    """Generate a chart based on the selected data."""
    plt.figure(figsize=(10, 6))
    if chart_type == "line":
        plt.plot(data[column_x], data[column_y], marker="o")
    elif chart_type == "bar":
        plt.bar(data[column_x], data[column_y], color="skyblue")
    elif chart_type == "scatter":
        plt.scatter(data[column_x], data[column_y], alpha=0.7, edgecolors="b")

    plt.title(f"{column_y} vs {column_x} ({chart_type.capitalize()} Chart)")
    plt.xlabel(column_x)
    plt.ylabel(column_y)
    plt.grid(True)

    # Save chart to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf


def generate_nlp_summary(api_key, analysis):
    """Generate a natural language summary using OpenAI GPT."""
    prompt = f"""
    Based on the following analysis, generate a concise summary in Traditional Chinese (#zh-tw):
    {json.dumps(analysis, ensure_ascii=False, indent=2)}
    """
    try:
        response = st.session_state.chat_model(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Failed to generate NLP summary: {e}")
        return None



# --- Main App ---
def main():
    st.set_page_config(page_title="ERP Data + Chatbot", page_icon="📊", layout="wide")
    st.title("📊 ERP Data Analysis + 🤖 Chatbot + 🧠 Memory")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if not api_key:
            st.warning("⬅️ Please enter the API key to proceed.")
            return

        if "conversation" not in st.session_state:
            st.session_state.chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key)
            st.session_state.memory = ConversationBufferMemory()
            st.session_state.conversation = ConversationChain(
                llm=st.session_state.chat_model,
                memory=st.session_state.memory
            )

        # Memory management buttons
        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

        # Show current memory state
        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

    # --- File Upload ---
    st.subheader("Upload ERP Data File")
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        data = process_data(uploaded_file)
        if data is not None:
            st.write("### Data Preview")
            st.dataframe(data.head())

            # Data Analysis
            st.subheader("Analysis Results")
            analysis = analyze_data(data)
            st.json(analysis)

            # Generate NLP Summary
            st.subheader("Natural Language Summary")
            nlp_summary = generate_nlp_summary(api_key, analysis)
            if nlp_summary:
                st.write(nlp_summary)

            # Generate Chart
            st.subheader("Generate Chart")
            columns = list(data.columns)
            column_x = st.selectbox("Select X-axis", options=columns)
            column_y = st.selectbox("Select Y-axis", options=columns)
            chart_type = st.selectbox("Select Chart Type", options=["line", "bar", "scatter"])

            if st.button("Generate Chart"):
                chart_buf = generate_chart(data, column_x, column_y, chart_type)
                st.image(chart_buf, caption="Generated Chart", use_column_width=True)
                st.download_button(
                    label="Download Chart",
                    data=chart_buf,
                    file_name="chart.png",
                    mime="image/png"
                )

            # Generate PDF Report
            st.subheader("Download Report")
            if st.button("Generate PDF Report"):
                report_buf = generate_report(data, analysis)
                st.download_button(
                    label="Download Report",
                    data=report_buf,
                    file_name="ERP_Report.pdf",
                    mime="application/pdf"
                )

    # --- Chat Interface ---
    st.subheader("Chat Interface")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
