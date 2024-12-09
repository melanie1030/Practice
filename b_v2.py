import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import json
import openai

# --- Initialize OpenAI ---
openai.api_key = st.secrets["openai_api_key"]


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


def generate_nlp_summary(analysis):
    """Generate a natural language summary using OpenAI GPT."""
    prompt = f"""
    Based on the following analysis, generate a concise summary in Traditional Chinese (#zh-tw):
    {json.dumps(analysis, ensure_ascii=False, indent=2)}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


# --- Main App ---
def main():
    st.set_page_config(page_title="ERP Data Analysis", page_icon="📊", layout="wide")
    st.title("📊 ERP Data Analysis Report Generator")

    # File Upload
    st.subheader("Upload ERP Data File")
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])
    if not uploaded_file:
        st.warning("Please upload a data file to proceed.")
        return

    # Process Data
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
        nlp_summary = generate_nlp_summary(analysis)
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


if __name__ == "__main__":
    main()
