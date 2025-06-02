# simplified_data_agent_streamlit_v3_enhanced.py

import streamlit as st
import pandas as pd
import os
import io  # To read CSV content as string
import json  # For data summary
import datetime  # For timestamping saved files
import matplotlib.pyplot  # Explicit import for placeholder and execution scope
import seaborn  # Explicit import for placeholder and execution scope
import numpy as np  # For numerical operations
import html  # For escaping HTML content

# --- Plotly Imports ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PDF Export ---
# pip install reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- Langchain/LLM Components ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- Configuration ---
LLM_API_KEY = os.environ.get("LLM_API_KEY", "AIzaSyCX5KJCOcTEY7IkOVjAB3uYsxCgYPSlt4I")  # Default API Key
# Updated TEMP_DATA_STORAGE to include "AI analysis" subfolder
TEMP_DATA_STORAGE = "temp_data_simplified_agent/AI analysis/"
os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)

AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-05-06"]
DEFAULT_WORKER_MODEL = "gemini-2.0-flash-lite"
DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"

if LLM_API_KEY == "YOUR_API_KEY_HERE" or LLM_API_KEY == "API PLZ":
    st.error(
        "Please set your LLM_API_KEY (e.g., GOOGLE_API_KEY) environment variable or Streamlit secret for full functionality.")


# --- LLM Initialization & Placeholder ---
class PlaceholderLLM:
    """Simulates LLM responses for when an API key is not available."""

    def __init__(self, model_name="placeholder_model"):
        self.model_name = model_name
        st.warning(f"Using PlaceholderLLM for {self.model_name} as API key is not set or invalid.")

    def invoke(self, prompt_input):
        prompt_str_content = str(prompt_input.to_string() if hasattr(prompt_input, 'to_string') else prompt_input)

        if "CDO, your first task is to provide an initial description of the dataset" in prompt_str_content:
            data_summary_json = {}
            try:
                summary_marker = "Data Summary (for context):"
                if summary_marker in prompt_str_content:
                    json_str_part = \
                        prompt_str_content.split(summary_marker)[1].split("\n\nDetailed Initial Description by CDO:")[
                            0].strip()
                    data_summary_json = json.loads(json_str_part)
            except Exception:
                pass  # Ignore errors if JSON parsing fails

            cols = data_summary_json.get("columns", ["N/A"])
            num_rows = data_summary_json.get("num_rows", "N/A")
            num_cols = data_summary_json.get("num_columns", "N/A")
            user_desc = data_summary_json.get("user_provided_column_descriptions", "No user descriptions provided.")
            dtypes_str = "\n".join(
                [f"- {col}: {data_summary_json.get('dtypes', {}).get(col, 'Unknown')}" for col in cols])

            return {"text": f"""
*Placeholder CDO Initial Data Description ({self.model_name}):*

**1. Dataset Overview (Simulated df.info()):**
   - Rows: {num_rows}, Columns: {num_cols}
   - Column Data Types:
{dtypes_str}
   - Potential Memory Usage: (Placeholder value) MB

**2. User-Provided Column Descriptions:**
   - {user_desc}

**3. Inferred Meaning of Variables (Example):**
   - `ORDERNUMBER`: Unique identifier for each order.
   - `QUANTITYORDERED`: Number of units for a product in an order.
   *(This is a generic interpretation; actual meanings depend on the dataset and user descriptions.)*

**4. Initial Data Quality Assessment (Example):**
   - **Missing Values:** (Placeholder - e.g., "Column 'ADDRESSLINE2' has 80% missing values.")
   - **Overall:** The dataset seems reasonably structured.
"""}

        elif "panel of expert department heads, including the CDO" in prompt_str_content:
            return {"text": """
*Placeholder Departmental Perspectives (after CDO's initial report, via {model_name}):*

**CEO:** Focus on revenue trends, considering the column meanings provided.
**CFO:** Assess regional profitability, cross-referencing with user descriptions.
**CDO (Highlighting for VPs):** Consider missing values and how user descriptions clarify data points.
""".format(model_name=self.model_name)}

        elif "You are the Chief Data Officer (CDO) of the company." in prompt_str_content and "synthesize these diverse perspectives" in prompt_str_content:
            return {"text": """
*Placeholder Final Analysis Strategy (Synthesized by CDO, via {model_name}):*

1.  **Visualize Core Sales Trends:** Line plot of 'SALES' over 'ORDERDATE' (interpret 'SALES' based on user description if available).
2.  **Tabulate Product Line Performance:** Table of 'SALES', 'PRICEEACH', 'QUANTITYORDERED' by 'PRODUCTLINE'.
3.  **Descriptive Summary of Order Status:** Table count by 'STATUS'.
4.  **Data Quality Table for Key Columns:** Table of missing value % for key columns.
5.  **Visualize Sales by Country:** Bar chart of 'SALES' by 'COUNTRY'.
""".format(model_name=self.model_name)}
        elif "Python code:" in prompt_str_content and "User Query:" in prompt_str_content:  # Code generation
            user_query_segment = prompt_str_content.split("User Query:")[1].split("\n")[0].lower()

            fallback_script = """
# Standard library imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os # Added os import

analysis_result = "Analysis logic executed. If you expected a specific output, please check the generated script."
plot_data_df = None 

# --- AI Generated Code Start ---
# Placeholder: The AI would generate its specific analysis logic here.
# --- AI Generated Code End ---

if 'analysis_result' not in locals() or (isinstance(analysis_result, str) and analysis_result == "Analysis logic executed. If you expected a specific output, please check the generated script."):
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty: # Check if df exists
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set by the AI's main logic. Displaying df.head() as a default."
        plot_data_df = df.head().copy() 
    else:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set, and no DataFrame was available."
"""
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['SALES'].mean()\nplot_data_df = None"}  # Assuming 'SALES' column
            elif "plot" in user_query_segment or "visualize" in user_query_segment:
                placeholder_plot_filename = "placeholder_plot.png"
                # Ensure TEMP_DATA_STORAGE is accessible here or passed correctly
                placeholder_full_save_path = os.path.join(TEMP_DATA_STORAGE, placeholder_plot_filename).replace("\\",
                                                                                                                "/")

                generated_plot_code = f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Ensure TEMP_DATA_STORAGE is defined in the execution scope if not globally available
# For this placeholder, we assume it's accessible or the path is constructed correctly.
# temp_data_storage_path = "{TEMP_DATA_STORAGE}" # Example if needed

fig, ax = plt.subplots()
if 'df' in locals() and not df.empty and len(df.columns) > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        ax.hist(df[numeric_cols[0]])
        plot_data_df = df[[numeric_cols[0]]].copy()
        # Construct path carefully using repr() for safety in generated code
        plot_save_path = {repr(placeholder_full_save_path)}
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(plot_save_path)
        plt.close(fig)
        analysis_result = {repr(placeholder_plot_filename)} 
    else: 
        if not df.empty:
            try:
                # Attempt to plot first column if no numerics, ensure it's plottable
                if df.iloc[:, 0].nunique() < 50 : # Heuristic for plottable categorical
                    counts = df.iloc[:, 0].value_counts().head(10) 
                    counts.plot(kind='bar', ax=ax)
                    plt.title(f'Value Counts for {{df.columns[0]}}')
                    plt.ylabel('Frequency')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plot_data_df = counts.reset_index()
                    plot_data_df.columns = [df.columns[0], 'count']
                    plot_save_path = {repr(placeholder_full_save_path)} # Use repr() here too
                    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
                    plt.savefig(plot_save_path)
                    plt.close(fig)
                    analysis_result = {repr(placeholder_plot_filename)} # Use repr() here too
                else:
                    ax.text(0.5, 0.5, 'First column not suitable for bar plot (too many unique values).', ha='center', va='center')
                    plot_data_df = pd.DataFrame()
                    analysis_result = "Failed to generate fallback plot: First column not plottable."
                    plt.close(fig)
            except Exception as e:
                ax.text(0.5, 0.5, 'Could not generate fallback plot.', ha='center', va='center')
                plot_data_df = pd.DataFrame()
                analysis_result = "Failed to generate fallback plot: " + str(e)
                plt.close(fig)
        else: # This case should be caught by the outer if 'df' in locals()
            ax.text(0.5, 0.5, 'No data to plot (df is empty).', ha='center', va='center')
            plot_data_df = pd.DataFrame()
            analysis_result = "No data to plot (df is empty)."
            plt.close(fig)
else:
    ax.text(0.5, 0.5, 'No data to plot (df not found or empty).', ha='center', va='center')
    plot_data_df = pd.DataFrame()
    analysis_result = "DataFrame is empty or not found, cannot plot."
    plt.close(fig)
"""
                return {"text": generated_plot_code}
            elif "table" in user_query_segment or "summarize" in user_query_segment:
                return {
                    "text": "analysis_result = df.describe()\nplot_data_df = df.describe().reset_index()"}
            else:
                return {"text": fallback_script}

        elif "Generate a textual report" in prompt_str_content:
            return {
                "text": f"### Placeholder Report ({self.model_name})\nThis is a placeholder report based on the CDO's focused analysis strategy, considering any user-provided column descriptions."}
        elif "Critique the following analysis artifacts" in prompt_str_content:
            return {"text": f"""
### Placeholder Critique ({self.model_name})
**Overall Assessment:** Placeholder. The analysis should ideally reflect insights from user-provided column descriptions if available.
**Python Code:** Placeholder.
**Data:** Placeholder.
**Report:** Placeholder.
**Suggestions for Worker AI:** Placeholder. Ensure user context from descriptions is used.
"""}
        # Placeholder for HTML generation
        elif "Generate a single, complete, and runnable HTML file" in prompt_str_content:
            return {"text": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placeholder Bento Report</title>
    <style>
        body { font-family: sans-serif; background-color: #1A1B26; color: #E0E0E0; margin: 0; padding: 20px; }
        .bento-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; max-width: 1600px; margin: auto; }
        .bento-item { background-color: #2A2D3E; border: 1px solid #3A3D4E; border-radius: 16px; padding: 20px; }
        .bento-item h2 { color: #FFFFFF; font-size: 1.5rem; border-bottom: 2px solid #7DF9FF; padding-bottom: 5px; margin-top:0; }
        p { color: #C0C0C0; }
    </style>
</head>
<body>
    <div class="bento-grid">
        <div class="bento-item"><h2>Analysis Goal</h2><p>Placeholder: User query will be displayed here.</p></div>
        <div class="bento-item"><h2>Data Snapshot & User Descriptions</h2><p>Placeholder: CDO's initial data description (including user insights) will be here.</p></div>
        <div class="bento-item" style="grid-column: span 2; background-color: #4A235A; border-color: #C778DD;">
            <h2 style="border-bottom-color: #C778DD;">Key Data Quality Alert</h2>
            <p>Placeholder: Missing data points will be listed here.</p>
        </div>
        <div class="bento-item"><h2>Actionable Insights</h2><p>Placeholder: Generated report insights (reflecting user descriptions) will be here.</p></div>
        <div class="bento-item"><h2>Critique Summary</h2><p>Placeholder: Critique text (considering use of descriptions) will be here.</p></div>
    </div>
</body>
</html>"""}
        else:
            return {
                "text": f"Placeholder response from {self.model_name} for unrecognized prompt: {prompt_str_content[:200]}..."}


def get_llm_instance(model_name: str):
    """
    Retrieves or initializes an LLM instance.
    Uses a cache (st.session_state.llm_cache) to store initialized models.
    Falls back to PlaceholderLLM if API key is not set or initialization fails.
    """
    if not model_name:
        st.error("No model name provided for LLM initialization.")
        return None
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    if model_name not in st.session_state.llm_cache:
        if not LLM_API_KEY or LLM_API_KEY == "LLM_API_KEY" or LLM_API_KEY == "YOUR_API_KEY_HERE" or LLM_API_KEY == "API PLZ":
            st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
        else:
            try:
                # Set temperature based on whether the model is a judge model or worker model
                temperature = 0.7 if st.session_state.get("selected_judge_model", "") == model_name else 0.2
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=LLM_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True  # Important for some models/versions
                )
                st.session_state.llm_cache[model_name] = llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM ({model_name}): {e}")
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
    return st.session_state.llm_cache[model_name]


@st.cache_data
def calculate_data_summary(df_input, user_column_descriptions_content=None):
    """
    Calculates a comprehensive summary of the input DataFrame.
    Includes row/column counts, dtypes, missing values, descriptive stats, previews,
    and integrates user-provided column descriptions if available.
    """
    if df_input is None or df_input.empty:
        return None
    df = df_input.copy()  # Work on a copy to avoid modifying the original DataFrame
    data_summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values_total": int(df.isnull().sum().sum()),  # Total missing values
        "missing_values_per_column": df.isnull().sum().to_dict(),  # Missing values per column
        "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",
        # Descriptive statistics
        "preview_head": df.head().to_dict(orient='records'),  # First 5 rows
        "preview_tail": df.tail().to_dict(orient='records'),  # Last 5 rows
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    # Calculate overall percentage of missing values
    data_summary["missing_values_percentage"] = (data_summary["missing_values_total"] / (
            data_summary["num_rows"] * data_summary["num_columns"])) * 100 if (data_summary["num_rows"] *
                                                                               data_summary[
                                                                                   "num_columns"]) > 0 else 0
    # Integrate user-provided column descriptions
    if user_column_descriptions_content:
        data_summary["user_provided_column_descriptions"] = user_column_descriptions_content

    return data_summary


def load_csv_and_get_summary(uploaded_csv_file, uploaded_desc_file=None):
    """
    Loads a CSV file and an optional description file.
    Generates a summary incorporating both.
    Updates session state and resets relevant workflow variables.
    """
    try:
        df = pd.read_csv(uploaded_csv_file)
        st.session_state.current_dataframe = df
        st.session_state.data_source_name = uploaded_csv_file.name

        user_column_descriptions_content = None
        if uploaded_desc_file:
            try:
                # Read as bytes first, then decode, common for Streamlit uploads
                user_column_descriptions_content = uploaded_desc_file.getvalue().decode('utf-8')
                st.session_state.desc_file_name = uploaded_desc_file.name
            except Exception as e:
                st.error(f"Error reading description file '{uploaded_desc_file.name}': {e}")
                # Decide if this should be a fatal error or just a warning; for now, it proceeds
        else:
            st.session_state.desc_file_name = None  # Clear if no file or removed

        st.session_state.current_analysis_artifacts = {}  # Reset artifacts for new data
        summary_for_state = calculate_data_summary(df.copy(), user_column_descriptions_content)

        if summary_for_state:
            summary_for_state["source_name"] = uploaded_csv_file.name  # Add source name to summary
        st.session_state.data_summary = summary_for_state

        # Reset CDO workflow state
        st.session_state.cdo_initial_report_text = None
        st.session_state.other_perspectives_text = None
        st.session_state.strategy_text = None
        if "cdo_workflow_stage" in st.session_state:
            del st.session_state.cdo_workflow_stage
        return True
    except Exception as e:
        st.error(f"Error loading CSV or generating summary: {e}")
        st.session_state.current_dataframe = None
        st.session_state.data_summary = None
        return False


@st.cache_data
def get_overview_metrics(df):
    """
    Calculates key overview metrics for a DataFrame.
    Returns: num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows.
    """
    if df is None or df.empty:
        return 0, 0, 0, 0, 0
    num_rows = len(df)
    num_cols = len(df.columns)
    missing_values_total = df.isnull().sum().sum()
    total_cells = num_rows * num_cols
    missing_percentage = (missing_values_total / total_cells) * 100 if total_cells > 0 else 0
    numeric_cols_count = len(df.select_dtypes(include=np.number).columns)
    duplicate_rows = df.duplicated().sum()
    return num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows


@st.cache_data
def get_column_quality_assessment(df_input):
    """
    Performs a quality assessment for each column in the DataFrame (up to max_cols_to_display).
    Calculates data type, missing percentage, unique values, range/common values, and a quality score.
    Returns a DataFrame with the assessment.
    """
    if df_input is None or df_input.empty:
        return pd.DataFrame()
    df = df_input.copy()
    quality_data = []
    max_cols_to_display = 10  # Limit displayed columns for performance in UI
    for col in df.columns[:max_cols_to_display]:
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_values = df[col].nunique()
        range_common = ""  # Placeholder for range or common values

        # Determine range or common values based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].dropna().empty:
                range_common = f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}"
            else:
                range_common = "N/A (all missing)"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if not df[col].dropna().empty:
                range_common = f"Min: {df[col].min()}, Max: {df[col].max()}"
            else:
                range_common = "N/A (all missing)"
        else:  # Categorical/Object
            if not df[col].dropna().empty:
                common_vals = df[col].mode().tolist()
                range_common = f"Top: {', '.join(map(str, common_vals[:3]))}"
                if len(common_vals) > 3:
                    range_common += "..."
            else:
                range_common = "N/A (all missing)"

        # Calculate a simple quality score
        score = 10
        if missing_percent > 50:
            score -= 5
        elif missing_percent > 20:
            score -= 3
        elif missing_percent > 5:
            score -= 1
        if unique_values == 1 and len(df) > 1: score -= 2  # Penalize if only one unique value (constant)
        if unique_values == len(df) and not pd.api.types.is_numeric_dtype(
                df[col]): score -= 1  # Penalize if all unique (potential ID)

        quality_data.append({
            "Column Name": col, "Data Type": dtype, "Missing %": f"{missing_percent:.2f}%",
            "Unique Values": unique_values, "Range / Common Values": range_common,
            "Quality Score ( /10)": max(0, score)  # Score cannot be negative
        })
    return pd.DataFrame(quality_data)


def generate_data_quality_dashboard(df_input):
    """
    Generates and displays the Data Quality Dashboard in Streamlit.
    Includes overview metrics, column-wise assessment, and distribution plots for numeric/categorical columns.
    """
    if df_input is None or df_input.empty:
        st.warning("No data loaded or DataFrame is empty. Please upload a CSV file.")
        return
    df = df_input.copy()  # Work on a copy

    st.header("📊 Data Quality Dashboard")
    st.markdown("An overview of your dataset's quality and characteristics.")

    # --- Key Dataset Metrics ---
    st.subheader("Key Dataset Metrics")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df.copy())
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Rows", f"{num_rows:,}")
    col2.metric("Total Columns", f"{num_cols:,}")
    # Conditional coloring for missing values metric
    if missing_percentage > 5:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%", delta_color="inverse",
                    help="Percentage of missing data cells in the entire dataset. Red if > 5%.")
    else:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%",
                    help="Percentage of missing data cells in the entire dataset.")
    col4.metric("Numeric Columns", f"{numeric_cols_count:,}")
    col5.metric("Duplicate Rows", f"{duplicate_rows:,}", help="Number of fully duplicated rows.")
    st.markdown("---")

    # --- Column-wise Quality Assessment ---
    st.subheader("Column-wise Quality Assessment")
    if len(df.columns) > 10:
        st.caption(
            f"Displaying first 10 columns out of {len(df.columns)}. Full assessment available via report (placeholder).")
    quality_df = get_column_quality_assessment(df.copy())
    if not quality_df.empty:
        # Function to style the quality table (background colors based on values)
        def style_quality_table(df_to_style):
            return df_to_style.style.apply(
                lambda row: ['background-color: #FFCDD2' if float(str(row["Missing %"]).replace('%', '')) > 20
                             else (
                    'background-color: #FFF9C4' if float(str(row["Missing %"]).replace('%', '')) > 5 else '')
                             for _ in row], axis=1, subset=["Missing %"]
            ).apply(
                lambda row: ['background-color: #FFEBEE' if row["Quality Score ( /10)"] < 5
                             else ('background-color: #FFFDE7' if row["Quality Score ( /10)"] < 7 else '')
                             for _ in row], axis=1, subset=["Quality Score ( /10)"]
            )

        st.dataframe(style_quality_table(quality_df), use_container_width=True)
    else:
        st.info("Could not generate column quality assessment table.")

    if st.button("Generate Full Data Quality PDF Report (Placeholder)", key="dq_pdf_placeholder"):
        st.info("PDF report generation for data quality is not yet implemented.")
    st.markdown("---")

    # --- Numeric Column Distribution ---
    st.subheader("Numeric Column Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found in the dataset.")
    else:
        selected_numeric_col = st.selectbox("Select Numeric Column for Distribution Analysis:", numeric_cols,
                                            key="dq_numeric_select")
        if selected_numeric_col:
            col_data = df[selected_numeric_col].dropna()  # Drop NA for plotting
            if not col_data.empty:
                # Histogram with box plot marginal
                fig = px.histogram(col_data, x=selected_numeric_col, marginal="box",
                                   title=f"Distribution of {selected_numeric_col}", opacity=0.75,
                                   histnorm='probability density')  # Normalize for density
                # Add a dummy scatter trace to ensure plotly layout calculations are correct (sometimes helps with sizing)
                fig.add_trace(
                    go.Scatter(x=col_data, y=[0] * len(col_data), mode='markers', marker=dict(color='rgba(0,0,0,0)'),
                               showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Key Statistics:**")
                stats_cols = st.columns(5)
                stats_cols[0].metric("Mean", f"{col_data.mean():.2f}")
                stats_cols[1].metric("Median", f"{col_data.median():.2f}")
                stats_cols[2].metric("Std Dev", f"{col_data.std():.2f}")
                stats_cols[3].metric("Min", f"{col_data.min():.2f}")
                stats_cols[4].metric("Max", f"{col_data.max():.2f}")
            else:
                st.info(f"Column '{selected_numeric_col}' contains only missing values.")
    st.markdown("---")

    # --- Categorical Column Distribution ---
    st.subheader("Categorical Column Distribution")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        st.info("No categorical columns found in the dataset.")
    else:
        selected_categorical_col = st.selectbox("Select Categorical Column for Distribution Analysis:",
                                                categorical_cols, key="dq_categorical_select")
        if selected_categorical_col:
            col_data = df[selected_categorical_col].dropna()
            if not col_data.empty:
                value_counts = col_data.value_counts(normalize=True).mul(100).round(2)  # Percentages
                count_abs = col_data.value_counts()  # Absolute counts
                # Bar chart for categorical distribution
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"Distribution of {selected_categorical_col}",
                             labels={'x': selected_categorical_col, 'y': 'Percentage (%)'},
                             text=[f"{val:.1f}% ({count_abs[idx]})" for idx, val in
                                   value_counts.items()])  # Display text on bars
                fig.update_layout(xaxis_title=selected_categorical_col, yaxis_title="Percentage (%)")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Column '{selected_categorical_col}' contains only missing values.")
    st.markdown("---")

    # --- Numeric Column Correlation Heatmap ---
    st.subheader("Numeric Column Correlation Heatmap")
    numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr) < 2:  # Need at least 2 numeric columns for correlation
        st.info("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")
    else:
        corr_matrix = df[numeric_cols_for_corr].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                                # Diverging color scale
                                title="Correlation Heatmap of Numeric Columns")
        fig_heatmap.update_xaxes(side="bottom")  # Move x-axis ticks to bottom for better readability
        fig_heatmap.update_layout(xaxis_tickangle=-45, yaxis_tickangle=0)  # Rotate x-axis labels
        st.plotly_chart(fig_heatmap, use_container_width=True)


class LocalCodeExecutionEngine:
    """
    Handles the execution of Python code strings in a controlled environment.
    The code is expected to operate on a pandas DataFrame named 'df'.
    It captures results, plots, and errors.
    """

    def execute_code(self, code_string, df_input):
        """
        Executes the provided Python code string.
        Args:
            code_string (str): The Python code to execute.
            df_input (pd.DataFrame): The DataFrame to be made available as 'df' in the execution scope.
        Returns:
            dict: A dictionary containing the execution result (type, value/path, message).
        """
        if df_input is None:
            return {"type": "error", "message": "No data loaded to execute code on."}

        # Prepare a safe execution environment
        exec_globals = globals().copy()  # Start with global scope
        # Explicitly add commonly used libraries to the execution global scope
        exec_globals['plt'] = matplotlib.pyplot
        exec_globals['sns'] = seaborn
        exec_globals['pd'] = pd
        exec_globals['np'] = np
        exec_globals['os'] = os  # Ensure os is in globals for code execution

        # Local scope for the execution, pre-populated with the DataFrame and libraries
        local_scope = {'df': df_input.copy(), 'pd': pd, 'plt': matplotlib.pyplot, 'sns': seaborn, 'np': np, 'os': os,
                       'TEMP_DATA_STORAGE': TEMP_DATA_STORAGE}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None
        default_analysis_result_message = "Code executed, but 'analysis_result' was not explicitly set by the script."

        # Initialize expected output variables in the local scope
        local_scope['analysis_result'] = default_analysis_result_message
        local_scope['plot_data_df'] = None  # For data specifically used in plots

        try:
            # Ensure the TEMP_DATA_STORAGE directory exists before executing code that might write to it
            os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)
            exec(code_string, exec_globals, local_scope)  # Execute the code string

            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')

            # MODIFIED: Check if analysis_result is still the default string.
            # This avoids the ValueError if analysis_result became a DataFrame.
            if isinstance(analysis_result, str) and analysis_result == default_analysis_result_message:
                st.warning("The executed script did not explicitly set 'analysis_result'.")

            # Handle error messages set within the executed code
            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}

            # Handle plot results (if analysis_result is a plot filename)
            if isinstance(analysis_result, str) and any(
                    analysis_result.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                plot_filename = os.path.basename(analysis_result)  # analysis_result should be just the filename
                final_plot_path = os.path.join(TEMP_DATA_STORAGE, plot_filename)

                # Verify plot file exists where expected
                if not os.path.exists(final_plot_path):
                    # Check if analysis_result was accidentally a full path (less ideal but handle)
                    if os.path.isabs(analysis_result) and os.path.exists(analysis_result):
                        final_plot_path = analysis_result  # It was a full path
                        st.warning(
                            f"Plot was saved with an absolute path: {analysis_result}. Prefer saving to TEMP_DATA_STORAGE with just filename.")
                    else:
                        return {"type": "error",
                                "message": f"Plot file '{plot_filename}' not found in the designated temporary directory '{TEMP_DATA_STORAGE}'. `analysis_result` was: '{analysis_result}'. Ensure the AI's code saves the plot correctly using `os.path.join(TEMP_DATA_STORAGE, 'filename.ext')` and sets `analysis_result` to just 'filename.ext'."}

                # Save associated plot data if plot_data_df is provided
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(plot_filename)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                    st.info(f"Plot-specific data saved to: {plot_data_df_saved_path}")
                elif plot_data_df is not None:  # plot_data_df was set but not a valid DataFrame
                    st.warning("`plot_data_df` was set but is not a valid DataFrame. Not saving associated data.")

                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}

            # Handle table results (if analysis_result is a DataFrame or Series)
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                analysis_result_df = analysis_result.to_frame() if isinstance(analysis_result,
                                                                              pd.Series) else analysis_result
                if analysis_result_df.empty: return {"type": "text",
                                                     "value": "The analysis resulted in an empty table."}

                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, f"table_result_{timestamp}.csv")
                analysis_result_df.to_csv(saved_csv_path, index=False)  # Save the DataFrame
                return {"type": "table", "data_path": saved_csv_path, "dataframe_result": analysis_result_df}


            # Handle text results
            else:
                return {"type": "text", "value": str(analysis_result)}

        except Exception as e:
            import traceback
            error_message_for_user = f"Error during code execution: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            # Try to get the last value of analysis_result if an error occurred mid-script
            current_analysis_res = local_scope.get('analysis_result', default_analysis_result_message)
            if current_analysis_res is None or (
                    isinstance(current_analysis_res, pd.DataFrame) and current_analysis_res.empty):
                local_scope['analysis_result'] = f"Execution Error: {str(e)}"  # Ensure some error state is captured
            return {"type": "error", "message": error_message_for_user,
                    "final_analysis_result_value": local_scope['analysis_result']}


code_executor = LocalCodeExecutionEngine()


# --- PDF Export Function ---
def export_analysis_to_pdf(artifacts, output_filename="analysis_report.pdf"):
    """
    Exports the analysis artifacts (query, CDO report, plot, data, report text, critique) to a PDF file.
    Args:
        artifacts (dict): Dictionary containing paths and content of analysis artifacts.
        output_filename (str): The desired name for the output PDF file.
    Returns:
        str: Path to the generated PDF file, or None if generation failed.
    """
    pdf_path = os.path.join(TEMP_DATA_STORAGE, output_filename)
    doc = SimpleDocTemplate(pdf_path)  # reportlab document template
    styles = getSampleStyleSheet()  # Predefined styles
    story = []  # List of flowables to add to the PDF

    # --- Title ---
    story.append(Paragraph("Comprehensive Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))  # Spacer

    # --- Analysis Goal ---
    story.append(Paragraph("1. Analysis Goal (User Query)", styles['h2']))
    analysis_goal = artifacts.get("original_user_query", "Not specified.")
    story.append(Paragraph(html.escape(analysis_goal), styles['Normal']))  # Escape user content
    story.append(Spacer(1, 0.2 * inch))

    # --- CDO's Initial Data Description ---
    story.append(Paragraph("2. CDO's Initial Data Description & Quality Assessment", styles['h2']))
    cdo_report_text = st.session_state.get("cdo_initial_report_text", "CDO initial report not available.")
    # Basic cleaning for PDF: remove markdown bold, escape HTML entities for safety
    cdo_report_text_cleaned = html.escape(cdo_report_text.replace("**", ""))
    for para_text in cdo_report_text_cleaned.split('\n'):
        if para_text.strip().startswith("- "):  # Handle bullet points
            story.append(Paragraph(para_text, styles['Bullet'], bulletText='-'))
        elif para_text.strip():  # Handle normal paragraphs
            story.append(Paragraph(para_text, styles['Normal']))
        else:  # Handle empty lines as small spacers
            story.append(Spacer(1, 0.1 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())  # New page

    # --- Generated Plot ---
    story.append(Paragraph("3. Generated Plot", styles['h2']))
    plot_image_path = artifacts.get("plot_image_path")
    if plot_image_path and os.path.exists(plot_image_path):
        try:
            img = Image(plot_image_path, width=6 * inch, height=4 * inch)  # Adjust size as needed
            img.hAlign = 'CENTER'  # Center the image
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error embedding plot: {html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("Plot image not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Plot Data / Executed Data Table ---
    story.append(Paragraph("4. Plot Data (or Executed Data Table)", styles['h2']))
    plot_data_csv_path = artifacts.get("executed_data_path")  # This could be general data too
    executed_df = artifacts.get("executed_dataframe_result")  # Get the actual DataFrame if available

    data_to_display_in_pdf = None
    if executed_df is not None and isinstance(executed_df, pd.DataFrame):
        data_to_display_in_pdf = executed_df
    elif plot_data_csv_path and os.path.exists(plot_data_csv_path) and plot_data_csv_path.endswith(".csv"):
        try:
            data_to_display_in_pdf = pd.read_csv(plot_data_csv_path)
        except Exception as e:
            story.append(Paragraph(f"Error reading CSV for PDF table: {html.escape(str(e))}", styles['Normal']))
            data_to_display_in_pdf = None  # Ensure it's None if read fails

    if data_to_display_in_pdf is not None and not data_to_display_in_pdf.empty:
        # Convert all data to string for PDF to avoid type issues with reportlab
        data_for_table = [data_to_display_in_pdf.columns.astype(str).tolist()] + data_to_display_in_pdf.astype(
            str).values.tolist()

        if len(data_for_table) > 1:  # Check if there's data beyond headers
            max_rows_in_pdf = 30  # Limit rows in PDF for readability
            if len(data_for_table) > max_rows_in_pdf:
                data_for_table = data_for_table[:max_rows_in_pdf]
                story.append(Paragraph(f"(Showing first {max_rows_in_pdf - 1} data rows)", styles['Italic']))

            table = Table(data_for_table, repeatRows=1)  # repeatRows=1 makes header repeat on new pages
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('WORDWRAP', (0, 0), (-1, -1), 'CJK')  # Added for better word wrapping
            ]))
            story.append(table)
        else:
            story.append(Paragraph("Data table is empty or contains only headers.", styles['Normal']))
    else:
        story.append(Paragraph("Data for table not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())

    # --- Generated Textual Report ---
    story.append(Paragraph("5. Generated Textual Report (Specific Analysis)", styles['h2']))
    report_text_path = artifacts.get("generated_report_path")
    if report_text_path and os.path.exists(report_text_path):
        try:
            with open(report_text_path, 'r', encoding='utf-8') as f:
                report_text_content = f.read()
            report_text_content_cleaned = html.escape(report_text_content.replace("**", ""))
            for para_text in report_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;",
                                       styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading report file: {html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("Generated report text file not available.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Analysis Critique ---
    story.append(Paragraph("6. Analysis Critique", styles['h2']))
    critique_text_path = artifacts.get("generated_critique_path")
    if critique_text_path and os.path.exists(critique_text_path):
        try:
            with open(critique_text_path, 'r', encoding='utf-8') as f:
                critique_text_content = f.read()
            critique_text_content_cleaned = html.escape(critique_text_content.replace("**", ""))
            for para_text in critique_text_content_cleaned.split('\n'):
                story.append(Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading critique file: {html.escape(str(e))}", styles['Normal']))
    else:
        story.append(Paragraph("Critique text file not available.", styles['Normal']))

    # --- Build PDF ---
    try:
        doc.build(story)
        return pdf_path
    except Exception as e:
        st.error(f"Failed to build PDF: {e}")
        return None


# --- HTML Bento Report Generation (Python-based) ---

def _generate_html_paragraphs(text_content):
    """Converts plain text with newlines into HTML paragraphs, escaping content."""
    if not text_content or text_content.strip() == "Not available.":
        return "<p><em>Not available.</em></p>"
    # Escape HTML first, then replace newlines with <p> tags
    escaped_content = html.escape(text_content)
    paragraphs = "".join([f"<p>{line}</p>" for line in escaped_content.split('\n') if line.strip()])
    return paragraphs if paragraphs else "<p><em>No content provided.</em></p>"


def _generate_html_table(csv_text_or_df):
    """Generates an HTML table from a CSV string or a pandas DataFrame."""
    df = None
    if isinstance(csv_text_or_df, pd.DataFrame):
        df = csv_text_or_df
    elif isinstance(csv_text_or_df, str):
        prefix_to_strip = None
        prefixes = [
            "Primary Table Data (CSV format for HTML table):\n",
            "Data for Chart (CSV format for Chart.js):\n",  # Though this usually goes to chartjs
            "DQ Column Assessment (CSV for table):\n",
            "DQ Correlation Matrix (CSV for table):\n"
        ]
        for p in prefixes:
            if csv_text_or_df.startswith(p):
                prefix_to_strip = p
                break
        if prefix_to_strip:
            csv_text_or_df = csv_text_or_df.split(prefix_to_strip, 1)[1]

        if not csv_text_or_df or csv_text_or_df.strip() == "Not available." or "No specific data table" in csv_text_or_df or "General table data not specifically generated" in csv_text_or_df:
            return "<p><em>Table data not available or not applicable for this section.</em></p>"
        if not csv_text_or_df.strip():
            return "<p><em>Table data not available or not applicable for this section.</em></p>"
        try:
            csv_file = io.StringIO(csv_text_or_df)
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            return "<p><em>Table data is empty or not in a valid CSV format.</em></p>"
        except Exception as e:
            return f"<p><em>Error parsing CSV for table: {html.escape(str(e))}.</em></p>"
    else:
        return "<p><em>Invalid data type for table generation.</em></p>"

    if df is None or df.empty:
        return "<p><em>Table data is empty.</em></p>"

    table_html = "<table>\n<thead>\n<tr>"
    for header in df.columns:
        table_html += f"<th>{html.escape(str(header))}</th>"
    table_html += "</tr>\n</thead>\n<tbody>\n"

    for _, row in df.iterrows():
        table_html += "<tr>"
        for cell in row:
            cell_str = str(cell)
            is_numeric = pd.api.types.is_number(cell) and not pd.isna(cell)  # More robust numeric check
            class_attr = ' class="numeric"' if is_numeric else ''
            table_html += f"<td{class_attr}>{html.escape(cell_str)}</td>"
        table_html += "</tr>\n"

    table_html += "</tbody>\n</table>"
    return table_html


def _generate_html_image_embed(image_path_from_artifact):
    """Generates HTML for embedding a local image with a note."""
    if not image_path_from_artifact or not os.path.exists(image_path_from_artifact):
        return "<p><em>Image not found or path is invalid.</em></p>"

    image_filename = os.path.basename(image_path_from_artifact)
    # Ensure the src path is relative for portability
    img_tag = f'<img src="{html.escape(image_filename)}" alt="Visualization" style="max-width: 100%; max-height: 100%; height: auto; display: block; margin: auto; border-radius: 8px;">'
    note = f'<p class="image-note"><strong>Note for image \'{html.escape(image_filename)}\':</strong> For this image to display, please ensure the file <code>{html.escape(image_filename)}</code> is located in the same directory as this HTML file.</p>'
    # Removed comment about base64 as it's not used.
    return f'<div class="visualization-container">{img_tag}</div>{note}'


def _generate_chartjs_embed(csv_text_or_df, chart_id):
    """Generates HTML and JavaScript for a Chart.js chart from CSV data or DataFrame."""
    df = None
    if isinstance(csv_text_or_df, pd.DataFrame):
        df = csv_text_or_df
    elif isinstance(csv_text_or_df, str):
        prefix_to_strip = None
        prefixes = [
            "Data for Chart (CSV format for Chart.js):\n",
            "DQ Numeric Dist. Example (CSV for Chart.js):\n",
            "DQ Categorical Dist. Example (CSV for Chart.js):\n"
        ]
        for p in prefixes:
            if csv_text_or_df.startswith(p):
                prefix_to_strip = p
                break
        if prefix_to_strip:
            csv_text_or_df = csv_text_or_df.split(prefix_to_strip, 1)[1]

        if not csv_text_or_df or "Not available" in csv_text_or_df or "No specific data table" in csv_text_or_df:
            return f"<div class='visualization-container'><p><em>Chart.js data not available or not applicable.</em></p></div>"
        if not csv_text_or_df.strip():
            return f"<div class='visualization-container'><p><em>Chart.js data is empty.</em></p></div>"
        try:
            csv_file = io.StringIO(csv_text_or_df)
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            return f"<div class='visualization-container'><p><em>Chart.js data is empty or not in valid CSV format.</em></p></div>"
        except Exception as e:
            return f"<div class='visualization-container'><p><em>Error parsing CSV for Chart.js: {html.escape(str(e))}</em></p></div>"
    else:
        return f"<div class='visualization-container'><p><em>Invalid data type for Chart.js generation.</em></p></div>"

    if df is None or df.empty or len(df.columns) < 2:
        return f"<div class='visualization-container'><p><em>Chart.js data requires at least two columns and non-empty data.</em></p></div>"

    labels = df.iloc[:, 0].astype(str).tolist()  # Ensure labels are strings
    data_values = df.iloc[:, 1].tolist()

    numeric_data_values = []
    for val in data_values:
        try:
            numeric_data_values.append(float(val))
        except (ValueError, TypeError):  # Catch TypeError if val is not convertible
            return f"<div class='visualization-container'><p><em>Chart.js data values in the second column must be numeric. Found non-numeric value: '{html.escape(str(val))}'</em></p></div>"

    chart_type = 'bar'  # Default
    # Use the actual column name for the label, escaped
    chart_label = html.escape(str(df.columns[1]))

    if all("-" in str(l) for l in labels[:3]) and len(labels) > 1:
        chart_label = "Frequency"  # Keep this heuristic for histograms

    canvas_html = f'<div class="visualization-container"><canvas id="{html.escape(chart_id)}"></canvas></div>'
    script_html = f"""
<script>
    const ctx_{html.escape(chart_id)} = document.getElementById('{html.escape(chart_id)}').getContext('2d');
    new Chart(ctx_{html.escape(chart_id)}, {{
        type: '{chart_type}',
        data: {{
            labels: {json.dumps(labels)},
            datasets: [{{
                label: '{chart_label}',
                data: {json.dumps(numeric_data_values)},
                backgroundColor: 'rgba(199, 120, 221, 0.6)', // Softer purple with alpha
                borderColor: 'rgba(74, 35, 90, 1)',    // Darker purple border
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false, // Important for fitting in bento box
            scales: {{
                y: {{
                    beginAtZero: true,
                    ticks: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }},
                    grid: {{ color: '#3A3D4E' }}
                }},
                x: {{
                    ticks: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }},
                    grid: {{ color: '#3A3D4E' }}
                }}
            }},
            plugins: {{
                legend: {{
                    labels: {{ color: '#E0E0E0', font: {{ family: "Inter" }} }}
                }},
                tooltip: {{
                    backgroundColor: '#2A2D3E',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: '#7DF9FF',
                    borderWidth: 1,
                    titleFont: {{ family: "Inter" }},
                    bodyFont: {{ family: "Inter" }}
                }}
            }}
        }}
    }});
</script>
"""
    return canvas_html + script_html


def generate_bento_html_report_python(artifacts, cdo_initial_report, data_summary_dict, main_df):
    """
    Generates a Bento-style HTML report using Python string formatting.
    Includes Data Quality Dashboard elements and user column descriptions.
    """
    # Pass the main_df to compile_report_text for DQ elements
    report_parts_dict = compile_report_text_for_html_generation(artifacts, cdo_initial_report, data_summary_dict,
                                                                main_df)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Generated Bento Report - {html.escape(artifacts.get("original_user_query", "Analysis")[:50])}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #1A1B26; color: #E0E0E0; margin: 0; padding: 20px; line-height: 1.6; }}
        .bento-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; max-width: 1600px; margin: 20px auto; padding:0; }}
        .bento-item {{ 
            background-color: #2A2D3E; 
            border: 1px solid #3A3D4E; 
            border-radius: 16px; 
            padding: 25px; 
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            overflow-wrap: break-word; 
            word-wrap: break-word; 
            overflow: hidden; /* Prevents content from spilling */
            display: flex; /* For better internal alignment */
            flex-direction: column; /* Stack title and content */
        }}
        .bento-item:hover {{ 
            transform: translateY(-5px); 
            border-color: #7DF9FF; 
            box-shadow: 0 10px 20px rgba(0,0,0,0.25);
        }}
        .bento-item h2 {{ 
            color: #FFFFFF; 
            font-size: 1.5rem; /* Slightly reduced for balance */
            font-weight: 600;
            border-bottom: 2px solid #7DF9FF; 
            padding-bottom: 10px; 
            margin-top:0; 
            margin-bottom: 15px; /* Consistent spacing */
        }}
        .bento-item .content-wrapper {{ flex-grow: 1; overflow-y: auto; }} /* Allows content to scroll if too long */
        .bento-item p {{ color: #C0C0C0; margin-bottom: 10px; }}
        .bento-item p:last-child {{ margin-bottom: 0; }}
        .bento-item strong {{ color: #7DF9FF; font-weight: 600; }}
        .bento-item ul {{ list-style-position: inside; padding-left: 5px; color: #C0C0C0; margin-bottom:10px; }}
        .bento-item li {{ margin-bottom: 6px; }}

        /* Spanning rules for different screen sizes */
        .bento-item.large {{ grid-column: span 1; }} /* Default for small screens */
        @media (min-width: 768px) {{ /* Tablets and up */
            .bento-item.large {{ grid-column: span 2; }}
        }}
        /* For very wide screens, could consider span 3 for some items if needed */
        @media (min-width: 1200px) {{ 
             .bento-grid {{ grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }}
        }}

        .bento-item.accent-box {{ background-color: #4A235A; border-color: #C778DD; }}
        .bento-item.accent-box h2 {{ border-bottom-color: #C778DD; }}
        .bento-item.accent-box strong {{ color: #FFA6FC; }}


        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; table-layout: auto; /* Changed to auto for better fit */ }}
        th, td {{ border: 1px solid #4A4D5E; padding: 10px; text-align: left; word-wrap: break-word; font-size: 0.9rem; }}
        th {{ background-color: #3A3D4E; color: #FFFFFF; font-weight: 600; }}
        td.numeric {{ text-align: right; color: #7DF9FF; }}

        .visualization-container {{ 
            min-height: 300px; /* Adjusted min-height */
            max-height: 400px; /* Adjusted max-height */
            width: 100%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            overflow: hidden; 
            background-color: #202230; 
            border-radius: 8px; 
            padding:10px; 
            margin-bottom: 10px; /* Space before image note */
        }}
        .visualization-container img, .visualization-container canvas {{ 
            max-width: 100%; 
            max-height: 100%; 
            object-fit: contain; /* Ensures entire image/chart is visible */
            display: block; 
            margin: auto; 
            border-radius: 8px;
        }}
        .image-note {{ font-size: 0.8rem; color: #A0A0A0; text-align: center; margin-top: 5px; }}
        .placeholder-text {{ color: #888; font-style: italic; }}
        .user-descriptions {{ border-left: 3px solid #7DF9FF; padding-left: 15px; margin-top:10px; background-color: rgba(125, 249, 255, 0.05); border-radius: 4px; }}
        .user-descriptions p {{ color: #D0D0D0; }}
    </style>
</head>
<body>
    <div class="bento-grid">
"""

    # MODIFIED: Removed DQ_NUMERIC_DIST_CHART and DQ_CATEGORICAL_DIST_CHART
    ordered_keys = [
        "ANALYSIS_GOAL", "DATA_SNAPSHOT_CDO_REPORT", "USER_PROVIDED_DESCRIPTIONS", "KEY_DATA_QUALITY_ALERT",
        "DATA_PREPROCESSING_NOTE",
        "DQ_COLUMN_ASSESSMENT_TABLE", "DQ_CORRELATION_MATRIX_TABLE",  # Removed numeric and categorical dist charts
        "VISUALIZATION_CHART_OR_IMAGE", "PRIMARY_ANALYSIS_TABLE",
        "ACTIONABLE_INSIGHTS_FROM_REPORT", "CRITIQUE_SUMMARY", "SUGGESTIONS_FOR_ENHANCED_ANALYSIS"
    ]

    for key_index, key in enumerate(ordered_keys):
        if key not in report_parts_dict:
            # st.warning(f"HTML Gen: Key '{key}' not found in report_parts_dict. Skipping.")
            continue

        # MODIFIED: Removed titles for the removed charts
        title_map = {
            "ANALYSIS_GOAL": "Analysis Goal",
            "DATA_SNAPSHOT_CDO_REPORT": "CDO's Data Snapshot",
            "USER_PROVIDED_DESCRIPTIONS": "User-Provided Column Insights",
            "KEY_DATA_QUALITY_ALERT": "Key Data Quality Alerts",
            "DATA_PREPROCESSING_NOTE": "Data Preprocessing Notes",
            "DQ_COLUMN_ASSESSMENT_TABLE": "Data Quality: Column Assessment",
            "DQ_CORRELATION_MATRIX_TABLE": "Data Quality: Correlation Matrix",
            "VISUALIZATION_CHART_OR_IMAGE": "Primary Visualization",
            "PRIMARY_ANALYSIS_TABLE": "Primary Analysis Table",
            "ACTIONABLE_INSIGHTS_FROM_REPORT": "Actionable Insights",
            "CRITIQUE_SUMMARY": "Critique Summary",
            "SUGGESTIONS_FOR_ENHANCED_ANALYSIS": "Suggestions for Enhancement"
        }
        title = title_map.get(key, key.replace('_', ' ').title())
        raw_content_obj = report_parts_dict.get(key)  # This can be string or DataFrame

        item_classes = ["bento-item"]
        item_content_html = ""

        # Determine if item should be large
        # MODIFIED: Removed DQ_NUMERIC_DIST_CHART and DQ_CATEGORICAL_DIST_CHART from large_keys
        large_keys = [
            "KEY_DATA_QUALITY_ALERT", "VISUALIZATION_CHART_OR_IMAGE", "PRIMARY_ANALYSIS_TABLE",
            "DQ_COLUMN_ASSESSMENT_TABLE",
            "DQ_CORRELATION_MATRIX_TABLE", "ACTIONABLE_INSIGHTS_FROM_REPORT", "DATA_SNAPSHOT_CDO_REPORT",
            "USER_PROVIDED_DESCRIPTIONS"
        ]
        if key in large_keys:
            item_classes.append("large")

        if key == "KEY_DATA_QUALITY_ALERT":
            item_classes.append("accent-box")
            if isinstance(raw_content_obj, str) and raw_content_obj.strip():
                alert_lines = [f"<li>{html.escape(line.strip())}</li>" for line in raw_content_obj.split('\n') if
                               line.strip().startswith("- ")]
                if alert_lines:
                    item_content_html = f"<ul>{''.join(alert_lines)}</ul>"
                else:  # Fallback if no bullet points, treat as paragraph
                    item_content_html = _generate_html_paragraphs(raw_content_obj)
            else:
                item_content_html = "<p><em>No specific quality alerts highlighted, or alerts are integrated into CDO report.</em></p>"

        elif key == "USER_PROVIDED_DESCRIPTIONS":
            if isinstance(raw_content_obj,
                          str) and raw_content_obj.strip() and raw_content_obj != "User descriptions not provided.":
                item_content_html = f"<div class='user-descriptions'>{_generate_html_paragraphs(raw_content_obj)}</div>"
            else:
                item_content_html = "<p><em>No additional column descriptions were provided by the user.</em></p>"

        elif key == "VISUALIZATION_CHART_OR_IMAGE":
            plot_image_path = artifacts.get("plot_image_path")  # From original artifacts
            executed_data_for_chart = raw_content_obj  # This should be the DataFrame for chart if no image

            if plot_image_path and os.path.exists(plot_image_path):
                item_content_html = _generate_html_image_embed(plot_image_path)
            elif isinstance(executed_data_for_chart, pd.DataFrame) and not executed_data_for_chart.empty:
                item_content_html = _generate_chartjs_embed(executed_data_for_chart, f"bentoChartAiAnalysis{key_index}")
            else:
                item_content_html = "<p><em>No visualization available or data provided was not suitable for a chart.</em></p>"

        elif key in ["PRIMARY_ANALYSIS_TABLE", "DQ_COLUMN_ASSESSMENT_TABLE", "DQ_CORRELATION_MATRIX_TABLE"]:
            if isinstance(raw_content_obj, pd.DataFrame):
                item_content_html = _generate_html_table(raw_content_obj)
            else:  # Fallback if it's a string (e.g. "Not available")
                item_content_html = _generate_html_paragraphs(str(raw_content_obj))

        # MODIFIED: Removed the elif block for DQ_NUMERIC_DIST_CHART and DQ_CATEGORICAL_DIST_CHART
        # as they are no longer in ordered_keys.

        elif isinstance(raw_content_obj, str):  # Default for other text-based content
            item_content_html = _generate_html_paragraphs(raw_content_obj)
        else:
            item_content_html = "<p><em>Content format not recognized for this item.</em></p>"

        html_content += f'<div class="{" ".join(item_classes)}">\n'
        html_content += f'<h2>{html.escape(title)}</h2>\n'
        html_content += f'<div class="content-wrapper">{item_content_html}</div>\n'  # Added wrapper
        html_content += '</div>\n'

    html_content += """
    </div>
</body>
</html>
"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query_part = "".join(c if c.isalnum() else "_" for c in artifacts.get("original_user_query", "report")[:30])
    html_filename = f"bento_report_{safe_query_part}_{timestamp}.html"
    html_filepath = os.path.join(TEMP_DATA_STORAGE, html_filename)

    try:
        with open(html_filepath, "w", encoding='utf-8') as f:
            f.write(html_content)
        return html_filepath
    except Exception as e:
        st.error(f"Error writing HTML report to file: {e}")
        return None


def get_content_from_path_helper(file_path, default_message="Not available."):
    """
    Safely reads content from a file path.
    Returns the file content as a string or a default message if an error occurs or file not found.
    """
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    return default_message


def compile_report_text_for_html_generation(artifacts, cdo_initial_report, data_summary_dict, main_df=None):
    """
    Compiles all necessary text pieces and data objects for the HTML generation.
    Now directly passes DataFrames for tables/charts where appropriate.
    """
    critique_text = get_content_from_path_helper(artifacts.get("generated_critique_path"), "Critique not available.")
    generated_report_text = get_content_from_path_helper(artifacts.get("generated_report_path"),
                                                         "Generated textual report not available.")

    # --- AI Analysis Artifacts ---
    # Primary table from AI analysis (could be a DataFrame or path to CSV)
    primary_analysis_table_obj = "No primary data table generated by AI analysis."
    if artifacts.get("executed_dataframe_result") is not None and isinstance(artifacts.get("executed_dataframe_result"),
                                                                             pd.DataFrame):
        primary_analysis_table_obj = artifacts.get("executed_dataframe_result")
    elif artifacts.get("executed_data_path") and "table_result" in os.path.basename(
            artifacts.get("executed_data_path", "")):
        try:  # Attempt to read if it's a path to a general table
            primary_analysis_table_obj = pd.read_csv(artifacts.get("executed_data_path"))
        except:  # Fallback to path content if read fails
            primary_analysis_table_obj = get_content_from_path_helper(artifacts.get("executed_data_path"))

    # Data for the primary visualization (Chart.js if no image, could be DataFrame or path to CSV)
    visualization_data_obj = "No specific data table for chart provided by AI."
    # Check if plot_data_df was saved by the AI code execution
    if artifacts.get("plot_specific_data_df") is not None and isinstance(artifacts.get("plot_specific_data_df"),
                                                                         pd.DataFrame):
        visualization_data_obj = artifacts.get("plot_specific_data_df")
    elif artifacts.get("executed_data_path") and "plot_data_for" in os.path.basename(
            artifacts.get("executed_data_path", "")):
        try:  # Attempt to read if it's a path to plot data
            visualization_data_obj = pd.read_csv(artifacts.get("executed_data_path"))
        except:  # Fallback
            visualization_data_obj = get_content_from_path_helper(artifacts.get("executed_data_path"))

    report_parts = {
        "ANALYSIS_GOAL": artifacts.get("original_user_query", "Not specified."),
        "DATA_SNAPSHOT_CDO_REPORT": cdo_initial_report if cdo_initial_report else "CDO initial report not available.",
        "USER_PROVIDED_DESCRIPTIONS": data_summary_dict.get("user_provided_column_descriptions",
                                                            "User descriptions not provided."),
        "KEY_DATA_QUALITY_ALERT": "",  # Will be populated below
        "DATA_PREPROCESSING_NOTE": "Data used as provided. Preprocessing steps, if any, were part of the direct analysis query or detailed in the CDO report. Standard data loading and type inference were performed.",
        "VISUALIZATION_CHART_OR_IMAGE": visualization_data_obj,  # This will be a DataFrame or string
        "PRIMARY_ANALYSIS_TABLE": primary_analysis_table_obj,  # This will be a DataFrame or string
        "ACTIONABLE_INSIGHTS_FROM_REPORT": generated_report_text,
        "CRITIQUE_SUMMARY": critique_text,  # Could be split later if needed
        "SUGGESTIONS_FOR_ENHANCED_ANALYSIS": critique_text  # Placeholder, could be a separate LLM call
    }

    missing_data_alerts_text = []
    if data_summary_dict and data_summary_dict.get("missing_values_per_column"):
        for col, count in data_summary_dict["missing_values_per_column"].items():
            if count > 0:
                num_rows = data_summary_dict.get("num_rows", 1)
                percentage = (count / num_rows) * 100 if num_rows > 0 else 0
                missing_data_alerts_text.append(
                    f"- Column '{html.escape(col)}': {count} missing values ({percentage:.2f}%).")
    if not missing_data_alerts_text:
        missing_data_alerts_text.append(
            "- No significant missing data points identified in the automated summary, or data quality was primarily qualitative in the CDO report.")
    report_parts["KEY_DATA_QUALITY_ALERT"] = "\n".join(missing_data_alerts_text)

    # --- Add Data Quality Dashboard Elements (as DataFrames or strings) ---
    if main_df is not None and not main_df.empty:
        quality_assessment_df = get_column_quality_assessment(main_df.copy())
        report_parts[
            "DQ_COLUMN_ASSESSMENT_TABLE"] = quality_assessment_df if not quality_assessment_df.empty else "Column assessment not available."

        numeric_cols_dq = main_df.select_dtypes(include=np.number).columns.tolist()
        # MODIFIED: Data for these charts is still prepared, but they won't be rendered if keys are removed from generate_bento_html_report_python
        if numeric_cols_dq:
            first_numeric_col = numeric_cols_dq[0]
            col_data_numeric = main_df[first_numeric_col].dropna()
            if not col_data_numeric.empty:
                counts, bins = np.histogram(col_data_numeric, bins=10)
                hist_df = pd.DataFrame(
                    {'Bin': [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)], 'Frequency': counts})
                report_parts[
                    "DQ_NUMERIC_DIST_CHART"] = hist_df  # Still add to dict, generate_bento will skip if key not in its ordered_keys
            else:
                report_parts[
                    "DQ_NUMERIC_DIST_CHART"] = f"No data in first numeric column ('{html.escape(first_numeric_col)}') for distribution chart."
        else:
            report_parts["DQ_NUMERIC_DIST_CHART"] = "No numeric columns found for distribution chart."

        categorical_cols_dq = main_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols_dq:
            first_cat_col = categorical_cols_dq[0]
            col_data_cat = main_df[first_cat_col].dropna()
            if not col_data_cat.empty:
                cat_counts = col_data_cat.value_counts().head(10)  # Top 10 for chart
                cat_df = cat_counts.reset_index()
                cat_df.columns = [first_cat_col, 'Count']  # Changed to 'Count' for clarity in bar chart
                report_parts["DQ_CATEGORICAL_DIST_CHART"] = cat_df  # Still add to dict
            else:
                report_parts[
                    "DQ_CATEGORICAL_DIST_CHART"] = f"No data in first categorical column ('{html.escape(first_cat_col)}') for distribution chart."
        else:
            report_parts["DQ_CATEGORICAL_DIST_CHART"] = "No categorical columns found for distribution chart."

        if len(numeric_cols_dq) >= 2:
            corr_matrix = main_df[numeric_cols_dq].corr().round(2)
            report_parts["DQ_CORRELATION_MATRIX_TABLE"] = corr_matrix
        else:
            report_parts["DQ_CORRELATION_MATRIX_TABLE"] = "Not enough numeric columns for correlation matrix."
    else:  # main_df is None or empty
        default_msg = "Main data not available for this section."
        report_parts["DQ_COLUMN_ASSESSMENT_TABLE"] = default_msg
        report_parts["DQ_NUMERIC_DIST_CHART"] = default_msg
        report_parts["DQ_CATEGORICAL_DIST_CHART"] = default_msg
        report_parts["DQ_CORRELATION_MATRIX_TABLE"] = default_msg
    return report_parts


# --- Streamlit App UI ---
st.set_page_config(page_title="AI CSV Analyst v3.2 (Col Descs)", layout="wide")
st.title("🤖 AI CSV Analyst v3.2")
st.caption(
    "Upload CSV & optional column descriptions (.txt), review Data Quality, explore, then optionally run CDO Workflow for AI analysis.")

# --- Initialize Session State Variables ---
# Basic app state
if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant",
                                                                     "content": "Hello! Select models, upload CSV (and optionally a .txt file with column descriptions) to start."}]
if "current_dataframe" not in st.session_state: st.session_state.current_dataframe = None
if "data_summary" not in st.session_state: st.session_state.data_summary = None  # This will hold the comprehensive summary
if "data_source_name" not in st.session_state: st.session_state.data_source_name = None
if "desc_file_name" not in st.session_state: st.session_state.desc_file_name = None  # For the description file
if "current_analysis_artifacts" not in st.session_state: st.session_state.current_analysis_artifacts = {}

# Model selection
if "selected_worker_model" not in st.session_state: st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state: st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL

# Langchain memory
if "lc_memory" not in st.session_state: st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history",
                                                                                              return_messages=False,
                                                                                              input_key="user_query")
# CDO Workflow specific state
if "cdo_initial_report_text" not in st.session_state: st.session_state.cdo_initial_report_text = None
if "other_perspectives_text" not in st.session_state: st.session_state.other_perspectives_text = None
if "strategy_text" not in st.session_state: st.session_state.strategy_text = None
if "cdo_workflow_stage" not in st.session_state: st.session_state.cdo_workflow_stage = None

# Action trigger flags
if "trigger_code_generation" not in st.session_state: st.session_state.trigger_code_generation = False
if "trigger_report_generation" not in st.session_state: st.session_state.trigger_report_generation = False
if "trigger_judging" not in st.session_state: st.session_state.trigger_judging = False
if "trigger_html_export" not in st.session_state: st.session_state.trigger_html_export = False

# State for report/judging targets
if "report_target_data_path" not in st.session_state: st.session_state.report_target_data_path = None
if "report_target_plot_path" not in st.session_state: st.session_state.report_target_plot_path = None
if "report_target_query" not in st.session_state: st.session_state.report_target_query = None

# --- Prompt Templates ---
# Note: Prompts now implicitly benefit from user_provided_column_descriptions being part of the {data_summary}
cdo_initial_data_description_prompt_template = PromptTemplate(input_variables=["data_summary", "chat_history"],
                                                              template="""You are the Chief Data Officer (CDO). A user has uploaded a CSV file and potentially a text file describing the columns.
Data Summary (for context, may include 'user_provided_column_descriptions'):
{data_summary}

CDO, your first task is to provide an initial description of the dataset. This should include:
1.  A brief overview similar to `df.info()` (column names, non-null counts, dtypes).
2.  Reference and incorporate the 'user_provided_column_descriptions' from the data summary if they exist. Explain how these descriptions clarify the data.
3.  Your inferred meaning or common interpretation for each variable/column, especially for those not covered by user descriptions.
4.  A preliminary assessment of data quality (e.g., obvious missing data patterns, potential outliers you notice from the summary, data type consistency), considering user descriptions for context.
This description will be shared with the other department heads.
Conversation History (for context, if any):
{chat_history}

Detailed Initial Description by CDO (incorporating user descriptions if provided):""")

individual_perspectives_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report"], template="""You are a panel of expert department heads, including the CDO.
A user has uploaded a CSV file (and possibly column descriptions), and the Chief Data Officer (CDO) has provided an initial data description and quality assessment.
Data Summary (Original, may include 'user_provided_column_descriptions'):
{data_summary}

CDO's Initial Data Description & Quality Report (which should have incorporated user descriptions):
--- BEGIN CDO REPORT ---
{cdo_initial_report}
--- END CDO REPORT ---

Based on BOTH the original data summary (including any user descriptions) AND the CDO's initial report, provide a detailed perspective from each of the following roles (CEO, CFO, CTO, COO, CMO).
For each role, outline 2-3 specific questions they would now ask, analyses they would want to perform, or observations they would make, considering the CDO's findings and any user-provided column meanings.
The CDO should also provide a brief perspective here, perhaps by reiterating 1-2 critical data quality points or highlighting features clarified by user descriptions.

Structure your response clearly, with each role's perspective under a bolded heading (e.g., **CEO Perspective:**).
* **CEO (Chief Executive Officer):**
* **CFO (Chief Financial Officer):**
* **CTO (Chief Technology Officer):**
* **COO (Chief Operating Officer):**
* **CMO (Chief Marketing Officer):**
* **CDO (Reiterating Key Points, referencing user descriptions where relevant):**

Conversation History (for context, if any):
{chat_history}

Detailed Perspectives from Department Heads (informed by CDO's initial report and user column descriptions):""")

synthesize_analysis_suggestions_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report", "generated_perspectives_from_others"],
    template="""You are the Chief Data Officer (CDO) of the company.
A user has uploaded a CSV file. You have already performed an initial data description (incorporating user-provided column meanings if available).
Subsequently, other department heads have provided their perspectives based on your initial findings and the data summary.

Original Data Summary (may include 'user_provided_column_descriptions'):
{data_summary}

Your Initial Data Description & Quality Report:
--- BEGIN YOUR INITIAL CDO REPORT ---
{cdo_initial_report}
--- END YOUR INITIAL CDO REPORT ---

Perspectives from other Department Heads (CEO, CFO, CTO, COO, CMO):
--- BEGIN OTHER PERSPECTIVES ---
{generated_perspectives_from_others}
--- END OTHER PERSPECTIVES ---

Your task is to synthesize all this information into a concise list of **5 distinct and actionable analysis strategy suggestions** for the user.
These suggestions must prioritize analyses that result in clear visualizations, well-structured tables, or concise descriptive summaries.
Ensure your suggestions leverage the understanding gained from user-provided column descriptions where applicable.
Present these 5 suggestions as a numbered list. Each suggestion should clearly state the type of analysis.

Conversation History (for context, if any):
{chat_history}

Final 5 Analysis Strategy Suggestions (Synthesized by the CDO, focusing on visualizations, tables, descriptive methods, and incorporating all prior inputs including user descriptions):""")

# TEMP_DATA_STORAGE_PROMPT will be replaced with the actual path
TEMP_DATA_STORAGE_PROMPT_PLACEHOLDER = "{TEMP_DATA_STORAGE_PATH_FOR_PROMPT}"
code_generation_prompt_template = PromptTemplate(input_variables=["data_summary", "user_query", "chat_history"],
                                                 template=f"""You are an expert Python data analysis assistant.
Data Summary (may include 'user_provided_column_descriptions'):
{'{data_summary}'}
User Query: "{{user_query}}"
Previous Conversation (for context):
{'{chat_history}'}

Your task is to generate a Python script to perform the requested analysis on a pandas DataFrame named `df`.
**Crucial Instructions for `analysis_result` and `plot_data_df`:**
1.  **`analysis_result` MUST BE SET**: The primary result of your script MUST be assigned to a variable named `analysis_result`.
2.  **For Plots**:
    a.  Save the plot to the designated temporary storage. The path is constructed as `os.path.join(TEMP_DATA_STORAGE, 'your_plot_filename.png')`. You have `TEMP_DATA_STORAGE` variable available in your execution scope.
    b.  `analysis_result` MUST be set to *only the filename string* of the plot (e.g., 'my_plot.png', NOT the full path).
    c.  Create a pandas DataFrame named `plot_data_df` containing ONLY the data that was actually visualized in the plot. If the entire `df` was used, then `plot_data_df = df.copy()`. If no specific subset of data was directly plotted (e.g., complex aggregation within plotting function), set `plot_data_df = None`.
3.  **For Tables (DataFrames/Series)**:
    a.  Assign the resulting pandas DataFrame or Series to `analysis_result`.
    b.  Set `plot_data_df = analysis_result.copy()` (if `analysis_result` is a DataFrame/Series) or `None` if not applicable.
4.  **For Text Results**:
    a.  Assign the textual result (string) to `analysis_result`.
    b.  Set `plot_data_df = None`.
5.  **Default `analysis_result`**: If the query is general or doesn't ask for a specific output type, `analysis_result` can be a descriptive string or `df.head()`. In this case, `plot_data_df` should be `df.head().copy()` or `None`.
6.  **Imports**: Your script MUST import all necessary libraries (e.g., `import pandas as pd`, `import matplotlib.pyplot as plt`, `import seaborn as sns`, `import numpy as np`, `import os`). The `TEMP_DATA_STORAGE` variable (string path) will be available in your execution scope.
7.  **Directory Creation**: Before saving a plot, ensure the directory exists using `os.makedirs(TEMP_DATA_STORAGE, exist_ok=True)`. This is usually handled by the execution engine but good practice.

**Safety Net - Fallback within your generated script (include this at the end of your script logic):**
```python
# --- Safety Net ---
if 'analysis_result' not in locals(): # Check if analysis_result was defined by your core logic
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Script completed. No specific 'analysis_result' was set by the AI's main logic. Displaying df.head() as a default."
        plot_data_df = df.head().copy()
    else:
        analysis_result = "Script completed. No specific 'analysis_result' was set, and no DataFrame 'df' was available or it was empty."
if 'plot_data_df' not in locals(): # Ensure plot_data_df is always defined
    plot_data_df = None
```
Output only raw Python code, without any surrounding text or markdown fences.
Python code:""")

report_generation_prompt_template = PromptTemplate(
    input_variables=["table_data_csv_or_text", "original_data_summary", "user_query_that_led_to_data", "chat_history",
                     "plot_info_if_any"],
    template="""You are an insightful data analyst. Generate a textual report based on the provided data and context.
Original Data Summary (may include 'user_provided_column_descriptions'):
{original_data_summary}

User Query that led to this data/plot: "{user_query_that_led_to_data}"
Plot Information (if applicable, 'N/A' otherwise): {plot_info_if_any}
Chat History (for context):
{chat_history}

Analysis Result Data (content of CSV, or textual output, or 'N/A if plot-only report'):
```
{table_data_csv_or_text}
```

**Report Structure:**
* **1. Executive Summary (1-2 sentences):** Concisely state the main conclusion or finding from the `Analysis Result Data` or plot.
* **2. Purpose of this Analysis (1 sentence):** Briefly restate the user's goal as understood from their query.
* **3. Key Observations (Bulleted list, 2-4 points):** Detail specific, quantified observations directly from the `Analysis Result Data` or visual trends from the plot. If user descriptions were relevant, mention how they aided interpretation.
* **4. Actionable Insights (1-2 insights):** What does this mean in the broader context? What actions could be taken based on these observations?
* **5. Data Focus & Limitations:** Clearly state that this report is *solely* based on the "Analysis Result Data" and/or "Plot Information" provided above. Mention if the data is a sample, aggregated, etc., if known.

**Tone:** Professional, clear, objective. Do NOT use phrases like "the CSV provided shows..." or "As we can see in the plot...". Instead, directly state the findings.
Report:""")

judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_content_for_judge", "report_text_content", "original_user_query",
                     "data_summary",
                     "plot_image_path", "plot_info_for_judge"], template="""You are an expert data science reviewer. Evaluate the AI assistant's artifacts based on the user's query and the provided data context.
Original User Query: "{original_user_query}"
Original Data Summary (may include 'user_provided_column_descriptions'):
{data_summary}

--- ARTIFACTS TO CRITIQUE ---
1.  **Python Code Generated by Worker AI:**
    ```python
{python_code}
    ```
2.  **Data Produced by Executing the Code (content of CSV, or textual output from `analysis_result`):**
    ```
{data_content_for_judge}
    ```
    **Plot Information:** {plot_info_for_judge} (This might state if a plot file '{plot_image_path}' exists, or if `plot_data_df` was provided for a chart.)

3.  **Textual Report Generated by Worker AI (if any):**
    ```text
{report_text_content}
    ```
--- END ARTIFACTS ---

**Critique Guidelines:**
1.  **Code Quality & Adherence (Crucial):**
    * Correctness: Does the code run without errors (assuming valid inputs)? Does it logically address the query?
    * Efficiency & Readability: Is the code reasonably efficient and easy to understand?
    * Best Practices: Use of appropriate libraries?
    * **`analysis_result` and `plot_data_df` Usage:**
        * Did the code correctly set `analysis_result` to the filename (string) for plots?
        * Did it correctly set `analysis_result` to a DataFrame/Series for tables, or a string for text?
        * Was `plot_data_df` appropriately created (data used in plot, or a copy of `analysis_result` for tables, or `None`)?
    * Plot Saving: If a plot was made, did the code attempt to save it to the `TEMP_DATA_STORAGE` directory (e.g., using `os.path.join(TEMP_DATA_STORAGE, 'filename.png')`)?

2.  **Data Analysis Quality:**
    * Relevance: Is the analysis (code output, plot) relevant to the user's query and the dataset's nature (considering user descriptions if available in the summary)?
    * Accuracy: Are the calculations or data manipulations likely correct?
    * Methods: Were appropriate analytical methods or visualizations chosen?
    * `plot_data_df` Content: If `plot_data_df` was generated, does its content logically correspond to what the plot should be visualizing?

3.  **Plot Quality (if a plot was generated, check `{plot_image_path}` or `plot_info_for_judge`):**
    * Appropriateness: Is the plot type suitable for the data and query?
    * Clarity: Is the plot well-labeled (title, axes, legend if needed)? Easy to understand?

4.  **Report Quality (if a report was generated):**
    * Clarity & Conciseness: Is the report easy to understand and to the point?
    * Insightfulness: Does it offer meaningful insights derived *directly* from the `data_content_for_judge` and/or plot?
    * Addresses Query: Does the report effectively answer or address the user's original query?
    * Objectivity: Does it stick to the data, or does it speculate?

5.  **Overall Effectiveness & Suggestions:**
    * Rate how well the AI assistant addressed the user's query (1-10, 10 being excellent).
    * Provide 1-2 specific, actionable suggestions for the Worker AI to improve its response for *this specific query* or similar future queries.

Critique:""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    st.session_state.selected_worker_model = st.selectbox("Select Worker Model:", AVAILABLE_MODELS,
                                                          index=AVAILABLE_MODELS.index(
                                                              st.session_state.selected_worker_model))
    st.session_state.selected_judge_model = st.selectbox("Select Judge Model:", AVAILABLE_MODELS,
                                                         index=AVAILABLE_MODELS.index(
                                                             st.session_state.selected_judge_model))
    st.header("📤 Upload Data")
    uploaded_csv_file = st.file_uploader("Upload your CSV file:", type="csv", key="csv_uploader")
    uploaded_desc_file = st.file_uploader("Optional: Upload column descriptions (.txt):", type="txt",
                                          key="desc_uploader")

    if uploaded_csv_file is not None:
        # Determine if a re-process is needed
        reprocess = False
        if st.session_state.get("data_source_name") != uploaded_csv_file.name:
            reprocess = True
        if uploaded_desc_file and st.session_state.get("desc_file_name") != uploaded_desc_file.name:
            reprocess = True
        if not uploaded_desc_file and st.session_state.get("desc_file_name") is not None:  # If desc file was removed
            reprocess = True

        if uploaded_desc_file is None and st.session_state.get(
                "desc_file_name") is None and not reprocess:  # if only CSV is uploaded and it's the same, no reprocess unless forced
            pass  # No change if only CSV is present and it's the same one, unless forced by CSV name change
        elif uploaded_desc_file is not None and st.session_state.get(
                "desc_file_name") == uploaded_desc_file.name and st.session_state.get(
                "data_source_name") == uploaded_csv_file.name:
            pass  # No change if both files are present and same
        else:
            reprocess = True

        if reprocess:
            with st.spinner("Processing CSV and descriptions..."):
                if load_csv_and_get_summary(uploaded_csv_file, uploaded_desc_file):  # Pass both files
                    st.success(f"CSV '{st.session_state.data_source_name}' processed.")
                    if uploaded_desc_file:
                        st.success(f"Description file '{st.session_state.desc_file_name}' processed.")
                    else:
                        st.info("No description file provided or it was removed.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Processed '{st.session_state.data_source_name}'" + (
                                                          f" with descriptions from '{st.session_state.desc_file_name}'." if st.session_state.desc_file_name else ".") + " View Data Quality Dashboard or other tabs."})
                    st.rerun()
                else:
                    st.error("Failed to process CSV and/or descriptions.")

    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(
            f"**{st.session_state.data_source_name}** ({len(st.session_state.current_dataframe)} rows x {len(st.session_state.current_dataframe.columns)} columns)")
        if st.session_state.desc_file_name:
            st.write(f"With descriptions from: **{st.session_state.desc_file_name}**")

        if st.button("Clear Loaded Data & Chat", key="clear_data_btn"):
            keys_to_reset = [
                "current_dataframe", "data_summary", "data_source_name", "desc_file_name",
                "current_analysis_artifacts", "messages", "lc_memory",
                "cdo_initial_report_text", "other_perspectives_text", "strategy_text", "cdo_workflow_stage",
                "trigger_code_generation", "trigger_report_generation", "trigger_judging", "trigger_html_export",
                "report_target_data_path", "report_target_plot_path", "report_target_query"
            ]
            for key in keys_to_reset:
                if key in st.session_state: del st.session_state[key]

            st.session_state.messages = [{"role": "assistant", "content": "Data and chat reset. Upload a new CSV."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}

            cleaned_files_count = 0
            if os.path.exists(TEMP_DATA_STORAGE):
                for item_name in os.listdir(TEMP_DATA_STORAGE):
                    item_path = os.path.join(TEMP_DATA_STORAGE, item_name)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                            cleaned_files_count += 1
                        # Optionally, remove subdirectories if your app creates them within TEMP_DATA_STORAGE
                        # elif os.path.isdir(item_path):
                        #     shutil.rmtree(item_path) # Be careful with rmtree
                    except Exception as e:
                        st.warning(f"Could not remove temp file {item_path}: {e}")
            st.success(f"Data, chat, and {cleaned_files_count} temp files cleared.")
            st.rerun()

    st.markdown("---")
    st.info(
        f"Worker: **{st.session_state.selected_worker_model}**\n\nJudge: **{st.session_state.selected_judge_model}**")
    st.info(f"Temp files in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning("⚠️ **Security Note:** Uses `exec()` for AI-generated code. Demo purposes ONLY.")

# --- Main Area with Tabs (only if data is loaded) ---
if st.session_state.current_dataframe is not None:
    tab_titles = ["📊 Data Quality Dashboard", "🔍 Data Explorer", "👨‍💼 CDO Workflow", "💬 AI Analysis Chat"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:  # Data Quality Dashboard
        generate_data_quality_dashboard(st.session_state.current_dataframe.copy())
    with tab2:  # Data Explorer
        st.header("🔍 Data Explorer")
        if st.session_state.data_summary:
            with st.expander("View Full Data Summary (JSON)", expanded=False):
                st.json(st.session_state.data_summary)
            if st.session_state.data_summary.get("user_provided_column_descriptions"):
                with st.expander("View User-Provided Column Descriptions", expanded=True):
                    st.markdown(st.session_state.data_summary["user_provided_column_descriptions"])
        else:
            st.write("No data summary available.")
        with st.expander(f"View DataFrame Head (First 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.head())
        with st.expander(f"View DataFrame Tail (Last 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.tail())
    with tab3:  # CDO Workflow
        st.header("👨‍💼 CDO-led Analysis Workflow")
        st.markdown(
            "Initiate an AI-driven analysis: CDO describes data (using your descriptions if provided), VPs discuss, CDO synthesizes strategy.")

        if st.button("🚀 Start CDO Analysis Workflow", key="start_cdo_workflow_btn"):
            st.session_state.cdo_workflow_stage = "initial_description"
            st.session_state.cdo_initial_report_text = None
            st.session_state.other_perspectives_text = None
            st.session_state.strategy_text = None
            st.session_state.messages.append({"role": "assistant",
                                              "content": f"Starting CDO initial data description with **{st.session_state.selected_worker_model}**..."})
            st.session_state.lc_memory.save_context(
                {"user_query": f"User initiated CDO workflow for {st.session_state.data_source_name}."},
                {"output": "Requesting CDO initial description."})
            st.rerun()

        worker_llm = get_llm_instance(st.session_state.selected_worker_model)
        current_stage = st.session_state.get("cdo_workflow_stage")

        if current_stage == "initial_description":
            if worker_llm and st.session_state.data_summary:
                with st.spinner(f"CDO ({st.session_state.selected_worker_model}) performing initial description..."):
                    try:
                        memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                        prompt_inputs = {"data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                         "chat_history": memory_ctx.get("chat_history", "")}
                        response = worker_llm.invoke(
                            cdo_initial_data_description_prompt_template.format_prompt(**prompt_inputs))
                        st.session_state.cdo_initial_report_text = response.content if hasattr(response,
                                                                                               'content') else response.get(
                            'text', "Error generating CDO report.")
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": f"**CDO's Initial Description (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.cdo_initial_report_text}"})
                        st.session_state.lc_memory.save_context(
                            {"user_query": "System: CDO initial description requested."},
                            {"output": f"CDO Report: {st.session_state.cdo_initial_report_text[:100]}..."})
                        st.session_state.cdo_workflow_stage = "departmental_perspectives"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in CDO description: {e}");
                        st.session_state.cdo_workflow_stage = None
            else:
                st.error("Worker LLM or data summary unavailable for CDO workflow.")

        if current_stage == "departmental_perspectives" and st.session_state.cdo_initial_report_text:
            with st.spinner(f"Department Heads ({st.session_state.selected_worker_model}) discussing..."):
                try:
                    memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                    prompt_inputs = {"data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                     "chat_history": memory_ctx.get("chat_history", ""),
                                     "cdo_initial_report": st.session_state.cdo_initial_report_text}
                    response = worker_llm.invoke(individual_perspectives_prompt_template.format_prompt(**prompt_inputs))
                    st.session_state.other_perspectives_text = response.content if hasattr(response,
                                                                                           'content') else response.get(
                        'text', "Error generating perspectives.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**Departmental Perspectives (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.other_perspectives_text}"})
                    st.session_state.lc_memory.save_context({"user_query": "System: VPs' perspectives requested."}, {
                        "output": f"VPs' Perspectives: {st.session_state.other_perspectives_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "strategy_synthesis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in VPs' perspectives: {e}");
                    st.session_state.cdo_workflow_stage = None

        if current_stage == "strategy_synthesis" and st.session_state.other_perspectives_text:
            with st.spinner(f"CDO ({st.session_state.selected_worker_model}) synthesizing strategy..."):
                try:
                    memory_ctx = st.session_state.lc_memory.load_memory_variables({})
                    prompt_inputs = {"data_summary": json.dumps(st.session_state.data_summary, indent=2),
                                     "chat_history": memory_ctx.get("chat_history", ""),
                                     "cdo_initial_report": st.session_state.cdo_initial_report_text,
                                     "generated_perspectives_from_others": st.session_state.other_perspectives_text}
                    response = worker_llm.invoke(
                        synthesize_analysis_suggestions_prompt_template.format_prompt(**prompt_inputs))
                    st.session_state.strategy_text = response.content if hasattr(response, 'content') else response.get(
                        'text', "Error generating strategy.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**CDO's Final Strategy (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.strategy_text}\n\nGo to 'AI Analysis Chat' tab for specific analyses."})
                    st.session_state.lc_memory.save_context({"user_query": "System: CDO strategy synthesis requested."},
                                                            {
                                                                "output": f"CDO Strategy: {st.session_state.strategy_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "completed"
                    st.success("CDO Workflow Completed!");
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in CDO strategy synthesis: {e}");
                    st.session_state.cdo_workflow_stage = None

        if st.session_state.cdo_initial_report_text:
            with st.expander("CDO's Initial Data Description", expanded=(
                    current_stage in ["initial_description", "departmental_perspectives", "strategy_synthesis",
                                      "completed"])):
                st.markdown(st.session_state.cdo_initial_report_text)
        if st.session_state.other_perspectives_text:
            with st.expander("Departmental Perspectives", expanded=(
                    current_stage in ["departmental_perspectives", "strategy_synthesis", "completed"])):
                st.markdown(st.session_state.other_perspectives_text)
        if st.session_state.strategy_text:
            with st.expander("CDO's Final Analysis Strategy",
                             expanded=(current_stage in ["strategy_synthesis", "completed"])):
                st.markdown(st.session_state.strategy_text)

    with tab4:  # AI Analysis Chat
        st.header("💬 AI Analysis Chat")
        st.caption("Interact with Worker AI for analyses, reports, and critiques from Judge AI.")
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "executed_result" in message:
                        exec_res = message["executed_result"]
                        res_type = exec_res.get("type")
                        orig_query = message.get("original_user_query",
                                                 st.session_state.current_analysis_artifacts.get("original_user_query",
                                                                                                 "Unknown query"))

                        if res_type == "table":
                            if exec_res.get("data_path") and os.path.exists(exec_res["data_path"]):
                                try:
                                    df_disp = pd.read_csv(exec_res["data_path"])
                                    st.dataframe(df_disp)
                                    if st.button(f"📊 Generate Report for Table##{i}", key=f"rep_tbl_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True
                                        st.session_state.report_target_data_path = exec_res["data_path"]
                                        st.session_state.report_target_plot_path = None
                                        st.session_state.report_target_query = orig_query
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error displaying table: {e}")
                            elif exec_res.get("dataframe_result") is not None:  # If DataFrame was passed directly
                                st.dataframe(exec_res.get("dataframe_result"))
                                # TODO: Decide if direct DataFrame result needs a report button / how to handle its data path
                                st.caption("Table displayed from direct DataFrame result.")


                        elif res_type == "plot":
                            if exec_res.get("plot_path") and os.path.exists(exec_res["plot_path"]):
                                st.image(exec_res["plot_path"])
                                report_button_label = "📄 Generate Report for Plot"
                                target_data_for_report = None
                                if exec_res.get("data_path") and os.path.exists(exec_res["data_path"]):
                                    report_button_label += " Data"
                                    target_data_for_report = exec_res["data_path"]

                                if st.button(f"{report_button_label}##{i}", key=f"rep_plot_btn_{i}_tab4"):
                                    st.session_state.trigger_report_generation = True
                                    st.session_state.report_target_data_path = target_data_for_report
                                    st.session_state.report_target_plot_path = exec_res["plot_path"]
                                    st.session_state.report_target_query = orig_query
                                    st.rerun()
                            else:
                                st.warning(f"Plot image not found: {exec_res.get('plot_path', 'Path not specified')}")

                        elif res_type == "text":
                            st.markdown(f"**Output:**\n```\n{exec_res.get('value', 'No text output.')}\n```")

                        elif res_type == "report_generated":
                            if exec_res.get("report_path") and os.path.exists(exec_res["report_path"]):
                                st.markdown(f"_Report generated: `{os.path.abspath(exec_res['report_path'])}`_")

                        # Judge button logic
                        artifacts_for_judge = st.session_state.get("current_analysis_artifacts", {})
                        can_judge = artifacts_for_judge.get("generated_code") and \
                                    (artifacts_for_judge.get("executed_data_path") or \
                                     artifacts_for_judge.get("executed_dataframe_result") is not None or \
                                     artifacts_for_judge.get("plot_image_path") or \
                                     artifacts_for_judge.get("executed_text_output") or \
                                     (res_type == "text" and exec_res.get("value")))
                        if can_judge:
                            if st.button(f"⚖️ Judge This Analysis##{i}", key=f"judge_btn_{i}_tab4"):
                                st.session_state.trigger_judging = True
                                st.rerun()

                    if message["role"] == "assistant" and "critique_text" in message:
                        with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                            st.markdown(message["critique_text"])
                        if st.button(f"📄 Export Full Analysis to PDF##{i}", key=f"pdf_exp_btn_{i}_tab4"):
                            # Ensure artifacts are up-to-date for PDF export
                            pdf_artifacts = st.session_state.current_analysis_artifacts.copy()
                            if not pdf_artifacts.get("executed_dataframe_result") and pdf_artifacts.get(
                                    "executed_data_path"):
                                try:  # Try to load df if only path is there
                                    pdf_artifacts["executed_dataframe_result"] = pd.read_csv(
                                        pdf_artifacts["executed_data_path"])
                                except:
                                    pass  # Ignore if fails, PDF func will handle

                            with st.spinner("Generating PDF..."):
                                pdf_path = export_analysis_to_pdf(pdf_artifacts)  # Pass potentially updated artifacts
                                if pdf_path and os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as f_pdf:
                                        st.download_button("Download PDF Report", f_pdf, os.path.basename(pdf_path),
                                                           "application/pdf",
                                                           key=f"dl_pdf_{i}_tab4_{datetime.datetime.now().timestamp()}")
                                    st.success(f"PDF report ready: {os.path.basename(pdf_path)}")
                                else:
                                    st.error("Failed to generate PDF.")

                        if st.button(f"📄 Export Bento HTML Report##{i}", key=f"html_exp_btn_{i}_tab4"):
                            st.session_state.trigger_html_export = True
                            st.rerun()

        if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)...",
                                       key="user_query_input_tab4"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
                st.warning("Please upload and process a CSV file first (via sidebar).")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I need CSV data. Please upload a file."})
            else:
                worker_llm_chat = get_llm_instance(st.session_state.selected_worker_model)
                if not worker_llm_chat:
                    st.error(f"Worker model {st.session_state.selected_worker_model} not initialized.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Worker LLM ({st.session_state.selected_worker_model}) unavailable."})
                else:
                    st.session_state.current_analysis_artifacts = {"original_user_query": user_query}
                    st.session_state.trigger_code_generation = True
                    st.rerun()
else:
    st.info(
        "👋 Welcome! Please upload a CSV file (and optionally a .txt with column descriptions) using the sidebar to get started.")
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message(message["role"]): st.markdown(message["content"])

# --- Code Generation Logic ---
if st.session_state.get("trigger_code_generation", False):
    st.session_state.trigger_code_generation = False
    user_query = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        gen_code_str = ""
        msg_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** generating code for: '{html.escape(user_query)}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} generating Python code..."):
            try:
                worker_llm_code_gen = get_llm_instance(st.session_state.selected_worker_model)
                mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                data_sum_prompt = json.dumps(st.session_state.data_summary,
                                             indent=2) if st.session_state.data_summary else "{}"

                # Replace placeholder in template with actual path for the LLM
                # Ensure TEMP_DATA_STORAGE is an absolute path for clarity if used directly in prompt,
                # or ensure the LLM knows it's a relative path it should use with os.path.join
                # For this version, TEMP_DATA_STORAGE is passed to exec scope.
                # The prompt instructs to use os.path.join(TEMP_DATA_STORAGE, 'filename.ext')

                # The code_generation_prompt_template is already an f-string that inserts the placeholder.
                # We format it here.
                formatted_code_gen_prompt = code_generation_prompt_template.format_prompt(
                    data_summary=data_sum_prompt,
                    user_query=user_query,
                    chat_history=mem_ctx.get("chat_history", "")
                )

                response = worker_llm_code_gen.invoke(formatted_code_gen_prompt)
                gen_code_str = response.content if hasattr(response, 'content') else response.get('text', "")

                for prefix in ["```python\n", "```python", "```\n", "```"]:
                    if gen_code_str.startswith(prefix): gen_code_str = gen_code_str[len(prefix):]
                if gen_code_str.endswith("\n```"):
                    gen_code_str = gen_code_str[:-len("\n```")]
                elif gen_code_str.endswith("```"):
                    gen_code_str = gen_code_str[:-len("```")]
                gen_code_str = gen_code_str.strip()

                st.session_state.current_analysis_artifacts["generated_code"] = gen_code_str
                assist_base_content = f"🔍 **Generated Code by {st.session_state.selected_worker_model} for '{html.escape(user_query)}':**\n```python\n{gen_code_str}\n```\n"
                msg_placeholder.markdown(assist_base_content + "\n⏳ Now executing the generated code...")
            except Exception as e:
                err_msg = f"An error occurred while generating code: {html.escape(str(e))}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query},
                                                        {"output": f"Code Generation Error: {e}"})
                if msg_placeholder: msg_placeholder.empty()
                st.rerun()

        if gen_code_str:
            curr_assist_resp_msg = {"role": "assistant", "content": assist_base_content,
                                    "original_user_query": user_query}
            with st.spinner("Executing the generated Python code..."):
                # The local_scope for exec will be created inside execute_code
                # We need to retrieve it if we want to access plot_data_df from it later for artifacts
                # However, execute_code itself should populate artifacts if plot_data_df is relevant
                exec_result = code_executor.execute_code(gen_code_str, st.session_state.current_dataframe.copy())

                # Store paths/outputs/DataFrames in artifacts
                if exec_result.get("data_path"): st.session_state.current_analysis_artifacts["executed_data_path"] = \
                exec_result["data_path"]
                if exec_result.get("plot_path"): st.session_state.current_analysis_artifacts["plot_image_path"] = \
                exec_result["plot_path"]
                if exec_result.get("type") == "text" and exec_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = exec_result.get("value")
                if exec_result.get("dataframe_result") is not None:  # Store DataFrame if returned
                    st.session_state.current_analysis_artifacts["executed_dataframe_result"] = exec_result.get(
                        "dataframe_result")

                # If plot_data_df was created within execute_code's local_scope and returned (e.g. via exec_result)
                # or if we modify execute_code to return it directly.
                # For now, we assume if a plot was made, and plot_data_df was set, it's handled by execute_code's logic for saving plot_data_df.
                # Let's refine artifact storing for plot_specific_data_df
                # This relies on plot_data_df being correctly set in the exec_result or LocalCodeExecutionEngine's scope
                # A more robust way would be for execute_code to explicitly return plot_data_df if it's relevant
                # For now, let's assume if a plot was generated and data_path for plot data exists, it's the plot_specific_data
                if exec_result.get("type") == "plot" and exec_result.get(
                        "data_path") and "plot_data_for" in os.path.basename(exec_result.get("data_path")):
                    try:
                        st.session_state.current_analysis_artifacts["plot_specific_data_df"] = pd.read_csv(
                            exec_result.get("data_path"))
                    except Exception as e:
                        st.warning(f"Could not load plot_specific_data_df from {exec_result.get('data_path')}: {e}")

                llm_mem_output = ""
                if exec_result["type"] == "error":
                    curr_assist_resp_msg[
                        "content"] += f"\n⚠️ **Execution Error:**\n```\n{html.escape(exec_result['message'])}\n```"
                    if str(st.session_state.current_analysis_artifacts.get("executed_text_output", "")).startswith(
                            "Code executed, but"):
                        st.session_state.current_analysis_artifacts[
                            "executed_text_output"] = f"Execution Error: {html.escape(str(exec_result.get('final_analysis_result_value', 'Unknown error')))}"
                    llm_mem_output = f"Execution Error: {html.escape(exec_result['message'][:100])}..."
                else:
                    curr_assist_resp_msg["content"] += "\n✅ **Code Executed Successfully!**"
                    curr_assist_resp_msg["executed_result"] = exec_result
                    if exec_result.get("type") == "text":
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            exec_result.get("value", ""))

                    if exec_result.get("data_path"): curr_assist_resp_msg[
                        "content"] += f"\n💾 Data saved to: `{os.path.abspath(exec_result['data_path'])}`"
                    if exec_result.get("plot_path"): curr_assist_resp_msg[
                        "content"] += f"\n🖼️ Plot saved to: `{os.path.abspath(exec_result['plot_path'])}`"
                    if exec_result.get("data_path") and "plot_data_for" in os.path.basename(
                            exec_result.get("data_path", "")):
                        curr_assist_resp_msg["content"] += " (Data specifically for the plot was also saved)."

                    if exec_result["type"] == "table":
                        llm_mem_output = f"Generated Table: {os.path.basename(exec_result.get('data_path', 'N/A'))}"
                    elif exec_result["type"] == "plot":
                        llm_mem_output = f"Generated Plot: {os.path.basename(exec_result.get('plot_path', 'N/A'))}"
                        if exec_result.get(
                            "data_path"): llm_mem_output += f" (Plot Data: {os.path.basename(exec_result.get('data_path'))})"
                    elif exec_result["type"] == "text":
                        llm_mem_output = f"Generated Text Output: {str(exec_result.get('value', ''))[:50]}..."
                    else:
                        llm_mem_output = "Code executed, result type unknown."

                st.session_state.lc_memory.save_context(
                    {"user_query": f"{user_query}\n---Generated Code---\n{gen_code_str}\n---End Code---"},
                    {"output": llm_mem_output})
                st.session_state.messages.append(curr_assist_resp_msg)
                if msg_placeholder: msg_placeholder.empty()
                st.rerun()

# --- Report Generation Logic ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False
    data_path_rep = st.session_state.get("report_target_data_path")
    plot_path_rep = st.session_state.get("report_target_plot_path")
    query_led_to_data = st.session_state.report_target_query
    worker_llm_rep = get_llm_instance(st.session_state.selected_worker_model)

    if not worker_llm_rep or not st.session_state.data_summary or (
            not data_path_rep and not plot_path_rep):  # Original check was fine
        st.error("Cannot generate report: LLM, data summary, or target data/plot path is missing.")
    else:
        csv_or_text_content_rep = "N/A - Report is descriptive of a plot image or no specific data was provided."
        if data_path_rep and os.path.exists(data_path_rep):
            try:
                with open(data_path_rep, 'r', encoding='utf-8') as f_rep_data:
                    csv_or_text_content_rep = f_rep_data.read()
            except Exception as e_read:
                st.error(f"Error reading data file for report: {html.escape(str(e_read))}")
                st.rerun()
        elif not data_path_rep and plot_path_rep:  # Plot-only report
            st.info("Generating descriptive report for the plot.")

        plot_info_for_prompt = "N/A"
        if plot_path_rep and os.path.exists(plot_path_rep):
            plot_info_for_prompt = f"Plot image is available at '{os.path.basename(plot_path_rep)}'."
            if data_path_rep and "plot_data_for" in os.path.basename(
                    data_path_rep):  # Check if data_path_rep is for this plot
                plot_info_for_prompt += f" Associated data for plot found in '{os.path.basename(data_path_rep)}'."

        with st.chat_message("assistant"):
            rep_spinner_container = st.empty()
            rep_spinner_container.markdown(
                f"📝 **{st.session_state.selected_worker_model}** drafting report for: '{html.escape(query_led_to_data)}'...")
            with st.spinner("Generating report..."):
                try:
                    mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                    data_sum_prompt = json.dumps(st.session_state.data_summary,
                                                 indent=2) if st.session_state.data_summary else "{}"
                    prompt_inputs = {"table_data_csv_or_text": csv_or_text_content_rep,
                                     "original_data_summary": data_sum_prompt,
                                     "user_query_that_led_to_data": query_led_to_data,
                                     "chat_history": mem_ctx.get("chat_history", ""),
                                     "plot_info_if_any": plot_info_for_prompt
                                     }
                    response = worker_llm_rep.invoke(report_generation_prompt_template.format_prompt(**prompt_inputs))
                    rep_text = response.content if hasattr(response, 'content') else response.get('text',
                                                                                                  "Error generating report.")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in query_led_to_data[:30])
                    filepath = os.path.join(TEMP_DATA_STORAGE, f"report_for_{safe_query}_{timestamp}.txt")
                    with open(filepath, "w", encoding='utf-8') as f_write_rep:
                        f_write_rep.write(rep_text)

                    st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                    st.session_state.current_analysis_artifacts["report_query"] = query_led_to_data

                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"📊 **Report by {st.session_state.selected_worker_model} for '{html.escape(query_led_to_data)}':**\n\n{rep_text}",
                                                      "original_user_query": query_led_to_data,
                                                      "executed_result": {"type": "report_generated",
                                                                          "report_path": filepath,
                                                                          "data_source_path": data_path_rep or "N/A",
                                                                          "plot_source_path": plot_path_rep or "N/A"}})
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"User requested report for: '{query_led_to_data}'"},
                        {"output": f"Generated Report: {rep_text[:100]}..."})
                    if rep_spinner_container: rep_spinner_container.empty()
                    st.rerun()
                except Exception as e_rep_gen:
                    err_msg_rep = f"Error during report generation: {html.escape(str(e_rep_gen))}"
                    st.error(err_msg_rep)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg_rep})
                    if rep_spinner_container: rep_spinner_container.empty()
                    st.rerun()
        for key in ["report_target_data_path", "report_target_plot_path", "report_target_query"]:
            if key in st.session_state: del st.session_state[key]

# --- Judging Logic ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False
    artifacts_judge = st.session_state.current_analysis_artifacts
    judge_llm_instance = get_llm_instance(st.session_state.selected_judge_model)
    orig_query_artifacts = artifacts_judge.get("original_user_query", "Unknown query (for judging)")

    if not judge_llm_instance or not artifacts_judge.get("generated_code"):
        st.error("Judge LLM unavailable or no generated code in artifacts for critique.")
    else:
        try:
            code_content = artifacts_judge.get("generated_code", "No Python code provided.")

            data_content_for_judge = "No specific data file/text output from worker AI."
            if artifacts_judge.get("executed_dataframe_result") is not None:
                data_content_for_judge = f"DataFrame result (first 5 rows for brevity):\n{artifacts_judge['executed_dataframe_result'].head().to_string()}"
            elif artifacts_judge.get("executed_data_path") and os.path.exists(artifacts_judge["executed_data_path"]):
                with open(artifacts_judge["executed_data_path"], 'r', encoding='utf-8') as f_data_judge:
                    data_content_for_judge = f_data_judge.read(500)  # Read first 500 chars for brevity
                    if len(data_content_for_judge) == 500: data_content_for_judge += "\n... (content truncated)"
            elif artifacts_judge.get("executed_text_output"):
                data_content_for_judge = f"Text output from worker AI: {artifacts_judge.get('executed_text_output')}"

            report_content_judge = "No textual report generated by worker AI."
            if artifacts_judge.get("generated_report_path") and os.path.exists(
                    artifacts_judge["generated_report_path"]):
                with open(artifacts_judge["generated_report_path"], 'r', encoding='utf-8') as f_report_judge:
                    report_content_judge = f_report_judge.read()

            plot_img_path_judge = artifacts_judge.get("plot_image_path", "N/A")
            plot_info_for_judge_prompt = "No plot generated or referenced."
            if plot_img_path_judge != "N/A":
                if os.path.exists(plot_img_path_judge):
                    plot_info_for_judge_prompt = f"Plot image available at: '{os.path.basename(plot_img_path_judge)}'."
                    # Check for plot_specific_data_df in artifacts
                    if artifacts_judge.get("plot_specific_data_df") is not None and not artifacts_judge.get(
                            "plot_specific_data_df").empty:
                        plot_info_for_judge_prompt += f" Associated plot_data_df (first 5 rows):\n{artifacts_judge['plot_specific_data_df'].head().to_string()}"
                    elif artifacts_judge.get("executed_data_path") and "plot_data_for" in os.path.basename(
                            artifacts_judge.get("executed_data_path", "")):
                        plot_info_for_judge_prompt += f" Associated plot data file: '{os.path.basename(artifacts_judge.get('executed_data_path'))}'."
                else:
                    plot_info_for_judge_prompt = f"Plot image expected at '{os.path.basename(plot_img_path_judge)}' but not found."

            with st.chat_message("assistant"):
                critique_spinner_container = st.empty()
                critique_spinner_container.markdown(
                    f"⚖️ **{st.session_state.selected_judge_model}** critiquing for: '{html.escape(orig_query_artifacts)}'...")
                with st.spinner("Generating critique..."):
                    data_sum_prompt = json.dumps(st.session_state.data_summary,
                                                 indent=2) if st.session_state.data_summary else "{}"
                    judge_inputs = {"python_code": code_content,
                                    "data_content_for_judge": data_content_for_judge,
                                    "report_text_content": report_content_judge,
                                    "original_user_query": orig_query_artifacts,
                                    "data_summary": data_sum_prompt,
                                    "plot_image_path": plot_img_path_judge,
                                    "plot_info_for_judge": plot_info_for_judge_prompt}
                    response = judge_llm_instance.invoke(judging_prompt_template.format_prompt(**judge_inputs))
                    critique_text = response.content if hasattr(response, 'content') else response.get('text',
                                                                                                       "Error generating critique.")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in orig_query_artifacts[:30])
                    critique_filepath = os.path.join(TEMP_DATA_STORAGE, f"critique_on_{safe_query}_{timestamp}.txt")
                    with open(critique_filepath, "w", encoding='utf-8') as f_critique: f_critique.write(critique_text)

                    st.session_state.current_analysis_artifacts["generated_critique_path"] = critique_filepath
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for '{html.escape(orig_query_artifacts)}' (Saved to `{os.path.abspath(critique_filepath)}`):**",
                                                      "critique_text": critique_text})
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"Critique requested for: '{orig_query_artifacts}'"},
                        {"output": f"Generated Critique: {critique_text[:100]}..."})
                    if critique_spinner_container: critique_spinner_container.empty()
                    st.rerun()
        except Exception as e_judge:
            err_msg_judge = f"Error during judging: {html.escape(str(e_judge))}"
            st.error(err_msg_judge)
            st.session_state.messages.append({"role": "assistant", "content": err_msg_judge})
            if 'critique_spinner_container' in locals() and critique_spinner_container: critique_spinner_container.empty()
            st.rerun()

# --- HTML Report Export Logic ---
if st.session_state.get("trigger_html_export", False):
    st.session_state.trigger_html_export = False
    artifacts_html = st.session_state.current_analysis_artifacts
    cdo_report_html = st.session_state.get("cdo_initial_report_text")
    data_summary_html_dict = st.session_state.get("data_summary")  # This is already a dict
    main_df_for_html = st.session_state.get("current_dataframe")

    if not artifacts_html or not cdo_report_html or not data_summary_html_dict or main_df_for_html is None:
        st.error(
            "Cannot generate HTML: Key info missing (artifacts, CDO report, data summary, or main DataFrame). Ensure CSV loaded, CDO workflow run, and analysis performed.")
    else:
        with st.chat_message("assistant"):
            html_spinner_container = st.empty()
            html_spinner_container.markdown(f"⏳ Generating Bento HTML report...")
            with st.spinner("Generating HTML report..."):
                html_file_path = generate_bento_html_report_python(artifacts_html, cdo_report_html,
                                                                   data_summary_html_dict, main_df_for_html)

                if html_file_path and os.path.exists(html_file_path):
                    st.session_state.current_analysis_artifacts["generated_html_report_path"] = html_file_path
                    with open(html_file_path, "rb") as fp_html:
                        st.download_button(
                            label="📥 Download Bento HTML Report", data=fp_html,
                            file_name=os.path.basename(html_file_path), mime="text/html",
                            key=f"download_html_{datetime.datetime.now().timestamp()}"
                        )
                    success_msg_html = f"Bento HTML report generated: **{os.path.basename(html_file_path)}**."
                    st.success(success_msg_html)
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"📄 {success_msg_html} (Full path: `{os.path.abspath(html_file_path)}`)"})
                else:
                    error_msg_html = "Failed to generate Bento HTML report. Check logs."
                    st.error(error_msg_html)
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Sorry, error generating Bento HTML. Details: {error_msg_html}"})
                if html_spinner_container: html_spinner_container.empty()
