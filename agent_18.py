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
LLM_API_KEY = os.environ.get("LLM_API_KEY", "LLM_API_KEY")  # Default API Key
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
            dtypes_str = "\n".join(
                [f"- {col}: {data_summary_json.get('dtypes', {}).get(col, 'Unknown')}" for col in cols])

            return {"text": f"""
*Placeholder CDO Initial Data Description ({self.model_name}):*

**1. Dataset Overview (Simulated df.info()):**
   - Rows: {num_rows}, Columns: {num_cols}
   - Column Data Types:
{dtypes_str}
   - Potential Memory Usage: (Placeholder value) MB

**2. Inferred Meaning of Variables (Example):**
   - `ORDERNUMBER`: Unique identifier for each order.
   - `QUANTITYORDERED`: Number of units for a product in an order.
   *(This is a generic interpretation; actual meanings depend on the dataset.)*

**3. Initial Data Quality Assessment (Example):**
   - **Missing Values:** (Placeholder - e.g., "Column 'ADDRESSLINE2' has 80% missing values.")
   - **Overall:** The dataset seems reasonably structured.
"""}

        elif "panel of expert department heads, including the CDO" in prompt_str_content:
            return {"text": """
*Placeholder Departmental Perspectives (after CDO's initial report, via {model_name}):*

**CEO:** Focus on revenue trends.
**CFO:** Assess regional profitability.
**CDO (Highlighting for VPs):** Consider missing values.
""".format(model_name=self.model_name)}

        elif "You are the Chief Data Officer (CDO) of the company." in prompt_str_content and "synthesize these diverse perspectives" in prompt_str_content:
            return {"text": """
*Placeholder Final Analysis Strategy (Synthesized by CDO, via {model_name}):*

1.  **Visualize Core Sales Trends:** Line plot of 'SALES' over 'ORDERDATE'.
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

analysis_result = "Analysis logic executed. If you expected a specific output, please check the generated script."
plot_data_df = None 

# --- AI Generated Code Start ---
# Placeholder: The AI would generate its specific analysis logic here.
# --- AI Generated Code End ---

if 'analysis_result' not in locals() or analysis_result == "Analysis logic executed. If you expected a specific output, please check the generated script.":
    if isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set by the AI's main logic. Displaying df.head() as a default."
        plot_data_df = df.head().copy() 
    else:
        analysis_result = "Script completed. No specific output variable 'analysis_result' was set, and no DataFrame was available."
"""
            if "average sales" in user_query_segment:
                return {"text": "analysis_result = df['sales'].mean()\nplot_data_df = None"}
            elif "plot" in user_query_segment or "visualize" in user_query_segment:
                placeholder_plot_filename = "placeholder_plot.png"
                placeholder_full_save_path = os.path.join(TEMP_DATA_STORAGE, placeholder_plot_filename).replace("\\",
                                                                                                                "/")
                generated_plot_code = f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

fig, ax = plt.subplots()
if not df.empty and len(df.columns) > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        ax.hist(df[numeric_cols[0]])
        plot_data_df = df[[numeric_cols[0]]].copy()
        plot_save_path = r'{placeholder_full_save_path}' 
        plt.savefig(plot_save_path)
        plt.close(fig)
        analysis_result = '{placeholder_plot_filename}' 
    else: 
        if not df.empty:
            try:
                counts = df.iloc[:, 0].value_counts().head(10) 
                counts.plot(kind='bar', ax=ax)
                plt.title(f'Value Counts for {{df.columns[0]}}')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_data_df = counts.reset_index()
                plot_data_df.columns = [df.columns[0], 'count']
                plot_save_path = r'{placeholder_full_save_path}'
                plt.savefig(plot_save_path)
                plt.close(fig)
                analysis_result = '{placeholder_plot_filename}'
            except Exception as e:
                ax.text(0.5, 0.5, 'Could not generate fallback plot.', ha='center', va='center')
                plot_data_df = pd.DataFrame()
                analysis_result = "Failed to generate fallback plot: " + str(e)
                plt.close(fig)
        else:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            plot_data_df = pd.DataFrame()
            analysis_result = "No data to plot"
            plt.close(fig)
else:
    ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
    plot_data_df = pd.DataFrame()
    analysis_result = "DataFrame is empty, cannot plot."
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
                "text": f"### Placeholder Report ({self.model_name})\nThis is a placeholder report based on the CDO's focused analysis strategy."}
        elif "Critique the following analysis artifacts" in prompt_str_content:
            return {"text": f"""
### Placeholder Critique ({self.model_name})
**Overall Assessment:** Placeholder.
**Python Code:** Placeholder.
**Data:** Placeholder.
**Report:** Placeholder.
**Suggestions for Worker AI:** Placeholder.
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
        <div class="bento-item"><h2>Data Snapshot</h2><p>Placeholder: CDO's initial data description will be here.</p></div>
        <div class="bento-item" style="grid-column: span 2; background-color: #4A235A; border-color: #C778DD;">
            <h2 style="border-bottom-color: #C778DD;">Key Data Quality Alert</h2>
            <p>Placeholder: Missing data points will be listed here.</p>
        </div>
        <div class="bento-item"><h2>Actionable Insights</h2><p>Placeholder: Generated report insights will be here.</p></div>
        <div class="bento-item"><h2>Critique Summary</h2><p>Placeholder: Critique text will be here.</p></div>
    </div>
</body>
</html>"""}
        else:
            return {
                "text": f"Placeholder response from {self.model_name} for unrecognized prompt: {prompt_str_content[:200]}..."}


def get_llm_instance(model_name: str):
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
                temperature = 0.7 if st.session_state.get("selected_judge_model", "") == model_name else 0.2
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=LLM_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True
                )
                st.session_state.llm_cache[model_name] = llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM ({model_name}): {e}")
                st.session_state.llm_cache[model_name] = PlaceholderLLM(model_name)
    return st.session_state.llm_cache[model_name]


@st.cache_data
def calculate_data_summary(df_input):
    if df_input is None or df_input.empty:
        return None
    df = df_input.copy()
    data_summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "descriptive_stats_sample": df.describe(include='all').to_json() if not df.empty else "N/A",
        "preview_head": df.head().to_dict(orient='records'),
        "preview_tail": df.tail().to_dict(orient='records'),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    data_summary["missing_values_percentage"] = (data_summary["missing_values_total"] / (
            data_summary["num_rows"] * data_summary["num_columns"])) * 100 if (data_summary["num_rows"] *
                                                                               data_summary[
                                                                                   "num_columns"]) > 0 else 0
    return data_summary


def load_csv_and_get_summary(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataframe = df
        st.session_state.data_source_name = uploaded_file.name
        st.session_state.current_analysis_artifacts = {}
        summary_for_state = calculate_data_summary(df.copy())
        if summary_for_state:
            summary_for_state["source_name"] = uploaded_file.name
        st.session_state.data_summary = summary_for_state
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
    if df_input is None or df_input.empty:
        return pd.DataFrame()
    df = df_input.copy()
    quality_data = []
    max_cols_to_display = 10
    for col in df.columns[:max_cols_to_display]:
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_values = df[col].nunique()
        range_common = ""
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
        else:
            if not df[col].dropna().empty:
                common_vals = df[col].mode().tolist()
                range_common = f"Top: {', '.join(map(str, common_vals[:3]))}"
                if len(common_vals) > 3:
                    range_common += "..."
            else:
                range_common = "N/A (all missing)"
        score = 10
        if missing_percent > 50:
            score -= 5
        elif missing_percent > 20:
            score -= 3
        elif missing_percent > 5:
            score -= 1
        if unique_values == 1 and len(df) > 1: score -= 2
        if unique_values == len(df) and not pd.api.types.is_numeric_dtype(df[col]): score -= 1
        quality_data.append({
            "Column Name": col, "Data Type": dtype, "Missing %": f"{missing_percent:.2f}%",
            "Unique Values": unique_values, "Range / Common Values": range_common,
            "Quality Score ( /10)": max(0, score)
        })
    return pd.DataFrame(quality_data)


def generate_data_quality_dashboard(df_input):
    if df_input is None or df_input.empty:
        st.warning("No data loaded or DataFrame is empty. Please upload a CSV file.")
        return
    df = df_input.copy()
    st.header("📊 Data Quality Dashboard")
    st.markdown("An overview of your dataset's quality and characteristics.")
    st.subheader("Key Dataset Metrics")
    num_rows, num_cols, missing_percentage, numeric_cols_count, duplicate_rows = get_overview_metrics(df.copy())
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Rows", f"{num_rows:,}")
    col2.metric("Total Columns", f"{num_cols:,}")
    if missing_percentage > 5:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%", delta_color="inverse",
                    help="Percentage of missing data cells in the entire dataset. Red if > 5%.")
    else:
        col3.metric("Missing Values", f"{missing_percentage:.2f}%",
                    help="Percentage of missing data cells in the entire dataset.")
    col4.metric("Numeric Columns", f"{numeric_cols_count:,}")
    col5.metric("Duplicate Rows", f"{duplicate_rows:,}", help="Number of fully duplicated rows.")
    st.markdown("---")
    st.subheader("Column-wise Quality Assessment")
    if len(df.columns) > 10:
        st.caption(
            f"Displaying first 10 columns out of {len(df.columns)}. Full assessment available via report (placeholder).")
    quality_df = get_column_quality_assessment(df.copy())
    if not quality_df.empty:
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
    st.subheader("Numeric Column Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found in the dataset.")
    else:
        selected_numeric_col = st.selectbox("Select Numeric Column for Distribution Analysis:", numeric_cols,
                                            key="dq_numeric_select")
        if selected_numeric_col:
            col_data = df[selected_numeric_col].dropna()
            if not col_data.empty:
                fig = px.histogram(col_data, x=selected_numeric_col, marginal="box",
                                   title=f"Distribution of {selected_numeric_col}", opacity=0.75,
                                   histnorm='probability density')
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
                value_counts = col_data.value_counts(normalize=True).mul(100).round(2)
                count_abs = col_data.value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"Distribution of {selected_categorical_col}",
                             labels={'x': selected_categorical_col, 'y': 'Percentage (%)'},
                             text=[f"{val:.1f}% ({count_abs[idx]})" for idx, val in value_counts.items()])
                fig.update_layout(xaxis_title=selected_categorical_col, yaxis_title="Percentage (%)")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Column '{selected_categorical_col}' contains only missing values.")
    st.markdown("---")
    st.subheader("Numeric Column Correlation Heatmap")
    numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr) < 2:
        st.info("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")
    else:
        corr_matrix = df[numeric_cols_for_corr].corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                                title="Correlation Heatmap of Numeric Columns")
        fig_heatmap.update_xaxes(side="bottom")
        fig_heatmap.update_layout(xaxis_tickangle=-45, yaxis_tickangle=0)
        st.plotly_chart(fig_heatmap, use_container_width=True)


class LocalCodeExecutionEngine:
    def execute_code(self, code_string, df_input):
        if df_input is None:
            return {"type": "error", "message": "No data loaded to execute code on."}
        exec_globals = globals().copy()
        exec_globals['plt'] = matplotlib.pyplot
        exec_globals['sns'] = seaborn
        exec_globals['pd'] = pd
        exec_globals['np'] = np
        exec_globals['os'] = os
        local_scope = {'df': df_input.copy(), 'pd': pd, 'plt': matplotlib.pyplot, 'sns': seaborn, 'np': np, 'os': os}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_df_saved_path = None
        default_analysis_result_message = "Code executed, but 'analysis_result' was not explicitly set by the script."
        local_scope['analysis_result'] = default_analysis_result_message
        local_scope['plot_data_df'] = None
        try:
            exec(code_string, exec_globals, local_scope)
            analysis_result = local_scope.get('analysis_result')
            plot_data_df = local_scope.get('plot_data_df')
            if analysis_result == default_analysis_result_message:
                st.warning("The executed script did not explicitly set 'analysis_result'.")
            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                return {"type": "error", "message": analysis_result}
            if isinstance(analysis_result, str) and any(
                    analysis_result.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                plot_filename = os.path.basename(analysis_result)
                final_plot_path = os.path.join(TEMP_DATA_STORAGE, plot_filename)
                if not os.path.exists(final_plot_path):
                    if os.path.exists(analysis_result):
                        final_plot_path = analysis_result
                    else:
                        return {"type": "error",
                                "message": f"Plot file '{plot_filename}' not found. Expected at '{os.path.join(TEMP_DATA_STORAGE, plot_filename)}' or as a direct valid path. `analysis_result` was: '{analysis_result}'. Ensure code saves plot to designated temp directory."}
                if isinstance(plot_data_df, pd.DataFrame) and not plot_data_df.empty:
                    plot_data_filename = f"plot_data_for_{os.path.splitext(plot_filename)[0]}_{timestamp}.csv"
                    plot_data_df_saved_path = os.path.join(TEMP_DATA_STORAGE, plot_data_filename)
                    plot_data_df.to_csv(plot_data_df_saved_path, index=False)
                    st.info(f"Plot-specific data saved to: {plot_data_df_saved_path}")
                elif plot_data_df is not None:
                    st.warning("`plot_data_df` was set but is not a valid DataFrame. Not saving associated data.")
                return {"type": "plot", "plot_path": final_plot_path, "data_path": plot_data_df_saved_path}
            elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                analysis_result = analysis_result.to_frame() if isinstance(analysis_result,
                                                                           pd.Series) else analysis_result
                if analysis_result.empty: return {"type": "text", "value": "The analysis resulted in an empty table."}
                saved_csv_path = os.path.join(TEMP_DATA_STORAGE, f"table_result_{timestamp}.csv")
                analysis_result.to_csv(saved_csv_path, index=False)
                return {"type": "table", "data_path": saved_csv_path}
            else:
                return {"type": "text", "value": str(analysis_result)}
        except Exception as e:
            import traceback
            error_message_for_user = f"Error during code execution: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            current_analysis_res = local_scope.get('analysis_result', default_analysis_result_message)
            if current_analysis_res is None or (
                    isinstance(current_analysis_res, pd.DataFrame) and current_analysis_res.empty):
                local_scope['analysis_result'] = f"Execution Error: {str(e)}"
            return {"type": "error", "message": error_message_for_user,
                    "final_analysis_result_value": local_scope['analysis_result']}


code_executor = LocalCodeExecutionEngine()


# --- PDF Export Function ---
def export_analysis_to_pdf(artifacts, output_filename="analysis_report.pdf"):
    pdf_path = os.path.join(TEMP_DATA_STORAGE, output_filename)
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Comprehensive Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("1. Analysis Goal (User Query)", styles['h2']))
    analysis_goal = artifacts.get("original_user_query", "Not specified.")
    story.append(Paragraph(analysis_goal, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("2. CDO's Initial Data Description & Quality Assessment", styles['h2']))
    cdo_report_text = st.session_state.get("cdo_initial_report_text", "CDO initial report not available.")
    cdo_report_text_cleaned = cdo_report_text.replace("**", "")
    for para_text in cdo_report_text_cleaned.split('\n'):
        if para_text.strip().startswith("- "):
            story.append(Paragraph(para_text, styles['Bullet'], bulletText='-'))
        elif para_text.strip():
            story.append(Paragraph(para_text, styles['Normal']))
        else:
            story.append(Spacer(1, 0.1 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())
    story.append(Paragraph("3. Generated Plot", styles['h2']))
    plot_image_path = artifacts.get("plot_image_path")
    if plot_image_path and os.path.exists(plot_image_path):
        try:
            img = Image(plot_image_path, width=6 * inch, height=4 * inch);
            img.hAlign = 'CENTER'
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error embedding plot: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Plot image not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("4. Plot Data (or Executed Data Table)", styles['h2']))
    plot_data_csv_path = artifacts.get("executed_data_path")
    if plot_data_csv_path and os.path.exists(plot_data_csv_path) and plot_data_csv_path.endswith(".csv"):
        try:
            df_plot = pd.read_csv(plot_data_csv_path)
            data_for_table = [df_plot.columns.to_list()] + df_plot.values.tolist()
            if len(data_for_table) > 1:
                max_rows_in_pdf = 30
                if len(data_for_table) > max_rows_in_pdf:
                    data_for_table = data_for_table[:max_rows_in_pdf]
                    story.append(Paragraph(f"(Showing first {max_rows_in_pdf - 1} data rows)", styles['Italic']))
                table = Table(data_for_table, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
                story.append(table)
            else:
                story.append(Paragraph("CSV file is empty or contains only headers.", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading/displaying CSV: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Plot data CSV not available or path incorrect.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch));
    story.append(PageBreak())
    story.append(Paragraph("5. Generated Textual Report (Specific Analysis)", styles['h2']))
    report_text_path = artifacts.get("generated_report_path")
    if report_text_path and os.path.exists(report_text_path):
        try:
            with open(report_text_path, 'r', encoding='utf-8') as f:
                report_text_content = f.read()
            report_text_content_cleaned = report_text_content.replace("**", "")
            for para_text in report_text_content_cleaned.split('\n'): story.append(
                Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading report file: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Generated report text file not available.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("6. Analysis Critique", styles['h2']))
    critique_text_path = artifacts.get("generated_critique_path")
    if critique_text_path and os.path.exists(critique_text_path):
        try:
            with open(critique_text_path, 'r', encoding='utf-8') as f:
                critique_text_content = f.read()
            critique_text_content_cleaned = critique_text_content.replace("**", "")
            for para_text in critique_text_content_cleaned.split('\n'): story.append(
                Paragraph(para_text if para_text.strip() else "&nbsp;", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error reading critique file: {e}", styles['Normal']))
    else:
        story.append(Paragraph("Critique text file not available.", styles['Normal']))
    try:
        doc.build(story); return pdf_path
    except Exception as e:
        st.error(f"Failed to build PDF: {e}"); return None


# --- HTML Bento Report Generation ---
USER_PROVIDED_HTML_PROMPT_TEMPLATE = """
## Prompt for Generating a Bento-Style HTML Report

**Objective:** Generate a single, complete, and runnable HTML file that displays a multi-section report in a responsive "bento grid" layout. The design should follow a modern dark theme with specified accent colors.

**Key Requirements:**

1.  **Output Format:**
    * A single HTML file (`.html`).
    * All CSS should be included within `<style>` tags in the `<head>` or use Tailwind CSS classes directly if possible (ensure Tailwind CSS is loaded via CDN if used).
    * All JavaScript (if any, for charts or minor animations) should be included within `<script>` tags, preferably at the end of the `<body>`.

2.  **Layout Style - Bento Grid:**
    * The main content area should be a responsive grid (e.g., using CSS Grid).
    * The grid should display multiple "bento items" or "boxes."
    * Grid items should adapt to screen size, typically `repeat(auto-fit, minmax(280px, 1fr))`.
    * Allow for some items to be larger, spanning multiple columns (e.g., a class like `.large` that makes an item `grid-column: span 2;`).
    * Ensure appropriate `gap` between grid items (e.g., `1.5rem`).
    * The overall grid should have a `max-width` (e.g., `1600px`) and be centered on the page, but utilize a good portion of the screen width (e.g., `width: 100%; max-width: 95%` within certain viewport ranges).

3.  **Styling - Dark Theme & Accents:**
    * **Overall Theme:** Dark.
    * **Body Background:** Very dark blue/purple (e.g., `#1A1B26`).
    * **Default Text Color:** Light gray/white (e.g., `#E0E0E0`).
    * **Bento Item Background:** Darker blue/purple, slightly lighter than the body (e.g., `#2A2D3E`).
    * **Bento Item Borders:** Subtle dark border (e.g., `#3A3D4E`).
    * **Bento Item Hover Effect:** Change border color to a bright accent on hover (e.g., `#7DF9FF` - Light Blue/Cyan). Consider a slight `transform: translateY(-5px);`.
    * **Rounded Corners:** Apply to bento items (e.g., `border-radius: 16px;`).
    * **Titles (within bento items):**
        * Color: White (e.g., `#FFFFFF`).
        * Font size: Slightly larger (e.g., `1.5rem`).
        * Underline: Use a bright accent color (e.g., `border-bottom: 2px solid #7DF9FF;`).
    * **Paragraph Text (within bento items):** Slightly dimmer white/gray (e.g., `#C0C0C0`).
    * **Strong/Emphasized Text:** Use a bright accent color (e.g., `#7DF9FF`).
    * **Special Accent Box (e.g., for critical alerts):**
        * Background: Dark Magenta/Purple (e.g., `#4A235A`).
        * Border/Title Underline: Brighter Magenta (e.g., `#C778DD`).
        * Text: Light gray/white for readability.
    * **Font:** Use a clean, modern sans-serif font like 'Inter' (loaded via Google Fonts).

4.  **Content Structure (Example Sections - adapt as needed):**
    * Each bento item should represent a distinct section of the report.
    * Include placeholder content if specific data isn't provided, but structure it realistically.
    * **Example Sections to include (maintain this sequence):**
        1.  **Analysis Goal:** (Standard size bento box)
        2.  **Data Snapshot:** (Standard size bento box)
        3.  **Key Data Quality Alert:** (Large, special accent-bg bento box) - list of missing data points.
        4.  **Data Preprocessing Note:** (Standard size bento box)
        5.  **Geographic Sales Performance (Visualization):** (Large bento box) - This box should contain a placeholder for a chart or an actual chart if data is provided.
        6.  **Top Sales Performers (Table):** (Large bento box) - Display data in a styled table.
        7.  **Actionable Insights:** (Standard size bento box) - Bullet points.
        8.  **Critique Summary:** (Standard size bento box)
        9.  **Suggestions for Enhanced Analysis:** (Large bento box) - Bullet points.

5.  **Visualization (Chart - Optional but Preferred):**
    * If a "Visualization" bento box is included, embed a bar chart using a library like Chart.js (loaded via CDN).
    * **Chart Data (Example for "Geographic Sales Performance"):**
        * Use a sample dataset of countries and sales figures (at least 10-15 countries).
        * Labels: Country names.
        * Data: Sales values.
    * **Chart Styling (to match the dark theme):**
        * Bar Color: Magenta (e.g., `#C778DD`).
        * Bar Border Color: Darker Magenta (e.g., `#4A235A`).
        * Axis Titles/Labels/Legend Text: Light gray/white (e.g., `#E0E0E0`, `#A0A0A0`).
        * Grid Lines: Subtle and dark (e.g., `#3A3D4E`).
        * Tooltips: Styled for dark theme (e.g., background `#2A2D3E`, text `#FFFFFF`, border `#7DF9FF`).
    * The chart should be responsive and maintain aspect ratio appropriately within its container. The container should have a defined height (e.g. `380px`).

6.  **Table Styling (for "Top Sales Performers"):**
    * Header Background: Slightly lighter than bento item background (e.g., `#3A3D4E`).
    * Header Text: White (`#FFFFFF`).
    * Row Borders: Darker border (e.g., `#3A3D4E`).
    * Cell Text: Light gray (`#C0C0C0`).
    * Numeric Data (e.g., sales figures): Align right and use an accent color (e.g., `#7DF9FF`).

7.  **Responsiveness:**
    * The entire layout, including the bento grid and any charts/tables, must be fully responsive and look good on all screen sizes (mobile, tablet, desktop).
    * Use appropriate media queries to adjust layout, font sizes, and spacing.

8.  **Code Quality:**
    * Produce clean, well-commented HTML and CSS.
    * Ensure semantic HTML where appropriate.
    * No placeholder comments like `...` in the code.

**Provide the complete HTML code as a single block.**
"""


def get_content_from_path_helper(file_path, default_message="Not available."):
    """Safely reads content from a file path."""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    return default_message


def compile_report_text_for_html_generation(artifacts, cdo_initial_report, data_summary):
    """Compiles all necessary text pieces for the HTML generation prompt."""

    critique_text = get_content_from_path_helper(artifacts.get("generated_critique_path"), "Critique not available.")
    generated_report_text = get_content_from_path_helper(artifacts.get("generated_report_path"),
                                                         "Generated textual report not available.")

    # Determine plot_data_content: if a plot was generated and has associated data CSV
    plot_data_content = "No specific data table for chart provided."
    if artifacts.get("plot_image_path") and artifacts.get("executed_data_path"):
        # Check if the executed_data_path seems to be for the plot
        if "plot_data_for" in os.path.basename(artifacts.get("executed_data_path", "")):
            plot_data_content = get_content_from_path_helper(artifacts.get("executed_data_path"))
        # If not plot-specific, it might be a general table that was also plotted
        elif "table_result" in os.path.basename(artifacts.get("executed_data_path", "")):
            plot_data_content = get_content_from_path_helper(artifacts.get("executed_data_path"))

    # Determine table_data_content: if a table was generated (and not the same as plot data)
    table_data_content = "Table data not available or was part of plot data."
    if artifacts.get("executed_data_path") and "table_result" in os.path.basename(
            artifacts.get("executed_data_path", "")):
        # If plot_data_content is already using this table, don't repeat, or indicate it's the same source
        if plot_data_content != get_content_from_path_helper(artifacts.get("executed_data_path")):
            table_data_content = get_content_from_path_helper(artifacts.get("executed_data_path"))
        # else: it's already captured by plot_data_content if it was a table that was also plotted.

    report_parts = {
        "ANALYSIS_GOAL": artifacts.get("original_user_query", "Not specified."),
        "DATA_SNAPSHOT_CDO_REPORT": cdo_initial_report if cdo_initial_report else "CDO initial report not available.",
        "KEY_DATA_QUALITY_ALERT": "Missing data points:\n",
        "DATA_PREPROCESSING_NOTE": "Data used as provided by the user. Any specific preprocessing steps were part of the direct analysis query if requested, or detailed in the CDO report if applicable.",
        "GEOGRAPHIC_SALES_PERFORMANCE_DATA_FOR_CHART": f"Plot data (CSV format for Chart.js):\n{plot_data_content}",
        "TOP_SALES_PERFORMERS_TABLE_DATA": f"Table data (CSV format for HTML table):\n{table_data_content}",
        "ACTIONABLE_INSIGHTS": generated_report_text,
        "CRITIQUE_SUMMARY": critique_text,
        "SUGGESTIONS_FOR_ENHANCED_ANALYSIS": critique_text  # Assuming suggestions are in critique
    }

    missing_data_alerts = []
    if data_summary and data_summary.get("missing_values_per_column"):
        for col, count in data_summary["missing_values_per_column"].items():
            if count > 0:
                num_rows = data_summary.get("num_rows", 1)
                percentage = (count / num_rows) * 100 if num_rows > 0 else 0
                missing_data_alerts.append(f"- Column '{col}': {count} missing values ({percentage:.2f}%).")
    if not missing_data_alerts:
        missing_data_alerts.append(
            "No significant missing data points identified in the summary, or data quality check was primarily qualitative in the CDO report.")
    report_parts["KEY_DATA_QUALITY_ALERT"] += "\n".join(missing_data_alerts)

    full_text_for_html_prompt = "--- DATA FOR BENTO BOX REPORT ---\n"
    for key, value in report_parts.items():
        full_text_for_html_prompt += f"\n## {key.replace('_', ' ').title()} ##\n{value}\n"
    full_text_for_html_prompt += "\n--- END OF DATA FOR BENTO BOX REPORT ---"
    return full_text_for_html_prompt


def export_report_to_html_via_gemini(artifacts, cdo_initial_report, data_summary):
    """Generates an HTML report by sending text content to Gemini."""
    worker_llm = get_llm_instance(st.session_state.selected_worker_model)
    if not worker_llm:
        st.error("Worker LLM not available for HTML report generation.")
        return None

    compiled_text = compile_report_text_for_html_generation(artifacts, cdo_initial_report, data_summary)

    final_html_prompt = f"{USER_PROVIDED_HTML_PROMPT_TEMPLATE}\n\n{compiled_text}\n\nPlease generate the complete HTML file based on the instructions above, populating the bento box sections with the provided data. Ensure the Chart.js data is correctly formatted if data is available."

    try:
        response_obj = worker_llm.invoke(final_html_prompt)
        html_content = response_obj.content if hasattr(response_obj, 'content') else response_obj.get('text', "")

        # Clean up markdown code block delimiters if present around the HTML
        if html_content.strip().startswith("```html"):
            html_content = html_content.strip()[len("```html"):]
        if html_content.strip().endswith("```"):
            html_content = html_content.strip()[:-len("```")]
        html_content = html_content.strip()

        if not html_content.lower().startswith("<!doctype html>") and not html_content.lower().startswith("<html>"):
            st.warning("The LLM did not return a full HTML document. The output might be partial or incorrect.")
            # Fallback: Wrap in basic HTML structure if it seems like a fragment
            if not ("<html" in html_content and "<body" in html_content):  # very basic check
                html_content = f"<!DOCTYPE html><html><head><title>Report</title></head><body>{html_content}</body></html>"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query_part = "".join(
            c if c.isalnum() else "_" for c in artifacts.get("original_user_query", "report")[:30])
        html_filename = f"bento_report_{safe_query_part}_{timestamp}.html"
        html_filepath = os.path.join(TEMP_DATA_STORAGE, html_filename)

        with open(html_filepath, "w", encoding='utf-8') as f:
            f.write(html_content)
        return html_filepath
    except Exception as e:
        st.error(f"Error generating HTML report via Gemini: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="AI CSV Analyst v3.1 (CDO Workflow + DQ Dashboard)", layout="wide")
st.title("🤖 AI CSV Analyst v3.1")
st.caption(
    "Upload CSV, review Data Quality Dashboard, explore data, then optionally run CDO Workflow for AI-driven analysis.")

if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant",
                                                                     "content": "Hello! Select models, upload CSV to view Data Quality Dashboard and start analysis."}]
if "current_dataframe" not in st.session_state: st.session_state.current_dataframe = None
if "data_summary" not in st.session_state: st.session_state.data_summary = None
if "data_source_name" not in st.session_state: st.session_state.data_source_name = None
if "current_analysis_artifacts" not in st.session_state: st.session_state.current_analysis_artifacts = {}
if "selected_worker_model" not in st.session_state: st.session_state.selected_worker_model = DEFAULT_WORKER_MODEL
if "selected_judge_model" not in st.session_state: st.session_state.selected_judge_model = DEFAULT_JUDGE_MODEL
if "lc_memory" not in st.session_state: st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history",
                                                                                              return_messages=False,
                                                                                              input_key="user_query")
if "cdo_initial_report_text" not in st.session_state: st.session_state.cdo_initial_report_text = None
if "other_perspectives_text" not in st.session_state: st.session_state.other_perspectives_text = None
if "strategy_text" not in st.session_state: st.session_state.strategy_text = None
if "cdo_workflow_stage" not in st.session_state: st.session_state.cdo_workflow_stage = None

# --- Prompt Templates (excluding the large HTML one already defined) ---
cdo_initial_data_description_prompt_template = PromptTemplate(input_variables=["data_summary", "chat_history"],
                                                              template="""You are the Chief Data Officer (CDO). A user has uploaded a CSV file.
Data Summary (for context):
{data_summary}
CDO, your first task is to provide an initial description of the dataset. This should include:
1.  A brief overview similar to `df.info()` (column names, non-null counts, dtypes).
2.  Your inferred meaning or common interpretation for each variable/column.
3.  A preliminary assessment of data quality (e.g., obvious missing data patterns, potential outliers you notice from the summary, data type consistency).
This description will be shared with the other department heads (CEO, CFO, CTO, COO, CMO) before they provide their perspectives.
Conversation History (for context, if any):
{chat_history}
Detailed Initial Description by CDO:""")
individual_perspectives_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report"], template="""You are a panel of expert department heads, including the CDO.
A user has uploaded a CSV file, and the Chief Data Officer (CDO) has provided an initial data description and quality assessment.
Data Summary (Original):
{data_summary}
CDO's Initial Data Description & Quality Report:
--- BEGIN CDO REPORT ---
{cdo_initial_report}
--- END CDO REPORT ---
Based on BOTH the original data summary AND the CDO's initial report, provide a detailed perspective from each of the following roles (CEO, CFO, CTO, COO, CMO).
For each role, outline 2-3 specific questions they would now ask, analyses they would want to perform, or observations they would make, considering the CDO's findings.
The CDO should also provide a brief perspective here, perhaps by reiterating 1-2 critical data quality points from their initial report that the other VPs *must* consider, or by highlighting specific data features that are now more apparent.
Structure your response clearly, with each role's perspective under a bolded heading (e.g., **CEO Perspective:**).
* **CEO (首席執行官 - Chief Executive Officer):**
* **CFO (首席財務官 - Chief Financial Officer):**
* **CTO (首席技術官 - Chief Technology Officer):**
* **COO (首席運營官 - Chief Operating Officer):**
* **CMO (首席行銷官 - Chief Marketing Officer):**
* **CDO (首席數據官 - Reiterating Key Points):**
Conversation History (for context, if any):
{chat_history}
Detailed Perspectives from Department Heads (informed by CDO's initial report):""")
synthesize_analysis_suggestions_prompt_template = PromptTemplate(
    input_variables=["data_summary", "chat_history", "cdo_initial_report", "generated_perspectives_from_others"],
    template="""You are the Chief Data Officer (CDO) of the company.
A user has uploaded a CSV file. You have already performed an initial data description and quality assessment.
Subsequently, the other department heads (CEO, CFO, CTO, COO, CMO) have provided their perspectives based on your initial findings and the data summary.
Original Data Summary:
{data_summary}
Your Initial Data Description & Quality Report:
--- BEGIN YOUR INITIAL CDO REPORT ---
{cdo_initial_report}
--- END YOUR INITIAL CDO REPORT ---
Perspectives from other Department Heads (CEO, CFO, CTO, COO, CMO):
--- BEGIN OTHER PERSPECTIVES ---
{generated_perspectives_from_others}
--- END OTHER PERSPECTIVES ---
Your task is to synthesize all this information (your initial findings AND the other VPs' inputs) into a concise list of **5 distinct and actionable analysis strategy suggestions** for the user.
These suggestions must prioritize analyses that result in clear visualizations (e.g., charts, plots), well-structured tables, or concise descriptive summaries.
This approach is preferred because it makes the analysis results easier to execute locally and interpret broadly.
Present these 5 suggestions as a numbered list. Each suggestion should clearly state the type of analysis (e.g., "Visualize X...", "Create a table for Y...", "Describe Z...").
Conversation History (for context, if any):
{chat_history}
Final 5 Analysis Strategy Suggestions (Synthesized by the CDO, focusing on visualizations, tables, and descriptive methods, incorporating all prior inputs):""")
code_generation_prompt_template = PromptTemplate(input_variables=["data_summary", "user_query", "chat_history"],
                                                 template="""You are an expert Python data analysis assistant.
Data Summary:
{data_summary}
User Query: "{user_query}"
Previous Conversation (for context):
{chat_history}
Your task is to generate a Python script to perform the requested analysis on a pandas DataFrame named `df`.
**Crucial Instructions for `analysis_result` and `plot_data_df`:**
1.  **`analysis_result` MUST BE SET**: The primary result MUST be assigned to `analysis_result`.
2.  **`plot_data_df` for Plots**: If creating a plot:
    a.  Save plot to '{TEMP_DATA_STORAGE}' (e.g., `os.path.join('{TEMP_DATA_STORAGE}', 'my_plot.png')`).
    b.  Set `analysis_result` to *only the filename string* (e.g., 'my_plot.png').
    c.  Create `plot_data_df` with ONLY data visualized. If full `df`, then `plot_data_df = df.copy()`. If none, `None`.
3.  **`plot_data_df` for Non-Plots**: If not a plot, `plot_data_df` is `None`. If `analysis_result` is a DataFrame for reporting, `plot_data_df = analysis_result.copy()`.
4.  **Default `analysis_result`**: For general queries, assign descriptive string or `df.head()` to `analysis_result`.
5.  **Imports**: Import all libraries (`matplotlib.pyplot as plt`, `seaborn as sns`, `pandas as pd`, `numpy as np`, `os`).
**Safety Net - Fallback within your generated script:**
```python
analysis_result = "Script started, but 'analysis_result' was not yet set by main logic."
plot_data_df = None
# --- Your main analysis code here ---
if analysis_result == "Script started, but 'analysis_result' was not yet set by main logic.":
    if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        analysis_result = "Analysis performed. No specific output. Displaying df.head()."
    else:
        analysis_result = "Analysis performed, but 'analysis_result' not set and no DataFrame."
```
Output only raw Python code.
Python code:""".replace("{TEMP_DATA_STORAGE}", TEMP_DATA_STORAGE.replace("\\", "/")))
report_generation_prompt_template = PromptTemplate(
    input_variables=["table_data_csv", "original_data_summary", "user_query_that_led_to_data", "chat_history"],
    template="""You are an insightful data analyst. Report based on data and context.
Original Data Summary: {original_data_summary}
User Query for this data: "{user_query_that_led_to_data}"
Chat History: {chat_history}
Analysis Result Data (CSV):
```csv
{table_data_csv}
```
**Report Structure:**
* 1. Executive Summary (1-2 sentences): Main conclusion.
* 2. Purpose (1 sentence): User's goal.
* 3. Key Observations (Bulleted list, 2-4 points): Quantified.
* 4. Actionable Insights (1-2 insights): Meaning in context.
* 5. Data Focus & Limitations: Report *solely* on "Analysis Result Data".
**Tone:** Professional, clear. Do NOT say "the CSV".
Report:""")
judging_prompt_template = PromptTemplate(
    input_variables=["python_code", "data_csv_content", "report_text_content", "original_user_query", "data_summary",
                     "plot_image_path", "plot_info"], template="""Expert data science reviewer. Evaluate AI assistant's artifacts.
Original User Query: "{original_user_query}"
Original Data Summary: {data_summary}
--- ARTIFACTS ---
1. Python Code: ```python\n{python_code}\n```
2. Data Produced (CSV or text): ```csv\n{data_csv_content}\n```\n{plot_info}
3. Report: ```text\n{report_text_content}\n```
--- END ARTIFACTS ---
Critique:
1. Code Quality: Correctness, efficiency, readability, best practices, bugs? Correct use of `analysis_result` (filename for plots) and `plot_data_df`? Code saves plot to correct temp subfolder?
2. Data Analysis: Relevance to query/data? Accurate? Appropriate methods? `plot_data_df` content match plot?
3. Plot Quality (if `{plot_image_path}` exists): Appropriate type? Well-labeled? Clear?
4. Report Quality: Clear, concise, insightful? Reflects `data_csv_content`? Addresses query?
5. Overall Effectiveness: How well query addressed (1-10)? Actionable suggestions for worker AI.
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
    st.header("📤 Upload CSV")
    uploaded_file = st.file_uploader("Select your CSV file:", type="csv", key="csv_uploader")
    if uploaded_file is not None:
        if st.session_state.get("data_source_name") != uploaded_file.name:
            with st.spinner("Processing CSV..."):
                if load_csv_and_get_summary(uploaded_file):
                    st.success(f"CSV '{st.session_state.data_source_name}' processed.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Processed '{st.session_state.data_source_name}'. View Data Quality Dashboard or other tabs."})
                    st.rerun()
                else:
                    st.error("Failed to process CSV.")
    if st.session_state.current_dataframe is not None:
        st.subheader("File Loaded:")
        st.write(
            f"**{st.session_state.data_source_name}** ({len(st.session_state.current_dataframe)} rows x {len(st.session_state.current_dataframe.columns)} columns)")
        if st.button("Clear Loaded Data & Chat", key="clear_data_btn"):
            keys_to_reset = ["current_dataframe", "data_summary", "data_source_name", "current_analysis_artifacts",
                             "messages", "lc_memory", "cdo_initial_report_text", "other_perspectives_text",
                             "strategy_text", "cdo_workflow_stage", "trigger_report_generation",
                             "report_target_data_path", "report_target_plot_path", "report_target_query",
                             "trigger_judging", "trigger_html_export"]
            for key in keys_to_reset:
                if key in st.session_state: del st.session_state[key]
            st.session_state.messages = [{"role": "assistant", "content": "Data and chat reset. Upload a new CSV."}]
            st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False,
                                                                  input_key="user_query")
            st.session_state.current_analysis_artifacts = {}
            cleaned_files_count = 0
            if os.path.exists(TEMP_DATA_STORAGE):
                for item in os.listdir(TEMP_DATA_STORAGE):
                    item_path = os.path.join(TEMP_DATA_STORAGE, item)
                    if os.path.isfile(item_path):
                        try:
                            os.remove(item_path); cleaned_files_count += 1
                        except Exception as e:
                            st.warning(f"Could not remove {item_path}: {e}")
            st.success(f"Data, chat, and {cleaned_files_count} temp files cleared.")
            st.rerun()
    st.markdown("---")
    st.info(
        f"Worker: **{st.session_state.selected_worker_model}**\n\nJudge: **{st.session_state.selected_judge_model}**")
    st.info(f"Temp files in: `{os.path.abspath(TEMP_DATA_STORAGE)}`")
    st.warning("⚠️ **Security Note:** Uses `exec()` for AI-generated code. Demo purposes ONLY.")

# --- Main Area with Tabs ---
if st.session_state.current_dataframe is not None:
    tab_titles = ["📊 Data Quality Dashboard", "🔍 Data Explorer", "👨‍💼 CDO Workflow", "💬 AI Analysis Chat"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        generate_data_quality_dashboard(st.session_state.current_dataframe.copy())
    with tab2:
        st.header("🔍 Data Explorer")
        if st.session_state.data_summary:
            with st.expander("View Full Data Summary (JSON)"):
                st.json(st.session_state.data_summary)
        else:
            st.write("No data summary.")
        with st.expander(f"View DataFrame Head (First 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.head())
        with st.expander(f"View DataFrame Tail (Last 5 rows of {st.session_state.data_source_name})"):
            st.dataframe(st.session_state.current_dataframe.tail())
    with tab3:
        st.header("👨‍💼 CDO-led Analysis Workflow")
        st.markdown("Initiate an AI-driven analysis: CDO describes data, VPs discuss, CDO synthesizes strategy.")
        if st.button("🚀 Start CDO Analysis Workflow", key="start_cdo_workflow_btn"):
            st.session_state.cdo_workflow_stage = "initial_description"
            st.session_state.cdo_initial_report_text = None;
            st.session_state.other_perspectives_text = None;
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
                            'text', "Error.")
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": f"**CDO's Initial Description (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.cdo_initial_report_text}"})
                        st.session_state.lc_memory.save_context(
                            {"user_query": "System: CDO initial description requested."},
                            {"output": f"CDO Report: {st.session_state.cdo_initial_report_text[:100]}..."})
                        st.session_state.cdo_workflow_stage = "departmental_perspectives";
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in CDO description: {e}"); st.session_state.cdo_workflow_stage = None
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
                        'text', "Error.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**Departmental Perspectives (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.other_perspectives_text}"})
                    st.session_state.lc_memory.save_context({"user_query": "System: VPs' perspectives requested."}, {
                        "output": f"VPs' Perspectives: {st.session_state.other_perspectives_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "strategy_synthesis";
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in VPs' perspectives: {e}"); st.session_state.cdo_workflow_stage = None

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
                        'text', "Error.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"**CDO's Final Strategy (via {st.session_state.selected_worker_model}):**\n\n{st.session_state.strategy_text}\n\nGo to 'AI Analysis Chat' tab for specific analyses."})
                    st.session_state.lc_memory.save_context({"user_query": "System: CDO strategy synthesis requested."},
                                                            {
                                                                "output": f"CDO Strategy: {st.session_state.strategy_text[:100]}..."})
                    st.session_state.cdo_workflow_stage = "completed";
                    st.success("CDO Workflow Completed!");
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in CDO strategy synthesis: {e}"); st.session_state.cdo_workflow_stage = None

        if st.session_state.cdo_initial_report_text:
            with st.expander("CDO's Initial Data Description",
                             expanded=current_stage == "initial_description"): st.markdown(
                st.session_state.cdo_initial_report_text)
        if st.session_state.other_perspectives_text:
            with st.expander("Departmental Perspectives",
                             expanded=current_stage == "departmental_perspectives"): st.markdown(
                st.session_state.other_perspectives_text)
        if st.session_state.strategy_text:
            with st.expander("CDO's Final Analysis Strategy",
                             expanded=current_stage in ["strategy_synthesis", "completed"]): st.markdown(
                st.session_state.strategy_text)

    with tab4:
        st.header("💬 AI Analysis Chat")
        st.caption("Interact with Worker AI for analyses, reports, and critiques from Judge AI.")
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "executed_result" in message:
                        exec_res = message["executed_result"];
                        res_type = exec_res.get("type")
                        orig_query = message.get("original_user_query",
                                                 st.session_state.current_analysis_artifacts.get("original_user_query",
                                                                                                 "Unknown"))
                        if res_type == "table":
                            try:
                                df_disp = pd.read_csv(exec_res["data_path"]);
                                st.dataframe(df_disp)
                                if st.button(f"📊 Report for Table##{i}", key=f"rep_tbl_btn_{i}_tab4"):
                                    st.session_state.trigger_report_generation = True;
                                    st.session_state.report_target_data_path = exec_res["data_path"]
                                    st.session_state.report_target_plot_path = None;
                                    st.session_state.report_target_query = orig_query;
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error displaying table: {e}")
                        elif res_type == "plot":
                            if os.path.exists(exec_res["plot_path"]):
                                st.image(exec_res["plot_path"])
                                if exec_res.get("data_path") and os.path.exists(exec_res["data_path"]):
                                    if st.button(f"📄 Report for Plot Data##{i}", key=f"rep_plot_data_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True;
                                        st.session_state.report_target_data_path = exec_res["data_path"]
                                        st.session_state.report_target_plot_path = exec_res["plot_path"];
                                        st.session_state.report_target_query = orig_query;
                                        st.rerun()
                                else:
                                    st.caption("Plot-specific data not saved/found.")
                                    if st.button(f"📄 Descriptive Report for Plot##{i}",
                                                 key=f"rep_plot_desc_btn_{i}_tab4"):
                                        st.session_state.trigger_report_generation = True;
                                        st.session_state.report_target_data_path = None
                                        st.session_state.report_target_plot_path = exec_res["plot_path"];
                                        st.session_state.report_target_query = orig_query;
                                        st.rerun()
                            else:
                                st.warning(f"Plot image not found: {exec_res['plot_path']}")
                        elif res_type == "text":
                            st.markdown(f"**Output:**\n```\n{exec_res.get('value', 'No text output.')}\n```")
                        elif res_type == "report_generated":
                            if exec_res.get("report_path") and os.path.exists(exec_res["report_path"]): st.markdown(
                                f"_Report: `{os.path.abspath(exec_res['report_path'])}`_")

                        artifacts_judge = st.session_state.get("current_analysis_artifacts", {})
                        can_judge = artifacts_judge.get("generated_code") and (
                                    artifacts_judge.get("executed_data_path") or artifacts_judge.get(
                                "plot_image_path") or artifacts_judge.get("executed_text_output") or (
                                                res_type == "text" and exec_res.get("value")))
                        if can_judge:
                            if st.button(f"⚖️ Judge Analysis##{i}", key=f"judge_btn_{i}_tab4"):
                                st.session_state.trigger_judging = True;
                                st.rerun()

                    if message["role"] == "assistant" and "critique_text" in message:
                        with st.expander(f"View Critique by {st.session_state.selected_judge_model}", expanded=True):
                            st.markdown(message["critique_text"])
                        if st.button(f"📄 Export Full Analysis to PDF##{i}", key=f"pdf_exp_btn_{i}_tab4"):
                            with st.spinner("Generating PDF..."):
                                pdf_path = export_analysis_to_pdf(st.session_state.current_analysis_artifacts)
                                if pdf_path and os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as f:
                                        st.download_button("Download PDF", f, os.path.basename(pdf_path),
                                                           "application/pdf", key=f"dl_pdf_{i}_tab4")
                                    st.success(f"PDF: {os.path.basename(pdf_path)}")
                                else:
                                    st.error("Failed to generate PDF.")
                        # Add HTML export button here
                        if st.button(f"📄 Export Bento HTML Report##{i}", key=f"html_exp_btn_{i}_tab4"):
                            st.session_state.trigger_html_export = True
                            st.rerun()

        if user_query := st.chat_input("Ask for analysis (Worker Model will generate and run code)...",
                                       key="user_query_input_tab4"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            if st.session_state.current_dataframe is None or st.session_state.data_summary is None:
                st.warning("Please upload and process a CSV file first (via sidebar).")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I need CSV data. Please upload a file first."})
            else:
                worker_llm_chat = get_llm_instance(st.session_state.selected_worker_model)
                if not worker_llm_chat:
                    st.error(f"Worker model {st.session_state.selected_worker_model} not initialized.")
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"Worker LLM ({st.session_state.selected_worker_model}) unavailable."})
                else:
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    st.session_state.current_analysis_artifacts = {"original_user_query": user_query}
                    st.session_state.trigger_code_generation = True;
                    st.rerun()
else:
    st.info("👋 Welcome! Please upload a CSV file using the sidebar to get started.")
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]): st.markdown(message["content"])

# --- Code Generation Logic ---
if st.session_state.get("trigger_code_generation", False):
    st.session_state.trigger_code_generation = False
    user_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        gen_code_str = ""
        msg_placeholder.markdown(
            f"⏳ **{st.session_state.selected_worker_model}** generating code for: '{user_query}'...")
        with st.spinner(f"{st.session_state.selected_worker_model} generating Python code..."):
            try:
                worker_llm_code_gen = get_llm_instance(st.session_state.selected_worker_model)
                mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                data_sum_prompt = json.dumps(st.session_state.data_summary,
                                             indent=2) if st.session_state.data_summary else "{}"
                prompt_inputs = {"data_summary": data_sum_prompt, "user_query": user_query,
                                 "chat_history": mem_ctx.get("chat_history", "")}
                response = worker_llm_code_gen.invoke(code_generation_prompt_template.format_prompt(**prompt_inputs))
                gen_code_str = response.content if hasattr(response, 'content') else response.get('text', "")
                for prefix in ["```python\n", "```\n", "```"]:
                    if gen_code_str.startswith(prefix): gen_code_str = gen_code_str[len(prefix):]
                if gen_code_str.endswith("\n```"):
                    gen_code_str = gen_code_str[:-len("\n```")]
                elif gen_code_str.endswith("```"):
                    gen_code_str = gen_code_str[:-len("```")]
                gen_code_str = gen_code_str.strip()
                st.session_state.current_analysis_artifacts["generated_code"] = gen_code_str
                assist_base_content = f"🔍 **Generated Code by {st.session_state.selected_worker_model} for '{user_query}':**\n```python\n{gen_code_str}\n```\n"
                msg_placeholder.markdown(assist_base_content + "\n⏳ Executing code...")
            except Exception as e:
                err_msg = f"Error generating code: {e}";
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.session_state.lc_memory.save_context({"user_query": user_query}, {"output": f"Code Gen Error: {e}"});
                st.rerun()

        if gen_code_str:
            curr_assist_resp_msg = {"role": "assistant", "content": assist_base_content,
                                    "original_user_query": user_query}
            with st.spinner("Executing code..."):
                exec_result = code_executor.execute_code(gen_code_str, st.session_state.current_dataframe.copy())
                if exec_result.get("data_path"): st.session_state.current_analysis_artifacts["executed_data_path"] = \
                exec_result["data_path"]
                if exec_result.get("plot_path"): st.session_state.current_analysis_artifacts["plot_image_path"] = \
                exec_result["plot_path"]
                if exec_result.get("type") == "text" and exec_result.get("value"):
                    st.session_state.current_analysis_artifacts["executed_text_output"] = exec_result.get("value")

                llm_mem_output = ""
                if exec_result["type"] == "error":
                    curr_assist_resp_msg["content"] += f"\n⚠️ **Execution Error:**\n```\n{exec_result['message']}\n```"
                    if str(st.session_state.current_analysis_artifacts.get("executed_text_output", "")).startswith(
                        "Code executed, but"): st.session_state.current_analysis_artifacts[
                        "executed_text_output"] = f"Exec Error: {exec_result.get('final_analysis_result_value', 'Unknown')}"
                    llm_mem_output = f"Exec Error: {exec_result['message'][:100]}..."
                else:
                    curr_assist_resp_msg["content"] += "\n✅ **Code Executed Successfully!**"
                    curr_assist_resp_msg["executed_result"] = exec_result
                    if exec_result.get("type") == "text" and str(exec_result.get("value", "")).startswith(
                        "Code executed, but"):
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            exec_result.get("value", ""))
                    elif exec_result.get("type") == "text":
                        st.session_state.current_analysis_artifacts["executed_text_output"] = str(
                            exec_result.get("value", ""))
                    if exec_result.get("data_path"): curr_assist_resp_msg[
                        "content"] += f"\n💾 Data: `{os.path.abspath(exec_result['data_path'])}`"
                    if exec_result.get("plot_path"): curr_assist_resp_msg[
                        "content"] += f"\n🖼️ Plot: `{os.path.abspath(exec_result['plot_path'])}`"
                    if exec_result.get("data_path") and "plot_data_for" in os.path.basename(
                        exec_result.get("data_path", "")): curr_assist_resp_msg[
                        "content"] += " (Plot-specific data saved)."

                    if exec_result["type"] == "table":
                        llm_mem_output = f"Table: {os.path.basename(exec_result['data_path'])}"
                    elif exec_result["type"] == "plot":
                        llm_mem_output = f"Plot: {os.path.basename(exec_result['plot_path'])}"
                        if exec_result.get(
                            "data_path"): llm_mem_output += f" (Data: {os.path.basename(exec_result['data_path'])})"
                    elif exec_result["type"] == "text":
                        llm_mem_output = f"Text: {str(exec_result['value'])[:50]}..."
                    else:
                        llm_mem_output = "Code exec, unknown result type."

                st.session_state.lc_memory.save_context(
                    {"user_query": f"{user_query}\n---Code---\n{gen_code_str}\n---End Code---"},
                    {"output": llm_mem_output})
                st.session_state.messages.append(curr_assist_resp_msg)
                msg_placeholder.empty();
                st.rerun()

# --- Report Generation Logic ---
if st.session_state.get("trigger_report_generation", False):
    st.session_state.trigger_report_generation = False
    data_path_rep = st.session_state.get("report_target_data_path")
    plot_path_rep = st.session_state.get("report_target_plot_path")
    query_led_to_data = st.session_state.report_target_query
    worker_llm_rep = get_llm_instance(st.session_state.selected_worker_model)

    if not worker_llm_rep or not st.session_state.data_summary or (not data_path_rep and not plot_path_rep):
        st.error("Cannot generate report: LLM, data summary, or target data/plot missing.")
    else:
        csv_content_rep = "N/A - Report is likely descriptive of a plot image."
        if data_path_rep and os.path.exists(data_path_rep):
            try:
                with open(data_path_rep, 'r', encoding='utf-8') as f:
                    csv_content_rep = f.read()
            except Exception as e:
                st.error(f"Error reading data for report: {e}"); st.rerun()
        elif plot_path_rep:
            st.info("Generating descriptive report for plot image (no specific data table).")

        with st.chat_message("assistant"):
            rep_spinner_container = st.empty()
            rep_spinner_container.markdown(
                f"📝 **{st.session_state.selected_worker_model}** generating report for: '{query_led_to_data}'...")
            with st.spinner("Generating report..."):
                try:
                    mem_ctx = st.session_state.lc_memory.load_memory_variables({})
                    data_sum_prompt = json.dumps(st.session_state.data_summary,
                                                 indent=2) if st.session_state.data_summary else "{}"
                    prompt_inputs = {"table_data_csv": csv_content_rep, "original_data_summary": data_sum_prompt,
                                     "user_query_that_led_to_data": query_led_to_data,
                                     "chat_history": mem_ctx.get("chat_history", "")}
                    response = worker_llm_rep.invoke(report_generation_prompt_template.format_prompt(**prompt_inputs))
                    rep_text = response.content if hasattr(response, 'content') else response.get('text', "Error.")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in query_led_to_data[:30])
                    filepath = os.path.join(TEMP_DATA_STORAGE, f"report_for_{safe_query}_{timestamp}.txt")
                    with open(filepath, "w", encoding='utf-8') as f:
                        f.write(rep_text)
                    st.session_state.current_analysis_artifacts["generated_report_path"] = filepath
                    st.session_state.current_analysis_artifacts["report_query"] = query_led_to_data
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"📊 **Report by {st.session_state.selected_worker_model} for '{query_led_to_data}':**\n\n{rep_text}",
                                                      "original_user_query": query_led_to_data,
                                                      "executed_result": {"type": "report_generated",
                                                                          "report_path": filepath,
                                                                          "data_source_path": data_path_rep or "N/A",
                                                                          "plot_source_path": plot_path_rep or "N/A"}})
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"Requested report for: '{query_led_to_data}'"},
                        {"output": f"Report: {rep_text[:100]}..."})
                    rep_spinner_container.empty();
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error generating report: {e}"})
                    if 'rep_spinner_container' in locals(): rep_spinner_container.empty()
                    st.rerun()
        for key in ["report_target_data_path", "report_target_plot_path", "report_target_query"]:
            if key in st.session_state: del st.session_state[key]

# --- Judging Logic ---
if st.session_state.get("trigger_judging", False):
    st.session_state.trigger_judging = False
    artifacts_judge = st.session_state.current_analysis_artifacts
    judge_llm_instance = get_llm_instance(st.session_state.selected_judge_model)
    orig_query_artifacts = artifacts_judge.get("original_user_query", "Unknown query")

    if not judge_llm_instance or not artifacts_judge.get("generated_code"):
        st.error("Judge LLM not available or generated code missing for critique.")
    else:
        try:
            code_content = artifacts_judge.get("generated_code", "No code.")
            data_content = "No data file produced."
            if artifacts_judge.get("executed_data_path") and os.path.exists(artifacts_judge["executed_data_path"]):
                with open(artifacts_judge["executed_data_path"], 'r', encoding='utf-8') as f:
                    data_content = f.read()
            elif artifacts_judge.get("executed_text_output"):
                data_content = f"Text output: {artifacts_judge.get('executed_text_output')}"

            report_content = "No report generated."
            if artifacts_judge.get("generated_report_path") and os.path.exists(
                    artifacts_judge["generated_report_path"]):
                with open(artifacts_judge["generated_report_path"], 'r', encoding='utf-8') as f:
                    report_content = f.read()
            elif artifacts_judge.get("report_query") and not artifacts_judge.get("generated_report_path"):
                report_content = f"Report expected for '{artifacts_judge.get('report_query')}' but not found."

            plot_img_path = artifacts_judge.get("plot_image_path", "N/A")
            plot_info_judge = f"Plot Image: {plot_img_path}."
            if plot_img_path == "N/A":
                plot_info_judge = "No plot."
            elif not os.path.exists(plot_img_path):
                plot_info_judge = f"Plot at '{plot_img_path}' not found."
            else:
                plot_info_judge = f"Plot at '{plot_img_path}'. "
                if artifacts_judge.get("executed_data_path") and "plot_data_for" in os.path.basename(
                    artifacts_judge.get("executed_data_path", "")):
                    plot_info_judge += f"Plot data at '{artifacts_judge.get('executed_data_path')}'."
                else:
                    plot_info_judge += "Plot-specific data not found/applicable."

            with st.chat_message("assistant"):
                critique_spinner_container = st.empty()
                critique_spinner_container.markdown(
                    f"⚖️ **{st.session_state.selected_judge_model}** critiquing for: '{orig_query_artifacts}'...")
                with st.spinner("Generating critique..."):
                    data_sum_prompt = json.dumps(st.session_state.data_summary,
                                                 indent=2) if st.session_state.data_summary else "{}"
                    judge_inputs = {"python_code": code_content, "data_csv_content": data_content,
                                    "report_text_content": report_content, "original_user_query": orig_query_artifacts,
                                    "data_summary": data_sum_prompt, "plot_image_path": plot_img_path,
                                    "plot_info": plot_info_judge}
                    response = judge_llm_instance.invoke(judging_prompt_template.format_prompt(**judge_inputs))
                    critique_text = response.content if hasattr(response, 'content') else response.get('text', "Error.")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_query = "".join(c if c.isalnum() else "_" for c in orig_query_artifacts[:30])
                    critique_filepath = os.path.join(TEMP_DATA_STORAGE, f"critique_on_{safe_query}_{timestamp}.txt")
                    with open(critique_filepath, "w", encoding='utf-8') as f: f.write(critique_text)
                    st.session_state.current_analysis_artifacts["generated_critique_path"] = critique_filepath
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": f"⚖️ **Critique by {st.session_state.selected_judge_model} for '{orig_query_artifacts}' (saved to `{os.path.abspath(critique_filepath)}`):**",
                                                      "critique_text": critique_text})
                    st.session_state.lc_memory.save_context(
                        {"user_query": f"Requested critique for: '{orig_query_artifacts}'"},
                        {"output": f"Critique: {critique_text[:100]}..."})
                    critique_spinner_container.empty();
                    st.rerun()
        except Exception as e:
            st.error(f"Error during critique: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error generating critique: {e}"})
            if 'critique_spinner_container' in locals(): critique_spinner_container.empty()
            st.rerun()

# --- HTML Report Export Logic ---
if st.session_state.get("trigger_html_export", False):
    st.session_state.trigger_html_export = False
    artifacts_html = st.session_state.current_analysis_artifacts
    cdo_report_html = st.session_state.get("cdo_initial_report_text")
    data_summary_html = st.session_state.get("data_summary")

    if not artifacts_html or not cdo_report_html or not data_summary_html:
        st.error("Not enough information to generate HTML report. Ensure CDO workflow ran and analysis was performed.")
    else:
        with st.chat_message("assistant"):
            html_spinner_container = st.empty()
            html_spinner_container.markdown(
                f"⏳ **{st.session_state.selected_worker_model}** generating Bento HTML report...")
            with st.spinner("Generating HTML report via Gemini..."):
                html_file_path = export_report_to_html_via_gemini(artifacts_html, cdo_report_html, data_summary_html)
                if html_file_path and os.path.exists(html_file_path):
                    st.session_state.current_analysis_artifacts["generated_html_report_path"] = html_file_path
                    with open(html_file_path, "rb") as fp:
                        st.download_button(
                            label="📥 Download Bento HTML Report",
                            data=fp,
                            file_name=os.path.basename(html_file_path),
                            mime="text/html",
                            key=f"download_html_{datetime.datetime.now().timestamp()}"
                        )
                    st.success(f"Bento HTML report generated: {os.path.basename(html_file_path)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📄 Bento HTML Report generated and download link provided: `{os.path.abspath(html_file_path)}`"
                    })
                else:
                    st.error("Failed to generate Bento HTML report.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error while trying to generate the Bento HTML report."
                    })
                html_spinner_container.empty()
                # No st.rerun() here to keep the download button visible until next interaction
