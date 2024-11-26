import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# --- Helper Functions ---

def plot_line_chart(data, x_column, y_column):
    """生成折線圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_bar_chart(data, x_column, y_column):
    """生成柱狀圖"""
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color='skyblue')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.tight_layout()
    return plt

def plot_scatter_chart(data, x_column, y_column):
    """生成散點圖"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], color='purple')
    plt.title(f"{y_column} vs {x_column} (Scatter Plot)", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_pie_chart(data, column):
    """生成餅圖"""
    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f"Distribution of {column}", fontsize=16)
    plt.ylabel('')
    plt.tight_layout()
    return plt

def generate_summary(data):
    """生成數據摘要"""
    summary = data.describe(include='all').transpose()
    return summary

# --- Main Function ---
def main():
    st.set_page_config(page_title="Chart Generator", page_icon="📊", layout="wide")
    st.title("📊 數據分析與圖表生成器")

    st.sidebar.header("📂 上傳數據文件")
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 文件:", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### 數據預覽")
        st.dataframe(data)

        st.write("### 數據摘要")
        summary = generate_summary(data)
        st.dataframe(summary)

        st.sidebar.header("📈 圖表設置")
        chart_type_input = st.sidebar.text_input("輸入圖表類型 (例: 折線圖, 散點圖, 柱狀圖, 餅圖)")
        x_column = st.sidebar.selectbox("選擇 X 軸", data.columns)
        y_column = st.sidebar.selectbox("選擇 Y 軸", data.columns)

        if st.sidebar.button("生成圖表"):
            chart_type = chart_type_input.lower()

            if "折線" in chart_type or "line" in chart_type:
                fig = plot_line_chart(data, x_column, y_column)
                st.write("### 折線圖")
                st.pyplot(fig)

                # 下載圖表
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="下載圖表",
                    data=buf,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            elif "柱狀" in chart_type or "bar" in chart_type:
                fig = plot_bar_chart(data, x_column, y_column)
                st.write("### 柱狀圖")
                st.pyplot(fig)

                # 下載圖表
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="下載圖表",
                    data=buf,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            elif "散點" in chart_type or "scatter" in chart_type:
                fig = plot_scatter_chart(data, x_column, y_column)
                st.write("### 散點圖")
                st.pyplot(fig)

                # 下載圖表
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="下載圖表",
                    data=buf,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            elif "餅圖" in chart_type or "pie" in chart_type:
                column = st.sidebar.selectbox("選擇繪製餅圖的列", data.columns)
                fig = plot_pie_chart(data, column)
                st.write("### 餅圖")
                st.pyplot(fig)

                # 下載圖表
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="下載圖表",
                    data=buf,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            else:
                st.warning("無法識別的圖表類型，請重新輸入。")
    else:
        st.info("請在左側上傳 CSV 文件以繼續。")

if __name__ == "__main__":
    main()
