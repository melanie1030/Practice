import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # 用于绘图
from io import BytesIO
from datetime import datetime


# --- Helper Functions ---

def plot_line_chart(data, x_column, y_column):
    """生成折线图."""
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_bar_chart(data, x_column, y_column):
    """生成柱状图."""
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color='skyblue')
    plt.title(f"{y_column} vs {x_column}", fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.tight_layout()
    return plt

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chart Generator", page_icon="📊", layout="wide")
    st.title("📊 数据驱动图表生成器")

    # --- Data Upload ---
    st.sidebar.header("📂 上传数据文件")
    uploaded_file = st.sidebar.file_uploader("上传 CSV 文件:", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### 数据预览:")
        st.dataframe(data)

        # --- Chart Settings ---
        st.sidebar.header("📈 图表设置")
        chart_type = st.sidebar.selectbox("选择图表类型", ["折线图", "柱状图"])
        x_column = st.sidebar.selectbox("选择 X 轴", data.columns)
        y_column = st.sidebar.selectbox("选择 Y 轴", data.columns)

        if st.sidebar.button("生成图表"):
            if chart_type == "折线图":
                fig = plot_line_chart(data, x_column, y_column)
                st.write("### 折线图:")
                st.pyplot(fig)
            elif chart_type == "柱状图":
                fig = plot_bar_chart(data, x_column, y_column)
                st.write("### 柱状图:")
                st.pyplot(fig)

            # --- Export Chart ---
            st.write("### 📥 下载生成的图表")
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="下载图表",
                data=buf,
                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
    else:
        st.info("请在左侧上传 CSV 文件以继续。")


if __name__ == "__main__":
    main()
