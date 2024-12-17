import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

# 设置页面配置
st.set_page_config(page_title="在 Streamlit 中使用 PyGWalker", layout="wide")

# 应用标题
st.title("在 Streamlit 中使用 PyGWalker")

# 上传 CSV 文件
uploaded_file = st.file_uploader("请上传您的 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取 CSV 文件
    df = pd.read_csv(uploaded_file)
    st.write("数据预览：")
    st.write(df.head())

    # 使用 PyGWalker 进行数据可视化
    st.subheader("交互式数据可视化")
    renderer = StreamlitRenderer(df)
    renderer.explorer()
else:
    st.info("请上传 CSV 文件以进行数据可视化。")
