import time
import streamlit as st
import numpy as np
import pandas as pd

# 标题
st.title('我的第一个应用程序')

# 上传 CSV 或 JSON 文件
uploaded_file = st.file_uploader("上传一个 CSV 或 JSON 文件", type=["csv", "json"])

if uploaded_file is not None:
    # 根据文件类型读取数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    # 显示数据表格
    st.write("上传的数据表格：")
    st.write(df)
    
    # 左侧栏
    with st.sidebar:
        # 表单选择框
        with st.form(key='my_form'):
            row_option = st.selectbox('选择要哪一行', 
                                      ['交易日期', '种类代码', '作物代号', '作物名称', '市场代号', '市场名称'])
            transaction_option = st.selectbox('想看哪笔交易量', 
                                              ['交易量', '上价', '中价', '下价', '平均价'])
            submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            # 显示选择的行和交易量数据
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"显示选择的行：{row_option}")
                st.write(df[[row_option]])
            
            with col2:
                st.write(f"显示选择的交易量：{transaction_option}")
                st.write(df[[transaction_option]])

    # 中间位置绘制图表
    if submit_button:
        st.write("绘制图表")
        chart_data = df[[row_option, transaction_option]].dropna()
        st.line_chart(chart_data.set_index(row_option))
    
    # 进度条
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.05)
    bar.progress(100, '加载完成！')
    
    # 消息通知
    if st.button('保存', type="primary"):
        st.toast(':rainbow[你编辑的内容已经保存]', icon='💾')
else:
    st.write("请上传一个 CSV 或 JSON 文件。")
