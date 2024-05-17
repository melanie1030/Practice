import time
import streamlit as st
import numpy as np
import pandas as pd

# 標題
st.title('我的第一個應用程式')

# 上傳 CSV 或 JSON 文件
uploaded_file = st.file_uploader("上傳一個 CSV 或 JSON 文件", type=["csv", "json"])

if uploaded_file is not None:
    # 根據文件類型讀取數據
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    # 顯示數據表格
    st.write("上傳的數據表格：")
    st.write(df)
    
    # 表單選擇框
    with st.form(key='my_form'):
        row_option = st.selectbox('選擇要哪一行', 
                                  ['交易日期', '種類代碼', '作物代號', '作物名稱', '市場代號', '市場名稱'])
        transaction_option = st.selectbox('想看哪筆交易量', 
                                          ['交易量', '上價', '中價', '下價', '平均價'])
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"選擇行：{row_option}")
            st.write(df[[row_option]])
        
        with col2:
            st.write(f"選擇價位或交易量：{transaction_option}")
            st.write(df[[transaction_option]])
    
    # 繪製圖表
    chart_data = df.select_dtypes(include=[np.number])  # 只選擇數值列
    if not chart_data.empty:
        st.line_chart(chart_data)
    else:
        st.write("沒有數值列可用於繪製圖表。")
    
    # 進度條
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.05)
    bar.progress(100, '載入完成！')
    
    # 消息通知
    if st.button('儲存', type="primary"):
        st.toast(':rainbow[你編輯的內容已經保存]', icon='💾')
else:
    st.write("請上傳一個 CSV 或 JSON 文件。")
