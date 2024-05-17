import time
import streamlit as st
import numpy as np
import pandas as pd

# 標題
st.title('我的第一個應用程式')

# 上傳 CSV 或 JSON 檔案
uploaded_file = st.file_uploader("上傳一個 CSV 或 JSON 檔案", type=["csv", "json"])

if uploaded_file is not None:
    # 根據檔案類型讀取數據
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    # 顯示數據表格
    st.write("上傳的數據表格：")
    st.write(df)
    
    # 左側欄
    with st.sidebar:
        # 表單選擇框
        with st.form(key='my_form'):
            x_option = st.selectbox('選擇 X 軸資料', 
                                      ['交易日期', '種類代碼', '作物代號', '作物名稱', '市場代號', '市場名稱'])
            y_option = st.selectbox('選擇 Y 軸資料', 
                                    ['交易量', '上價', '中價', '下價', '平均價'])
            submit_button = st.form_submit_button(label='提交')
        
        if submit_button:
            # 顯示選擇的 X 軸和 Y 軸數據
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"顯示選擇的 X 軸資料：{x_option}")
                st.write(df[[x_option]])
            
            with col2:
                st.write(f"顯示選擇的 Y 軸資料：{y_option}")
                st.write(df[[y_option]])

    # 中間位置繪製圖表
    if submit_button:
        st.write("繪製散點圖")
        chart_data = df[[x_option, y_option]].dropna()
        st.scatter_chart(chart_data.set_index(x_option))
    
    # 進度條
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.05)
    bar.progress(100, '加載完成！')
    
    # 消息通知
    if st.button('保存', type="primary"):
        st.toast(':rainbow[你編輯的內容已經保存]', icon='💾')
else:
    st.write("請上傳一個 CSV 或 JSON 檔案。")
