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
    
    # 提供選擇框-側邊欄
    option = st.sidebar.selectbox(
        '你想查看哪一列？',
        df.columns.tolist()
    )
    st.sidebar.text(f'你選擇的列：{option}')
    
    # 顯示選擇的列
    st.write(f"顯示選擇的列：{option}")
    st.write(df[[option]])
    
    # 繪製圖表
    chart_data = df.select_dtypes(include=[np.number])  # 只選擇數值列
    if not chart_data.empty:
        st.line_chart(chart_data)
    else:
        st.write("沒有數值列可用於繪製圖表。")
    
    # 繪製地圖
    if 'lat' in df.columns and 'lon' in df.columns:
        if st.checkbox('顯示地圖圖表'):
            st.map(df)
    else:
        st.write("數據中沒有 'lat' 和 'lon' 列，無法繪製地圖。")
    
    # 繪製按鈕
    if st.button('不要按!'):
        st.text("不是叫你不要按了嗎！")
    
    # 進度條
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.05)
    bar.progress(100, '載入完成！')
    
    # 消息通知
    if st.button('儲存', type="primary"):
        st.toast(':rainbow[你編輯的內容已經保存]', icon='💾')
    
    with st.form(key='my_form'):
        form_name = st.text_input(label='姓名', placeholder='請輸入姓名')
        form_gender = st.selectbox('性別', ['男', '女', '其他'])
        form_birthday = st.date_input("生日")
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        st.write(f'hello {form_name}, 性別:{form_gender}, 生日:{form_birthday}')
else:
    st.write("請上傳一個 CSV 或 JSON 文件。")
