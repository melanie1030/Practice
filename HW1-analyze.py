import time
import streamlit as st
import numpy as np
import pandas as pd
#標題
st.title('我的第一個應用程式')
#建立表格
st.write("嘗試創建**表格**：")

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
df
#繪製圖表
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)
#繪製地圖
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [22.6, 120.4],
    columns=['lat', 'lon'])
st.map(map_data)

#繪製按鈕
if st.button('不要按!'):
    st.text("不是叫你不要按了嗎！")
