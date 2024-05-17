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
