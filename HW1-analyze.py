import time
import streamlit as st
import pandas as pd
import altair as alt

# 標題
st.title('第一次作業')

# 上傳任何類型的檔案
uploaded_file = st.file_uploader("上傳一個 CSV 或 JSON 檔案")

# 選擇要繪製的圖表類型
chart_type_options = ['折線圖', '長條圖', '盒鬚圖', '散點圖']
chart_type = st.sidebar.radio("選擇圖表類型", chart_type_options)

# 新增地圖選項
show_map_button = st.sidebar.button("顯示地圖")

# 在左侧栏添加颜色选项
color_options = ['藍色', '綠色', '紅色']
selected_color = st.sidebar.selectbox("選擇顏色", color_options)
if selected_color:
    if selected_color == '藍色':
        st.write("你選擇了藍色")
        # 在此处绘制相应的图表或进行其他操作
    elif selected_color == '綠色':
        st.write("你選擇了綠色")
        # 在此处绘制相应的图表或进行其他操作
    elif selected_color == '紅色':
        st.write("你選擇了紅色")
        # 在此处绘制相应的图表或进行其他操作

if uploaded_file is not None:
    # 根據檔案類型讀取數據
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    else:
        st.error("請上傳 CSV 或 JSON 檔案。")
        st.stop()  # 中止程式執行
    
    # 顯示數據表格
    st.write("上傳的數據表格：")
    st.write(df)
    
    # 左側欄
    with st.sidebar:
        # 表單選擇框
        with st.form(key='my_form'):
            x_option = st.selectbox('選擇 X 軸資料', df.columns)
            y_option = st.selectbox('選擇 Y 軸資料', df.columns)
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
        st.write(f"繪製{chart_type}")
        chart_data = df[[x_option, y_option]].dropna()
        
        if chart_type == '折線圖':
            st.line_chart(chart_data.set_index(x_option))
        elif chart_type == '長條圖':
            st.bar_chart(chart_data.set_index(x_option))
        elif chart_type == '盒鬚圖':
            if uploaded_file.name.endswith('.json'):
                st.error("此資料不適合使用盒鬚圖")
            else:
                try:
                    st.altair_chart(alt.Chart(chart_data).mark_boxplot().encode(
                        x=x_option,
                        y=y_option
                    ), use_container_width=True)
                except ValueError:
                    st.error("此資料無法顯示盒鬚圖")
        elif chart_type == '散點圖':
            st.scatter_chart(chart_data, x=x_option, y=y_option)
    
    # 按下按鈕才顯示地圖
    if show_map_button:
        if uploaded_file.name.endswith('.json'):
            st.write("顯示地圖：")
            st.map(df)
        else:
            st.write("資料無法使用地圖")
    
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

