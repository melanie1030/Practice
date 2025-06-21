import streamlit as st
import pandas as pd

# 設定頁面標題與 icon
st.set_page_config(
    page_title="動態 CSV 分析器",
    page_icon="📊"
)

# --- 標題與說明 ---
st.title("📊 動態 CSV 資料分析器")
st.write("上傳您的 CSV 檔案，即可快速預覽資料並生成互動式圖表。")

# --- 步驟 1: 上傳 CSV 檔案 ---
st.header("1. 上傳您的 CSV 檔案")
uploaded_file = st.file_uploader("請選擇一個 .csv 檔案", type="csv")

# --- 條件式顯示區塊：只有在成功上傳檔案後才會顯示 ---
if uploaded_file is not None:
    # 讀取 CSV 檔案
    # 使用 try-except 來處理可能的編碼錯誤
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤: {e}")
        st.warning("您可以嘗試使用 UTF-8 或 Big5 編碼儲存您的 CSV 檔案後再試一次。")
        st.stop() # 如果讀取失敗，停止執行後續程式碼

    st.success("✅ 檔案上傳成功！")

    # --- 步驟 2: 顯示資料集預覽 ---
    st.header("2. 資料集預覽")
    st.write("以下是您上傳資料集的前五行：")
    st.dataframe(df.head()) # 使用 st.dataframe 以獲得更好的互動式表格

    # --- 步驟 3: 生成相關圖表 ---
    st.header("3. 資料視覺化")
    st.write("請選擇您想要視覺化的欄位。")

    # 偵測數值型態與類別型態的欄位，以提供不同的繪圖選項
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 如果資料集中有數值型欄位，才顯示繪圖選項
    if not numeric_columns:
        st.warning("您的資料集中沒有找到數值型態的欄位，無法繪製圖表。")
    else:
        # --- 長條圖 ---
        st.subheader("長條圖")
        
        # 讓使用者選擇 X 軸 (通常是類別) 和 Y 軸 (通常是數值)
        # 如果沒有類別欄位，就用第一個數值欄位當作 X 軸
        x_bar_axis = st.selectbox(
            "選擇長條圖的 X 軸 (類別):",
            options=categorical_columns if categorical_columns else numeric_columns,
            key='bar_x'
        )
        y_bar_axis = st.selectbox(
            "選擇長條圖的 Y 軸 (數值):",
            options=numeric_columns,
            key='bar_y'
        )

        if x_bar_axis and y_bar_axis:
            # 依據 X 軸分組並加總 Y 軸數值
            # 避免因類別過多導致圖表混亂，只取前 20 名
            bar_data = df.groupby(x_bar_axis)[y_bar_axis].sum().nlargest(20)
            st.bar_chart(bar_data)

        # --- 折線圖 ---
        st.subheader("折線圖")

        # 折線圖通常用來顯示趨勢，X 軸可以是數值或日期，Y 軸是數值
        x_line_axis = st.selectbox(
            "選擇折線圖的 X 軸:",
            options=numeric_columns + categorical_columns, # X 軸可以是任何類型
            key='line_x'
        )
        y_line_axis = st.selectbox(
            "選擇折線圖的 Y 軸 (數值):",
            options=numeric_columns,
            key='line_y'
        )

        if x_line_axis and y_line_axis:
            # 為了讓折線圖有意義，通常會先將 X 軸排序
            line_data = df.sort_values(by=x_line_axis)
            # 設定 X 軸和 Y 軸來繪圖
            st.line_chart(line_data.rename(columns={x_line_axis:'index'}).set_index('index')[y_line_axis])

else:
    st.info("請上傳一個 CSV 檔案以開始分析。")

# --- 側邊欄 ---
st.sidebar.header("關於")
st.sidebar.info(
    "這是一個使用 Streamlit 打造的互動式資料分析應用程式。\n\n"
    "由 Gemini AI 協助生成程式碼。"
)
