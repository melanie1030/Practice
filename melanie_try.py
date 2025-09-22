import streamlit as st
import pandas as pd
import io
import google.generativeai as genai
import plotly.express as px
import os

# --- 新增 LangChain 相關套件 ---
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# --- 輔助函式：產生資料上下文 (已修正) ---
def get_df_context(df: pd.DataFrame) -> str:
    """
    為 DataFrame 產生一個文字描述的上下文，供 LLM 參考。
    此版本已修正變數未傳入的問題。
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # 為了清晰起見，先將 df.head() 的結果轉換為字串
    head_str = df.head().to_string()
    
    # 【修正處】將 info_str 和 head_str 正確地放入 f-string 中
    context = f"""
以下是您需要分析的 Pandas DataFrame 的詳細資訊。
DataFrame 變數名稱為 `df`。

1. DataFrame 的基本資訊 (df.info()):
```
{info_str}
```

2. DataFrame 的前 5 筆資料 (df.head()):
```
{head_str}
```
"""
    return context

# --- 核心函式 1：Pandas 分析師 Agent ---
def run_pandas_analyst_agent(api_key: str, df: pd.DataFrame, user_query: str) -> str:
    """
    第一階段：執行 Pandas Agent 進行數據探索，並回傳文字結論。
    """
    try:
        # 為了 Agent 的穩定性，推薦使用 OpenAI 的 gpt-4-turbo 或 gpt-3.5-turbo
        llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo", temperature=0)
        
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True, # 可以在終端機看到 Agent 的思考過程
            allow_dangerous_code=True # 允許 Agent 執行生成的 Python 程式碼
        )

        # 精心設計的 Prompt，指導 Agent 的任務
        agent_prompt = f"""
作為一名資深的數據分析師，你的任務是深入探索提供的 DataFrame (`df`)。
使用者的目標是："{user_query}"

請你執行以下步驟：
1.  徹底地探索和分析 `df`，找出其中最重要、最有趣、最值得透過視覺化來呈現的一個核心洞見。
2.  不要生成任何繪圖程式碼。
3.  你的最終輸出**必須是**一段簡潔的文字摘要。這段摘要需要清楚地描述你發現的洞見，並建議應該繪製什麼樣的圖表來展示這個洞見。

例如，你的結論可能是：「經過分析，我發現 '電子產品' 類別的總銷售額遠超其他類別。建議使用長條圖來清晰地比較各個產品類別的總銷售額。」

現在，請開始分析。
"""
        
        response = pandas_agent.invoke({"input": agent_prompt})
        return response['output']

    except Exception as e:
        return f"Pandas Agent 執行時發生錯誤: {e}"


# --- 核心函式 2：Plotly 視覺化 Coder ---
def generate_plot_code(api_key: str, df_context: str, user_query: str, analyst_conclusion: str = None) -> str:
    """
    第二階段：根據資料上下文、使用者問題和分析師結論，生成 Plotly Express 程式碼。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        # 根據是否有分析師結論，動態建構 Prompt
        if analyst_conclusion:
            prompt = f"""
你是一位頂尖的 Python 數據視覺化專家，精通使用 Plotly Express 函式庫。
你的任務是根據數據分析師的結論和使用者的原始目標，編寫一段 Python 程式碼來生成最合適的圖表。

**數據分析師的結論:**
{analyst_conclusion}

**原始使用者目標:**
"{user_query}"

**DataFrame 的資訊:**
{df_context}

**嚴格遵守以下規則:**
1.  你只能生成 Python 程式碼，絕對不能包含任何文字解釋、註解或 ```python 標籤。
2.  程式碼必須基於上述**數據分析師的結論**來生成。
3.  生成的程式碼必須使用 `plotly.express` (匯入為 `px`)。
4.  DataFrame 的變數名稱固定為 `df`。
5.  最終生成的圖表物件必須賦值給一個名為 `fig` 的變數。

現在，請生成程式碼：
"""
        else: # 原本的直接模式
            prompt = f"""
你是一位頂尖的 Python 數據視覺化專家，精通使用 Plotly Express 函式庫。
你的任務是根據提供的 DataFrame 資訊和使用者的要求，編寫一段 Python 程式碼來生成一個圖表。

**嚴格遵守以下規則:**
1.  你只能生成 Python 程式碼，絕對不能包含任何文字解釋、註解或 ```python 標籤。
2.  生成的程式碼必須使用 `plotly.express` (匯入為 `px`)。
3.  DataFrame 的變數名稱固定為 `df`。
4.  最終生成的圖表物件必須賦值給一個名為 `fig` 的變數。

**DataFrame 的資訊:**
{df_context}

**使用者的繪圖要求:**
"{user_query}"

現在，請生成程式碼：
"""
        
        response = model.generate_content(prompt)
        code = response.text.strip().replace("```python", "").replace("```", "").strip()
        return code

    except Exception as e:
        return f"繪圖程式碼生成時發生錯誤: {e}"


# --- Streamlit 主應用程式 ---
def main():
    st.set_page_config(page_title="進階圖表生成 Agent", page_icon="📊", layout="wide")
    st.title("📊 進階圖表生成 Agent")
    st.markdown("上傳 CSV，然後選擇模式：您可以直接命令 AI 畫圖，也可以讓 AI 先分析再畫圖！")

    # --- API Key 和檔案上傳 ---
    with st.sidebar:
        st.header("⚙️ API 金鑰設定")
        gemini_api_key = st.text_input("請輸入 Google Gemini API Key", type="password", help="用於生成繪圖程式碼。")
        openai_api_key = st.text_input("請輸入 OpenAI API Key", type="password", help="僅在『分析與繪圖模式』下需要，用於驅動 Pandas Agent。")
        
        st.divider()
        st.header("📁 資料上傳")
        uploaded_file = st.file_uploader("上傳您的 CSV 檔案", type=["csv"])

    if uploaded_file is None:
        st.info("請在側邊欄上傳一個 CSV 檔案以開始。")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤: {e}")
        return

    st.subheader("資料預覽")
    st.dataframe(df.head())
    
    st.divider()

    # --- 模式選擇 ---
    st.subheader("請選擇操作模式")
    agent_mode = st.radio(
        "模式選擇",
        ["直接繪圖模式", "分析與繪圖模式"],
        captions=[
            "您很清楚要畫什麼圖，請下達具體指令。",
            "您不確定要畫什麼，希望 AI 先分析數據找出洞見再畫圖。"
        ],
        horizontal=True
    )
    
    st.subheader("請下達您的指令")
    if agent_mode == "直接繪圖模式":
        user_query = st.text_area(
            "請輸入具體的繪圖指令：",
            "範例：畫出 x 軸是 'sepal_length'，y 軸是 'sepal_width' 的散點圖",
            height=100
        )
    else: # 分析與繪圖模式
        user_query = st.text_area(
            "請輸入模糊的、高層次的分析目標：",
            "範例：分析這份數據，幫我找出最重要的趨勢並視覺化",
            height=100
        )

    # --- 執行按鈕與主邏輯 ---
    if st.button("🚀 生成圖表", disabled=(not user_query)):
        
        generated_code = "" # 初始化變數以供後續使用
        # --- 直接繪圖模式邏輯 ---
        if agent_mode == "直接繪圖模式":
            if not gemini_api_key:
                st.error("請在側邊欄輸入您的 Google Gemini API Key！")
                return
            
            with st.spinner("AI 正在為您撰寫繪圖程式碼..."):
                df_context = get_df_context(df)
                generated_code = generate_plot_code(gemini_api_key, df_context, user_query)
            
            st.subheader("🤖 AI 生成的繪圖程式碼 (直接模式)")
            st.code(generated_code, language='python')
        
        # --- 分析與繪圖模式邏輯 ---
        else:
            if not openai_api_key or not gemini_api_key:
                st.error("分析模式需要同時在側邊欄輸入 Google Gemini 和 OpenAI 的 API Keys！")
                return

            analyst_conclusion = ""
            with st.status("執行分析與繪圖工作流...", expanded=True) as status:
                st.write("第一階段：Pandas Agent 正在進行深度數據分析...")
                analyst_conclusion = run_pandas_analyst_agent(openai_api_key, df, user_query)
                st.write("✅ 分析完成！")
                status.update(label="第一階段分析完成！")

                st.write("第二階段：視覺化 Coder 正在根據分析結論生成程式碼...")
                df_context = get_df_context(df)
                generated_code = generate_plot_code(gemini_api_key, df_context, user_query, analyst_conclusion)
                st.write("✅ 程式碼生成完成！")
                status.update(label="工作流執行完畢！", state="complete")

            st.subheader("🧐 Pandas Agent 的分析結論")
            st.info(analyst_conclusion)
            st.subheader("🤖 AI 生成的繪圖程式碼 (分析模式)")
            st.code(generated_code, language='python')

        # --- 統一的圖表渲染邏輯 ---
        st.subheader("📈 生成的圖表")
        if "error" in generated_code.lower():
             st.error(f"程式碼生成失敗：{generated_code}")
        elif generated_code: # 確保 generated_code 不是空的
            try:
                local_vars = {}
                exec(generated_code, {'df': df, 'px': px}, local_vars)
                fig = local_vars.get('fig')

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("程式碼執行成功，但未找到名為 'fig' 的圖表物件。")
            except Exception as e:
                st.error(f"執行生成程式碼時發生錯誤：\n{e}")

if __name__ == "__main__":
    main()
