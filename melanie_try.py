import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
from PIL import Image
from datetime import datetime
import dotenv
import os
import time

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

dotenv.load_dotenv()

def generate_chart_code_snippet(chart_type, x_column, y_column):
    """根據參數組裝 Python 繪圖程式碼字串。"""
    code_snippet = f"""
plt.figure(figsize=(10, 6))
if chart_type == "line":
    plt.plot(csv_data['{x_column}'], csv_data['{y_column}'], marker='o')
elif chart_type == "bar":
    plt.bar(csv_data['{x_column}'], csv_data['{y_column}'], color='skyblue')
elif chart_type == "scatter":
    plt.scatter(csv_data['{x_column}'], csv_data['{y_column}'], alpha=0.7, edgecolors='b')
elif chart_type == "box":
    plt.boxplot(csv_data['{y_column}'], vert=True, patch_artist=True)
    plt.xticks([1], ['{y_column}'])

plt.title('{y_column} vs {x_column} ({chart_type.capitalize()} Chart)')
plt.xlabel('{x_column}' if chart_type != 'box' else '')
plt.ylabel('{y_column}')
plt.grid(True)
plt.tight_layout()
plt.show()
"""
    return code_snippet.strip()

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response and return the BytesIO buffer."""
    try:
        chart_type = response.get("chart_type", "line")
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])

        plt.figure(figsize=(10, 6))
        if chart_type == "line":
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
        elif chart_type == "bar":
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
        elif chart_type == "scatter":
            plt.scatter(csv_data[x_column], csv_data[y_column], alpha=0.7, edgecolors='b')
        elif chart_type == "box":
            if y_column in csv_data.columns:
                plt.boxplot(csv_data[y_column], vert=True, patch_artist=True)
                plt.xticks([1], [y_column])
            else:
                raise ValueError("Boxplot requires a valid column for Y-axis.")

        plt.title(f"{y_column} vs {x_column} ({chart_type.capitalize()} Chart)", fontsize=16)
        plt.xlabel(x_column if chart_type != "box" else "", fontsize=14)
        plt.ylabel(y_column, fontsize=14)
        plt.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)

        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory")

    # --- 側邊欄 ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        # 如果 memory/conversation 還沒初始化，就初始化
        if "conversation" not in st.session_state:
            if api_key:
                # ★ 在這裡加入 system message，明確告訴模型它能讀 CSV 並生成圖表 ★
                system_message = (
                    "你是一個可以使用內部工具讀取 CSV 資料並且產生圖表的 AI 助手。"
                    "使用者上傳的 CSV 會儲存在 csv_data，"
                    "你可以依據所需 x_column 與 y_column 幫助他們做圖表。"
                    "請直接輸出 JSON 格式（chart_type、x_column、y_column、content），"
                    "並且回答以繁體中文。"
                )

                # 初始化 LLM、記憶體
                st.session_state.chat_model = ChatOpenAI(
                    model="gpt-4", 
                    temperature=0.5, 
                    openai_api_key=api_key
                )
                st.session_state.memory = ConversationBufferMemory()
                
                # 先把 system_message 塞到記憶體裡
                st.session_state.memory.save_context(
                    {"role": "system", "content": system_message}, 
                    {}
                )

                # 建立對話鏈
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ Please enter the API key to initialize the chatbot.")
                return

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

    # 如果沒有 messages，就初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 先把歷史紀錄回放
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                # 顯示文字
                st.write(msg["content"])
                # 顯示程式碼
                if "code" in msg and msg["code"] is not None:
                    st.code(msg["code"], language="python")
                # 顯示圖表
                if "chart_buf" in msg and msg["chart_buf"] is not None:
                    st.image(msg["chart_buf"], caption="Generated Chart", use_container_width=True)

    # 等待使用者輸入
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # 儲存使用者訊息
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 即時顯示使用者訊息
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                # 準備 prompt
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
請用 JSON 格式回應：
{{
    "chart_type": "line",
    "x_column": "{csv_data.columns[0]}",
    "y_column": "{csv_data.columns[1]}",
    "content": "根據 {csv_data.columns[0]} 和 {csv_data.columns[1]} 的數據分析，這是我的觀察：{{在此處填入分析}}"
}}
使用者問題：{user_input}
可用欄位：{csv_columns}
"""
                else:
                    prompt = f"請以繁體中文回答：{user_input}"

                response = st.session_state.conversation.run(prompt)

                # 建立助理訊息結構
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "code": None,
                    "chart_buf": None
                }

                # 嘗試解析 JSON
                try:
                    response_json = json.loads(response)
                    text_feedback = response_json.get("content", "")
                    assistant_msg["content"] = text_feedback

                    if csv_data is not None:
                        chart_type = response_json.get("chart_type", "line")
                        x_column = response_json.get("x_column", csv_data.columns[0])
                        y_column = response_json.get("y_column", csv_data.columns[1])

                        # 生成程式碼
                        code_snippet = generate_chart_code_snippet(chart_type, x_column, y_column)
                        assistant_msg["code"] = code_snippet

                        # 生成圖表
                        chart_buf = generate_image_from_gpt_response(response_json, csv_data)
                        assistant_msg["chart_buf"] = chart_buf
                except (json.JSONDecodeError, TypeError):
                    # 如果 JSON 解析失敗，就把整段回應當作文字
                    assistant_msg["content"] = response

                # 將助理訊息存入 session_state
                st.session_state.messages.append(assistant_msg)

                # 即時顯示助理回應
                with st.chat_message("assistant"):
                    st.write(assistant_msg["content"])
                    if assistant_msg["code"] is not None:
                        st.code(assistant_msg["code"], language="python")
                    if assistant_msg["chart_buf"] is not None:
                        st.image(assistant_msg["chart_buf"], caption="Generated Chart", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
