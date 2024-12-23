import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
from PIL import Image
from datetime import datetime
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os
import time

# --- 新增：匯入 pygwalker ---
import pygwalker as pyg

dotenv.load_dotenv()

def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def display_code_line_by_line_in_block(code_snippet):
    """Display code one line at a time within a single code block."""
    displayed_code = ""
    code_lines = code_snippet.split("\n")
    code_placeholder = st.empty()
    for line in code_lines:
        if line.strip():
            displayed_code += line + "\n"
            code_placeholder.code(displayed_code, language="python")
            time.sleep(0.2)

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response and save it in session_state."""
    try:
        chart_type = response.get("chart_type", "line")
        x_column = response.get("x_column", csv_data.columns[0] if csv_data is not None else "")
        y_column = response.get("y_column", csv_data.columns[1] if csv_data is not None else "")

        # 這裡只是示範如何用 matplotlib 產生圖表，你也可以同時或改用 PyGWalker
        code_snippet = f"""
        # 以下示範如何生成 matplotlib 圖表
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
        display_code_line_by_line_in_block(code_snippet)

        # 產生 matplotlib 圖表
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

        # 用 BytesIO 暫存圖表
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)

        if "charts" not in st.session_state:
            st.session_state.charts = []
        st.session_state.charts.append(buf)

        return buf

    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None

# --- 新增：示範如何利用 GPT 回傳的 pygwalker_spec 參數，生成 PyGWalker 圖表 ---
def generate_pygwalker_from_spec(df, pygwalker_spec):
    """
    根據 GPT 回傳的 pygwalker_spec 生成 PyGWalker 圖表。
    pygwalker_spec 是一個 dict，裡頭包含 X/Y 軸、圖表類型等設定。
    """
    try:
        # 直接用 pyg.walk(df, spec=pygwalker_spec) 來渲染
        st.subheader("PyGWalker Visualization")
        pyg.walk(df, spec=pygwalker_spec)
    except Exception as e:
        st.error(f"Failed to render PyGWalker chart: {e}")

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory (with PyGWalker)")

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = ChatOpenAI(
                    model="gpt-4-turbo", 
                    temperature=0.5, 
                    openai_api_key=api_key
                )
                st.session_state.memory = ConversationBufferMemory()
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

        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            img_bytes = BytesIO(uploaded_image.read())
            st.session_state.messages.append({"role": "user", "image": img_bytes.getvalue()})
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示之前產生的圖表
    if "charts" in st.session_state and st.session_state.charts:
        st.write("### Previously Generated Charts")
        for i, chart_buf in enumerate(st.session_state.charts):
            st.image(chart_buf, caption=f"Chart {i + 1}", use_container_width=True)
            st.download_button(
                label=f"Download Chart {i + 1}",
                data=chart_buf,
                file_name=f"chart_{i + 1}.png",
                mime="image/png"
            )

    # 顯示聊天紀錄
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "image" in message:
                img = Image.open(BytesIO(message["image"]))
                st.image(img, caption="Uploaded Image", use_container_width=True)

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                # 修改 Prompt：要求 GPT 回傳 JSON，並包含 pygwalker_spec & contentx
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
請直接回傳 JSON，格式如下：
{{
    "chart_type": "line",
    "x_column": "{csv_data.columns[0]}",
    "y_column": "{csv_data.columns[1]}",
    "pygwalker_spec": {{
        "dataSource": "...",
        "fields": [...],
        "visualization": {{
            "type": "bar_chart",
            "x": "Category",
            "y": "Values"
        }}
    }},
    "contentx": "在這裡輸出你的分析說明（繁體中文）"
}}

注意：
1. "pygwalker_spec" 會用來給 PyGWalker Render 圖表，你可以修改 "visualization" 裡的 type, x, y 等參數。
2. "chart_type" 用於產生 matplotlib 的圖（line, bar, scatter, box）。
3. "x_column" 與 "y_column" 為你的資料欄位名稱，必須是以下欄位其中之一： {csv_columns}
4. "contentx" 請寫上你的分析結論，全程使用繁體中文回應。

依照上面的 JSON 結構回傳，不要添加多餘內容。以下為使用者問題：{user_input}
                    """
                else:
                    prompt = f"""
請直接回傳 JSON，格式如下：
{{
    "chart_type": "line",
    "x_column": "預設X",
    "y_column": "預設Y",
    "pygwalker_spec": {{
        "dataSource": "...",
        "fields": [...],
        "visualization": {{
            "type": "bar_chart",
            "x": "Category",
            "y": "Values"
        }}
    }},
    "contentx": "在這裡輸出你的分析說明（繁體中文）"
}}
使用者的提問：{user_input}
                    """

                response = st.session_state.conversation.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})

                with st.chat_message("assistant"):
                    try:
                        response_json = json.loads(response)
                        # 讀取 GPT 回傳的欄位
                        chart_type = response_json.get("chart_type", "line")
                        x_column = response_json.get("x_column", None)
                        y_column = response_json.get("y_column", None)
                        pygwalker_spec = response_json.get("pygwalker_spec", {})
                        text_feedback = response_json.get("contentx", "")

                        # 顯示 GPT 的分析說明
                        st.write(text_feedback)

                        # 如果有上傳 csv 就產生圖表
                        if csv_data is not None:
                            # (1) 產生 matplotlib 圖表
                            chart_buf = generate_image_from_gpt_response(response_json, csv_data)
                            if chart_buf:
                                st.image(chart_buf, caption="Generated Chart", use_container_width=True)

                            # (2) 產生 PyGWalker 圖表
                            # 如果要讓 PyGWalker spec 完整：建議將資料與欄位帶入 pygwalker_spec
                            # 這裡只示範「dataSource」和「fields」的概念
                            # 你可以在 GPT 的回答中塞入對應 config
                            if isinstance(pygwalker_spec, dict):
                                # 自行更新 pygwalker_spec["dataSource"] 為真正的 df
                                # 這裡只是示範
                                # pygwalker_spec["dataSource"] = "Your DF"
                                # pygwalker_spec["fields"] = [
                                #     {"name": x_column, "type": "dimension"},
                                #     {"name": y_column, "type": "measure"},
                                # ]
                                generate_pygwalker_from_spec(csv_data, pygwalker_spec)
                            else:
                                st.write("沒有有效的 pygwalker_spec，無法生成 PyGWalker 圖表。")

                        else:
                            st.write("未上傳 CSV，因此無法產生圖表。")

                    except (json.JSONDecodeError, TypeError):
                        # 如果不是 JSON 或解析失敗，就直接顯示文字
                        st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
