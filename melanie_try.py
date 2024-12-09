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

# --- 初始化設定 ---
dotenv.load_dotenv()

def initialize_client(api_key):
    """根據 API 金鑰初始化 OpenAI 客戶端"""
    return OpenAI(api_key=api_key) if api_key else None

def generate_image_from_gpt_response(response, csv_data):
    """根據 GPT 回應生成圖表"""
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
                raise ValueError("盒狀圖需要有效的 Y 軸資料欄位。")

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
        st.error(f"生成圖表失敗：{e}")
        return None

def save_conversation_to_file():
    """保存對話內容和記憶體到 JSON 檔案"""
    try:
        messages = st.session_state.messages
        memory = st.session_state.memory.load_memory_variables({})
        data = {"messages": messages, "memory": memory}

        file_name = "conversation_history.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        st.success(f"對話內容已保存到 {file_name}")
    except Exception as e:
        st.error(f"保存對話失敗：{e}")

def load_conversation_from_file():
    """從 JSON 檔案載入對話內容和記憶體"""
    try:
        file_name = st.file_uploader("上傳對話記錄檔案", type="json")
        if file_name:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)

            st.session_state.messages = data.get("messages", [])
            memory_data = data.get("memory", {})
            if memory_data:
                st.session_state.memory = ConversationBufferMemory.from_memory_variables(memory_data)
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            st.success("對話記錄載入成功！")
    except Exception as e:
        st.error(f"載入對話記錄失敗：{e}")

def main():
    # --- 頁面設定 ---
    st.set_page_config(page_title="Chatbot + 資料分析", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 資料分析 + 🧠 記憶體")

    # --- 側邊欄 ---
    with st.sidebar:
        st.subheader("🔒 輸入您的 API 金鑰")
        api_key = st.text_input("OpenAI API 金鑰", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ 請輸入 API 金鑰以初始化 Chatbot。")
                return

        # 記憶體管理按鈕
        if st.button("🗑️ 清除記憶體"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("記憶體已清除！")

        # 顯示當前記憶體
        st.subheader("🧠 記憶體內容")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("當前記憶體", value=str(memory_content), height=200)

        # 上傳 CSV
        st.subheader("📂 上傳 CSV 檔案")
        uploaded_file = st.file_uploader("選擇 CSV 檔案：", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### 資料預覽")
            st.dataframe(csv_data)

    # --- 聊天界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示對話記錄
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])

    # 使用者輸入
    user_input = st.chat_input("Hi！問我任何問題...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 生成回應
        with st.spinner("思考中..."):
            try:
                prompt = f"請以繁體中文回答：{user_input}"
                response = st.session_state.conversation.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()
