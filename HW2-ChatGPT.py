import streamlit as st
import openai

# 初始化 OpenAI API 金鑰
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key

# 訊息輸入區
user_input = st.text_input("請輸入訊息")

# 送出按鈕邏輯
if st.button("送出"):
    # 將使用者的訊息傳送給聊天機器人 (此處需加入 OpenAI Chat API 邏輯)
    pass

# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 如果有輸入訊息，將使用者的訊息新增到對話紀錄
if user_input:
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })

# 模擬回覆訊息，將機器人的訊息新增到對話紀錄
if st.button("模擬機器人回覆"):
    message = {
        "role": "bot",
        "content": "哈囉！我是聊天機器人。"
    }
    st.session_state["messages"].append(message)

# 顯示對話紀錄
for message in st.session_state["messages"]:
    st.chat_message(message)

# 計數器部分
if "counter" not in st.session_state:
    st.session_state["counter"] = 0

# 計數器按鈕
if st.button("增加計數器"):
    st.session_state["counter"] += 1

# 顯示計數器數值
st.write("計數器：", st.session_state["counter"])
