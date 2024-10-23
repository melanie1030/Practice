import streamlit as st
import subprocess
import json

# 直接硬編 API Key，僅用於測試
OPENAI_API_KEY = "sk-svcacct-fb_-GzpFTmE6wtv222EkZdGrZrVUnZdTIP-AkvTvtcxO8n7D-tZvHHAL6ChEGT3BlbkFJCwdg-PbyzjyhbVo99UJNUKYTHayGD-I0QpeVibX_K7x6F8UE9Q7j0flr-VmAA"

# Streamlit App 標題
st.title("ChatGPT Service 打造 🤖")
st.subheader("您好!! 歡迎您問我答~")

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "你是一個幫助人的助理，請用繁體中文回答。"}
    ]

# 自訂聊天氣泡的 CSS 樣式
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
    }
    .ai-bubble {
        background-color: #E8E8E8;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
    }
    .user-container, .ai-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .user-container img, .ai-container img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    .user-container {
        justify-content: flex-end;
    }
    .user-container img {
        margin-left: 10px;
    }
    .ai-container {
        justify-content: flex-start;
    }
    .ai-container img {
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 定義函數來渲染訊息
def render_messages():
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            if message["role"] == "system":
                continue
            elif message["role"] == "user":
                st.markdown(f"""
                <div class="user-container">
                    <div class="user-bubble">{message['content']}</div>
                    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEh6XGT5Hz9MpAiyfTHlBczavuUjyTBza9zWdzYmoifglj0p1lsylcTEScnpSa-Youh7YXw-ssgO-mMQmw-DBz4NeesioQPTe8beOH_QS-A4JMnfZAGP-01gxPQrS-pPEnrnJxbdVnWguhCC/s1600/pose_pien_uruuru_woman.png" alt="User">
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="ai-container">
                    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjCHBgyqLrwRdbSM72R9PutXIqxbI9yR5UzXWC0TYIYVlKgHH5TzkaHijRkdxQMRSJx8upcecs2RGHYW7gVOSQPH-LUrPUg3esbqx5-7Q04BPJWD-DdzTealzGBQehfXpDeLxYe29MjQQgo/s1600/megane_hikaru_woman.png" alt="AI">
                    <div class="ai-bubble">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)

# 創建一個佔位符來顯示聊天訊息
chat_placeholder = st.empty()
render_messages()

# 在螢幕底部的輸入框
user_input = st.chat_input("輸入訊息：")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_messages()

    with st.spinner("AI 正在回應..."):
        try:
            # 將 messages 轉換為 JSON 字符串
            messages_json = json.dumps(st.session_state["messages"])

            # 構建 curl 命令
            curl_command = [
                "curl", "https://api.openai.com/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {OPENAI_API_KEY}",
                "-d", json.dumps({
                    "model": "gpt-3.5-turbo",
                    "messages": st.session_state["messages"]
                })
            ]

            # 使用 subprocess 執行 curl 命令
            result = subprocess.run(curl_command, capture_output=True, text=True)

            # 檢查是否有錯誤輸出
            if result.stderr:
                st.error(f"請求錯誤：{result.stderr}")
            else:
                # 解析回應
                response = json.loads(result.stdout)
                if 'choices' in response and len(response['choices']) > 0:
                    full_response = response['choices'][0]['message']['content'].strip()
                    st.session_state["messages"].append({"role": "assistant", "content": full_response})
                    render_messages()
                else:
                    st.error("未收到 AI 的回應。")
        except Exception as e:
            st.error(f"發生錯誤：{e}")
