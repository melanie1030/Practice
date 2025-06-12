import streamlit as st
import os
import openai
from langchain_openai import ChatOpenAI
import httpx
import traceback

# --- 測試一：原生 OpenAI 函式庫 ---
def test_native_openai(api_key):
    st.info("--- 開始測試一：原生 OpenAI ---")
    if not api_key:
        st.error("API Key 為空，無法測試。")
        return

    # 使用我們之前最穩健的代理處理邏輯
    PROXY_ENV_VARS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
    original_proxies = {key: os.environ.get(key) for key in PROXY_ENV_VARS if os.environ.get(key)}
    proxy_url_for_httpx = next((p for p in original_proxies.values() if p), None)
    
    for key in original_proxies:
        if key in os.environ: del os.environ[key]
    
    client = None
    try:
        st.write("環境變數已暫時移除，準備初始化 Client...")
        http_client = httpx.Client(proxies=proxy_url_for_httpx) if proxy_url_for_httpx else None
        
        # 直接初始化基礎 OpenAI Client
        client = openai.OpenAI(api_key=api_key, http_client=http_client)
        
        st.write("原生 OpenAI Client 初始化成功！正在嘗試發送請求...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say this is a test!"}],
            max_tokens=5
        )
        
        st.success(f"✅ **測試一成功！** 模型回應: {response.choices[0].message.content}")

    except Exception as e:
        st.error(f"❌ **測試一失敗！**")
        st.code(f"錯誤類型: {type(e)}\n錯誤訊息: {e}")
        st.text("錯誤追蹤:")
        st.code(traceback.format_exc())
    finally:
        # 恢復環境變數
        for key, value in original_proxies.items():
            os.environ[key] = value
        st.info("環境變數已恢復。")


# --- 測試二：LangChain 的 ChatOpenAI ---
def test_langchain_openai(api_key):
    st.info("--- 開始測試二：LangChain ChatOpenAI ---")
    if not api_key:
        st.error("API Key 為空，無法測試。")
        return

    try:
        st.write("準備初始化 LangChain 的 ChatOpenAI...")
        
        # 這是我們一直遇到問題的地方
        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o", 
            api_key=api_key
        )
        
        st.write("LangChain ChatOpenAI 初始化成功！正在嘗試發送請求...")
        
        response = llm.invoke("Say this is a test!")
        
        st.success(f"✅ **測試二成功！** 模型回應: {response.content}")

    except Exception as e:
        st.error(f"❌ **測試二失敗！**")
        st.code(f"錯誤類型: {type(e)}\n錯誤訊息: {e}")
        st.text("錯誤追蹤:")
        st.code(traceback.format_exc())


# --- 主應用介面 ---
def main():
    st.title("🔴 最終根本原因診斷")
    st.write("這個頁面用來隔離並找出導致 `proxies` 錯誤的根本原因。")

    api_key = st.text_input(
        "請輸入您的 OpenAI API Key", 
        type="password", 
        help="此金鑰僅用於本次測試，不會被儲存。"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("執行測試一：原生 OpenAI 函式庫", use_container_width=True):
            test_native_openai(api_key)

    with col2:
        if st.button("執行測試二：LangChain 的 ChatOpenAI", use_container_width=True, type="primary"):
            test_langchain_openai(api_key)
            
if __name__ == "__main__":
    main()
