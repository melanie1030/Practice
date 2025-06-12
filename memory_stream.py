import os
import openai
import httpx
import traceback
from dotenv import load_dotenv

print("--- 開始執行 OpenAI Client 初始化偵錯腳本 ---")

# 載入 .env 檔案中的環境變數 (主要用於讀取代理)
load_dotenv()

# --- 修改處：改為直接從終端機輸入 API Key ---
api_key = input("請在此貼上您的 OpenAI API Key 並按下 Enter: ")

# 從環境中讀取代理設定
proxy_from_env = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

print(f"已收到 API Key (前五位): {str(api_key)[:5]}...")
print(f"讀取到的代理 (HTTPS_PROXY/HTTP_PROXY): {proxy_from_env}")

# ------------------------------------------------------------------
# 這是我們在 Streamlit 中嘗試的最終解決方案的隔離版本
# ------------------------------------------------------------------

# 備份並暫時移除代理環境變數
PROXY_ENV_VARS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
original_proxies = {}
for key in PROXY_ENV_VARS:
    if key in os.environ:
        original_proxies[key] = os.environ[key]
        del os.environ[key]
        print(f"已暫時移除環境變數: {key}")

client_instance = None
try:
    print("\n--- 進入初始化區塊 (已移除環境變數) ---")
    if proxy_from_env:
        print(f"準備手動設定代理: {proxy_from_env}")
        proxies_for_httpx = {"http://": proxy_from_env, "https://": proxy_from_env}
        http_client = httpx.Client(proxies=proxies_for_httpx)
        
        print("嘗試初始化 OpenAI.Client (帶有 http_client)...")
        client_instance = openai.OpenAI(api_key=api_key, http_client=http_client)
        
    else:
        print("嘗試初始化 OpenAI.Client (不帶代理)...")
        client_instance = openai.OpenAI(api_key=api_key)

    print("\n--- 初始化成功！---")
    print(f"成功創建 Client 實例: {type(client_instance)}")

except Exception as e:
    print("\n--- 初始化失敗！---")
    print(f"遇到的錯誤類型: {type(e)}")
    print(f"錯誤訊息: {e}")
    print("--- 錯誤追蹤 (Traceback) ---")
    traceback.print_exc()

finally:
    # 無論成功或失敗，都恢復環境變數
    for key, value in original_proxies.items():
        os.environ[key] = value
    print("\n--- 已恢復原始環境變數 ---")

print("--- 偵錯腳本執行完畢 ---")
