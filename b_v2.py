import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
import io

# -- LangChain 相關 --
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# -- Editor --
from streamlit_ace import st_ace

# -- 以下為您測試成功的 openai<1.0.0 方式 --
from openai import OpenAI
from PIL import Image
from audio_recorder_streamlit import audio_recorder  # 如果有需要保留錄音功能，可保留
import random

# --- dotenv 加載環境 ---
dotenv.load_dotenv()

# === 全域設定 ===
UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4o",       # 假設可解析圖片的實驗模型
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k"
]

# -------------------------------------------------
# 以下區塊: b_v2.py 裡的輔助函式
# -------------------------------------------------
def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.write(msg)
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.error(msg)
        print(msg)

def initialize_client(api_key, model_name):
    """
    原 b_v2 用 LangChain ChatOpenAI
    """
    return ChatOpenAI(
        model=model_name,
        temperature=0.5,
        openai_api_key=api_key
    ) if api_key else None

def save_uploaded_file(uploaded_file):
    """將上傳檔案存至指定資料夾"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"DEBUG: saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"DEBUG: files in {UPLOAD_DIR}: " + str(os.listdir(UPLOAD_DIR)))
    return file_path

def execute_code(code, global_vars=None):
    """執行使用者產生的Python程式碼"""
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("DEBUG: Ready to exec the following code:")
        if st.session_state.get("debug_mode", False):
            st.code(code, language="python")

        debug_log("[DEBUG] Exec code with global_vars: " + str(list(exec_globals.keys())))
        exec(code, exec_globals)
        return "Code executed successfully. Output: " + str(exec_globals.get("output", "(No output returned)"))
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log("[DEBUG] Execution error: " + error_msg)
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode)."

def extract_json_block(response: str) -> str:
    """從文字中擷取 JSON 內容的區塊 (以 ```json ... ``` 為主)"""
    pattern = r'```(?:json)?(.*)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json_str
    else:
        return response.strip()

# -------------------------------------------------
# 以下區塊: from openai import OpenAI 方式所需
#         參考您給的 snippet
# -------------------------------------------------

def initialize_openai_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def load_image_base64(image):
    """Convert a PIL Image to Base64 encoding."""
    buffer = io.BytesIO()
    # 若 image.format 不存在，可手動指定 PNG
    image.save(buffer, format=image.format if image.format else "PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def add_user_image(image):
    """Add an image message to st.session_state.messages (compat with streaming logic)."""
    img_base64 = load_image_base64(image)
    # 加到 messages 末尾
    st.session_state.messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    })

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")

def stream_llm_response(client, model_params):
    """
    以 for-chunk 方式串流回傳文字。
    參考您給的 snippet: client.chat.completions.create(...)
    """
    for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-4o"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=4096,
            stream=True):
        chunk_text = chunk.choices[0].delta.content or ""
        yield chunk_text

# -------------------------------------------------
# 合併主程式 main()
# -------------------------------------------------
def main():
    st.set_page_config(page_title="Chatbot + Data Analysis + StreamingImage", 
                       page_icon="🤖", 
                       layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Editor + (OpenAI streaming for only-image)")

    # --- 初始化 session state 變數 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""
    if "editor_location" not in st.session_state:
        st.session_state.editor_location = "Main"
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state:
        st.session_state.deep_analysis_mode = False
    if "second_response" not in st.session_state:
        st.session_state.second_response = ""
    if "third_response" not in st.session_state:
        st.session_state.third_response = ""
    if "deep_analysis_image" not in st.session_state:
        st.session_state.deep_analysis_image = None

    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        # -- LangChain ChatOpenAI 初始化 (b_v2)
        selected_model = st.selectbox("選擇模型 (for b_v2 logic):", OPENAI_MODELS, index=0)
        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = initialize_client(api_key, selected_model)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ 請輸入 API Key 以初始化聊天機器人 (LangChain).")
                return

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("深度分析模式", value=False)

        # -- openai client (from openai import OpenAI)
        #    用於「只有圖片時」的 streaming chat
        if "client_4o" not in st.session_state:
            st.session_state.client_4o = initialize_openai_client(api_key)
        else:
            # 若後續要更新key
            st.session_state.client_4o = initialize_openai_client(api_key)

        if st.button("🗑️ Clear Memory (b_v2 & messages)"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.ace_code = ""
            st.session_state.uploaded_file_path = None
            st.session_state.uploaded_image_path = None
            st.session_state.image_base64 = None
            st.session_state.deep_analysis_mode = False
            st.session_state.second_response = ""
            st.session_state.third_response = ""
            st.session_state.deep_analysis_image = None
            st.success("Memory cleared!")

        st.subheader("🧠 Memory State (LangChain b_v2)")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        # -- 圖像上傳/拍照 (4o snippet) (可以和 b_v2 的功能共存)
        st.write("### 4o: 上傳圖像或拍照 (Streaming only-image scenario)")
        uploaded_img_4o = st.file_uploader("選擇一張圖片:", type=["png", "jpg", "jpeg"], key="4o_uploader")
        if uploaded_img_4o:
            img = Image.open(uploaded_img_4o)
            add_user_image(img)
            st.success("圖像已上傳至 messages!")

        camera_img = st.camera_input("拍照")
        if camera_img:
            cimg = Image.open(camera_img)
            add_user_image(cimg)
            st.success("拍照已成功，已加入messages!")

        # -- 重置對話 (4o snippet)
        st.button("🗑️ 清除對話 (stream messages)", on_click=reset_session_messages)

        # -- CSV 上傳 (b_v2)
        st.subheader("📂 Upload a CSV File (b_v2 style)")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"], key="b_v2_csv")
        csv_data = None
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            debug_log(f"DEBUG: st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading CSV: {e}")
                debug_log(f"[DEBUG] Error reading CSV: {e}")

        # -- 圖片上傳 (b_v2)；可以與 4o snippet 併存
        st.subheader("🖼️ Upload an Image (b_v2 style)")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="b_v2_image")
        if uploaded_image:
            st.session_state.uploaded_image_path = save_uploaded_file(uploaded_image)
            debug_log(f"DEBUG: st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

            st.image(st.session_state.uploaded_image_path, caption="Uploaded Image Preview", use_column_width=True)
            try:
                with open(st.session_state.uploaded_image_path, "rb") as f:
                    img_bytes = f.read()
                st.session_state.image_base64 = base64.b64encode(img_bytes).decode("utf-8")
                debug_log("DEBUG: Image has been converted to base64.")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error converting image to base64: {e}")
                debug_log(f"[DEBUG] Error converting image to base64: {e}")

        # -- Editor Location
        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location

    # ---- 顯示對話歷史 (messages) ----
    st.write("## Current Messages:")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 逐則輸出 messages
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            # message["content"] 可能是list( 4o snippet )或單純string(b_v2)
            contents = message["content"]
            if isinstance(contents, list):
                # 4o snippet: content是list of dict
                for c in contents:
                    if c["type"] == "text":
                        st.write(c.get("text", ""))
                    elif c["type"] == "image_url":
                        st.image(c["image_url"].get("url", ""), caption="User's image")
            else:
                # b_v2: 直接是string
                st.write(contents)

            if "code" in message:  # b_v2: 可能有 code
                st.code(message["code"], language="python")

    # ---- User input for b_v2 logic ----
    user_input = st.chat_input("b_v2 or 4o: Hi! Ask me anything ...")
    if user_input:
        # 先把這次 user_input 加入對話
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                debug_log(f"DEBUG: uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: uploaded_image_path = {st.session_state.uploaded_image_path}")

                # 判斷「只有圖片且沒有 CSV」 => 代表 b_v2_image or 4o_uploader/camera 有圖，但 CSV 是 None
                # 但是現在 messages 可能也有 4o snippet 的圖
                # 我們以 b_v2 方式判斷: st.session_state.uploaded_file_path is None => 沒CSV
                # 且 (st.session_state.uploaded_image_path is not None or messages裡面有 image_url)
                
                # 簡化: 如果 "uploaded_file_path" is None (沒csv) AND "uploaded_image_path" is not None => only image
                only_image = False
                if (st.session_state.uploaded_file_path is None) and (st.session_state.uploaded_image_path is not None):
                    only_image = True

                # 另外一種檢查: messages 中是否存在 image_url?
                # 但這可能會與 b_v2 相衝，先用上面 simpler method

                if only_image and st.session_state.client_4o is not None:
                    # 改用 streaming approach (client_4o)

                    debug_log("DEBUG: Only Image scenario => using client.chat.completions.create streaming...")

                    # 這裡 content 可能是文字 + base64（若 b_v2 圖片），或者 snippet 中 user_image
                    # 不過我們已經把 user_input append 了 => messages
                    # 預備 streaming: model_params
                    model_params = {
                        "model": "gpt-4o",  # 或可改成 selected_model
                        "temperature": 0.3
                    }
                    # 直接呼叫 stream_llm_response
                    full_response = ""
                    with st.chat_message("assistant"):
                        stream_placeholder = st.empty()
                        try:
                            for chunk_text in stream_llm_response(st.session_state.client_4o, model_params):
                                full_response += chunk_text
                                stream_placeholder.markdown(full_response)
                        except Exception as e:
                            if st.session_state.debug_mode:
                                st.error(f"Streaming error: {e}")
                            debug_log(f"[DEBUG] Streaming error: {e}")

                    # 將串流最終結果寫回 messages
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                else:
                    # 其餘情況 => 走舊有 b_v2 JSON + LangChain logic
                    # 例如 => 有 CSV 或完全沒上傳檔案
                    if st.session_state.uploaded_image_path and st.session_state.image_base64:
                        # [情境] 有上傳圖片 + (也許CSV?? or not??) => 舊邏輯
                        prompt = f"User input: {user_input}\nHere is the image data in base64:\n{st.session_state.image_base64}..."
                    else:
                        if st.session_state.uploaded_file_path is not None:
                            try:
                                df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                                csv_columns = ", ".join(df_temp.columns)
                            except Exception as e:
                                csv_columns = "無法讀取欄位"
                                if st.session_state.debug_mode:
                                    st.error(f"Error reading columns: {e}")
                                debug_log(f"[DEBUG] Error reading columns: {e}")
                        else:
                            csv_columns = "無上傳檔案"

                        prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "這是我的觀察：{{{{分析內容}}}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# 讀取 CSV 檔案 (請直接使用 st.session_state.uploaded_file_path 變數)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# 在這裡加入你要的繪圖或分析邏輯\\n\\n# 例如使用 st.pyplot() 來顯示圖表:\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)\\n"
}}
Important:
1) Must use st.session_state.uploaded_file_path as the CSV path (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)

Based on the request: {user_input}.
Available columns: {csv_columns}.
"""
                        if csv_columns == "無上傳檔案":
                            prompt = f"請全部以繁體中文回答此問題：{user_input}"

                    debug_log(f"DEBUG: Prompt => {prompt}")

                    # 用LangChain conversation
                    raw_response = st.session_state.conversation.run(prompt)
                    if st.session_state.debug_mode:
                        st.write("Model raw response:", raw_response)
                    debug_log(f"[DEBUG] Model raw response => {raw_response}")

                    # 嘗試解析 JSON
                    json_str = extract_json_block(raw_response)
                    try:
                        response_json = json.loads(json_str)
                    except Exception as e:
                        debug_log(f"json.loads parsing error: {e}")
                        debug_error(f"json.loads parsing error: {e}")
                        response_json = {"content": json_str, "code": ""}

                    content = response_json.get("content", "這是我的分析：")
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    with st.chat_message("assistant"):
                        st.write(content)

                    code = response_json.get("code", "")
                    if code:
                        st.session_state.messages.append({"role": "assistant", "code": code})
                        with st.chat_message("assistant"):
                            st.code(code, language="python")
                        st.session_state.ace_code = code

                    # 若深度分析 => 執行並再送 GPT-4o 做二階/三階回覆
                    if st.session_state.deep_analysis_mode and code:
                        st.write("### [深度分析] 自動執行產生的程式碼並將圖表送至 GPT-4o 解析...")

                        global_vars = {
                            "uploaded_file_path": st.session_state.uploaded_file_path,
                            "uploaded_image_path": st.session_state.uploaded_image_path,
                        }
                        exec_result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                        st.write("#### Execution Result")
                        st.text(exec_result)

                        fig = plt.gcf()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        st.session_state.deep_analysis_image = chart_base64

                        # 第二階段
                        deep_model = ChatOpenAI(
                            model="gpt-4o",
                            temperature=0.5,
                            openai_api_key=api_key
                        ) if api_key else None
                        if deep_model:
                            prompt_2 = f"""
這是一張我從剛才的程式碼中產生的圖表，以下是圖表的base64編碼：
{chart_base64[:300]}...

請你為我進行進一步的分析，解釋這張圖表可能代表什麼樣的數據趨勢或觀察。
"""
                            debug_log(f"DEBUG: Deep Analysis Prompt => {prompt_2}")
                            second_raw_response = deep_model.call_as_llm(prompt_2)
                            st.session_state.second_response = second_raw_response

                            st.write("#### [深度分析] 圖表解析結果 (第二次回覆) :")
                            st.write(second_raw_response)

                            # 第三階段
                            final_model = ChatOpenAI(
                                model="gpt-4o",
                                temperature=0.5,
                                openai_api_key=api_key
                            ) if api_key else None
                            if final_model:
                                prompt_3 = f"""
第一階段回覆內容：{content}
第二階段圖表解析內容：{second_raw_response}

請你幫我把以上兩階段的內容好好做一個文字總結，並提供額外的建議或見解。
"""
                                debug_log(f"DEBUG: Final Summary Prompt => {prompt_3}")
                                third_raw_response = final_model.call_as_llm(prompt_3)
                                st.session_state.third_response = third_raw_response

                                st.write("#### [深度分析] 結論 (第三次回覆) :")
                                st.write(third_raw_response)

                                st.write("#### [深度分析] 圖表：")
                                img_data = base64.b64decode(st.session_state.deep_analysis_image)
                                st.image(img_data, caption="深度分析產生的圖表", use_column_width=True)

            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"An error occurred: {e}")
                debug_log(f"[DEBUG] An error occurred: {e}")

    # --- Debug logs ---
    debug_log(f"DEBUG: editor_location = {st.session_state.editor_location}")
    debug_log(f"DEBUG: final st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
    debug_log(f"DEBUG: final st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

    # --- 顯示 Code Editor ---
    if st.session_state.editor_location == "Main":
        with st.expander("🖋️ Persistent Code Editor (Main)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_main"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code

            if st.button("▶️ Execute Code", key="execute_code_main"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"DEBUG: executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)
    else:
        with st.sidebar.expander("🖋️ Persistent Code Editor (Sidebar)", expanded=False):
            edited_code = st_ace(
                value=st.session_state.ace_code,
                language="python",
                theme="monokai",
                height=300,
                key="persistent_editor_sidebar"
            )
            if edited_code != st.session_state.ace_code:
                st.session_state.ace_code = edited_code

            if st.button("▶️ Execute Code", key="execute_code_sidebar"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"DEBUG: executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"DEBUG: executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)

# --- 程式入口 ---
if __name__ == "__main__":
    main()
