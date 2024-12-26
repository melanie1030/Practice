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

# --------------- (來自4o_image_handle的import) ---------------
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import random

# --------------- (LangChain相關) ---------------
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# --- dotenv 初始化 ---
dotenv.load_dotenv()

# === 全域設定 ===
UPLOAD_DIR = "uploaded_files"
OPENAI_MODELS = ["gpt-4o"]  # 以4o作為可選模型

# -------------------------------------------
# 以下區塊為 b_v2.py 原有的功能 & 函式
# -------------------------------------------

def debug_log(msg):
    """若 debug_mode=True 則在畫面與console印出提示"""
    if st.session_state.get("debug_mode", False):
        st.write(msg)
        print(msg)

def debug_error(msg):
    """若 debug_mode=True 則在畫面與console印出錯誤訊息"""
    if st.session_state.get("debug_mode", False):
        st.error(msg)
        print(msg)

def initialize_client(api_key, model_name):
    """
    由 b_v2 原先使用的 LangChain ChatOpenAI。
    不要移除此函式，以兼容原有對話流程 (ConversationChain)。
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

    debug_log(f"DEBUG: files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
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

# -------------------------------------------
# 以下區塊為 4o_image_handle.py 主要功能 & 函式
# -------------------------------------------

import openai  # 注意：若無法使用 `from openai import OpenAI`，可直接 `import openai`
               # 以下以 openai 官方套件為例，如有客製 OpenAI(api_key=...) 物件時，需自行改寫

def initialize_openai_client(api_key):
    """
    由 4o_image_handle.py 中的 initialize_client(api_key) 改寫而來。
    不移除。此處直接給 openai.api_key 。
    """
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = None

def load_image_base64(image: Image.Image):
    """將 PIL Image 轉換為 base64 字串"""
    buffer = io.BytesIO()
    # 若不確定 image.format，可視情況手動指定，如 PNG
    image.save(buffer, format=image.format if image.format else "PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def add_user_image(image):
    """
    在 4o_image_handle.py 中，會把圖像加到 session_state.messages 供 Chat 使用。
    但 b_v2.py 與此邏輯不同。此處先保留以便在合併後可參考。
    """
    img_base64 = load_image_base64(image)
    if "stream_messages" not in st.session_state:
        st.session_state.stream_messages = []
    st.session_state.stream_messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            }
        ]
    })

def reset_session_messages():
    """清除對話訊息 (針對 4o_image_handle 的對話流)"""
    if "stream_messages" in st.session_state:
        st.session_state.pop("stream_messages")

def stream_llm_response(messages, model="gpt-4o", temperature=0.3):
    """
    參考 4o_image_handle.py：使用 openai.ChatCompletion.create() + stream=True
    進行串流輸出。
    這與 b_v2.py 用 LangChain ChatOpenAI 執行對話不同，是兩套機制。
    """
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=True
        )
        for chunk in completion:
            chunk_text = chunk["choices"][0].get("delta", {}).get("content", "")
            if chunk_text:
                yield chunk_text
    except Exception as e:
        yield f"【串流發生錯誤】{str(e)}"

# -------------------------------------------
# 上述兩大區塊為 b_v2.py & 4o_image_handle.py 的函式庫
# 以下開始撰寫一個「合併後的 main()」，同時保留 b_v2.py 原本的主介面
# 並增添 4o_image_handle.py 的功能 (如串流、拍照、錄音等)。
# -------------------------------------------

def main():
    # === 依 b_v2.py 設置首頁基本配置 ===
    st.set_page_config(page_title="Chatbot + Data Analysis + 4o Image+Audio", 
                       page_icon="🤖", 
                       layout="wide")

    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas + (4o Streaming / Image / Audio)")

    # -------------- 初始化 session_state 變數 --------------
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
    
    # 4o_image_handle 的 stream_messages
    if "stream_messages" not in st.session_state:
        st.session_state.stream_messages = []

    # --- 側邊欄 ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        # 4o_image_handle & b_v2 皆需要 API Key
        # initialize_openai_client 先設定 openai.api_key
        initialize_openai_client(api_key)

        # b_v2.py 用於 langchain ChatOpenAI
        selected_model = st.selectbox("選擇模型 (LangChain):", OPENAI_MODELS, index=0)

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("深度分析模式", value=False)

        # 若尚未有 conversation (LangChain)
        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = initialize_client(api_key, selected_model)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ 請輸入 API Key 以初始化聊天機器人。")

        if st.session_state.debug_mode:
            debug_log(f"DEBUG: Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory (b_v2)"):
            # 只清除 b_v2 的 Memory
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
            st.success("Memory cleared for b_v2 main chat!")
        
        # 4o_image_handle 的 stream_messages 清除
        if st.button("🗑️ Clear Stream Messages (4o)"):
            reset_session_messages()
            st.success("Stream messages cleared for 4o approach!")

        st.subheader("🧠 Memory State (b_v2)")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory (LangChain)", value=str(memory_content), height=200)

        # === CSV 上傳功能 (b_v2) ===
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
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

        # === 圖片上傳 (b_v2) ===
        st.subheader("🖼️ Upload an Image (b_v2)")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
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

        # === 4o_image_handle.py 相關：圖像上傳 & 相機拍照 & 音訊錄製 ===
        st.subheader("4o: 上傳圖像 / 拍照 / 錄音（串流回應）")

        # 相機拍照
        camera_img = st.camera_input("拍照")
        if camera_img:
            cam_image = Image.open(camera_img)
            add_user_image(cam_image)
            st.success("拍照已成功並加到 stream_messages!")

        # 圖像上傳 (獨立於 b_v2 原本的)
        uploaded_img_4o = st.file_uploader("4o 追加圖片:", type=["png", "jpg", "jpeg"], key="4o_uploader")
        if uploaded_img_4o:
            up_image_4o = Image.open(uploaded_img_4o)
            add_user_image(up_image_4o)
            st.success("圖像已上傳並加到 stream_messages!")

        # 音訊錄製
        st.write("### 錄製音訊並上傳 (demo)")
        audio_data = audio_recorder()
        if audio_data is not None:
            st.success("錄音完成！(目前未自動加入對話，如需可自行客製化處理)")
            # 這邊可自行決定要如何把音訊資料送到 GPT-4o
            # 例如先加到 session_state.stream_messages，或透過 openai 的語音API 轉文字。
            # 這邊僅示範顯示已錄音。

        st.subheader("Editor Location (b_v2)")  # 保留 b_v2 既有的 Editor Location
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location

    # ----------------
    # (b_v2) 顯示「主要對話記錄」：st.session_state.messages
    # ----------------
    st.write("## b_v2 主對話區 (LangChain ConversationChain)")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "code" in message:
                st.code(message["code"], language="python")

    # --- b_v2 的用戶輸入 ---
    user_input = st.chat_input("b_v2: Hi! Ask me anything (LangChain conversation)...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            if not api_key:
                st.warning("尚未填入 API Key，無法使用 b_v2 的對話功能。")
            else:
                try:
                    debug_log(f"DEBUG: Currently st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
                    debug_log(f"DEBUG: Currently st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

                    # 準備要丟給 GPT-4 (LangChain) 的 prompt
                    if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                        # [情境] 有上傳圖片 -> 只給 user_input + 圖片 base64
                        prompt = f"User input: {user_input}\nHere is the image data in base64:\n{st.session_state.image_base64}..."
                    else:
                        # [情境] 沒有上傳圖片 -> 維持舊有複雜 JSON 邏輯
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

                    debug_log(f"DEBUG: Prompt used => {prompt}")

                    raw_response = st.session_state.conversation.run(prompt)
                    if st.session_state.debug_mode:
                        st.write("Model raw response:", raw_response)
                    debug_log(f"[DEBUG] Model raw response => {raw_response}")

                    # 嘗試擷取 JSON
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

                    # --- 若勾選深度分析模式 & 有程式碼 -> 執行程式、二次解析圖表 ---
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

                        # 做第二階段呼叫: Deep Analysis
                        if api_key:
                            deep_model = ChatOpenAI(
                                model="gpt-4o",
                                temperature=0.5,
                                openai_api_key=api_key
                            )
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

                            # 最後總結
                            final_model = ChatOpenAI(
                                model="gpt-4o",
                                temperature=0.5,
                                openai_api_key=api_key
                            )
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

    # ---------------
    # 4o_image_handle: 在 main 頁面中額外展示「stream_messages」區塊
    # ---------------
    st.write("## 4o Streaming Chat 區 (使用 openai.ChatCompletion.stream)")
    # 顯示 4o 的對話 messages
    for idx, msg in enumerate(st.session_state.stream_messages):
        role = msg["role"]
        contents = msg["content"]
        with st.chat_message(role):
            for c in contents:
                if c["type"] == "text":
                    st.write(c.get("text", ""))
                elif c["type"] == "image_url":
                    st.image(c["image_url"].get("url", ""), caption=f"Image in message {idx}")

    # 4o: 用戶輸入
    user_input_4o = st.chat_input("4o: Hi! Ask me anything (Streaming approach)...")
    if user_input_4o:
        st.session_state.stream_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input_4o}]
        })
        with st.chat_message("user"):
            st.write(user_input_4o)

        if not api_key:
            st.warning("尚未填入 API Key，無法使用 4o 串流功能。")
        else:
            # 串流回應
            with st.chat_message("assistant"):
                # 建立一個空的容器，逐行寫入串流結果
                stream_placeholder = st.empty()
                full_response = ""

                # 準備 openai 的 messages
                # 轉為 openai.ChatCompletion.create() 能理解的格式
                openai_messages = []
                for m in st.session_state.stream_messages:
                    # 4o_image_handle 裡的 msg["content"] 可能是多筆 text / image
                    # openai 需要 "role", "content" (都是 text)
                    # 這裡簡化做法：將多個 content 只合成一個 string
                    msg_role = m["role"]
                    msg_text = []
                    for cobj in m["content"]:
                        if cobj["type"] == "text":
                            msg_text.append(cobj["text"])
                        elif cobj["type"] == "image_url":
                            # 直接將 base64 當成文字描述
                            # （真實應用中可整合Vision API或特殊處理）
                            msg_text.append(f"[image_url: {cobj['image_url']['url']}]")
                    combined_text = "\n".join(msg_text)
                    openai_messages.append({"role": msg_role, "content": combined_text})

                for chunk_text in stream_llm_response(openai_messages, model="gpt-4o", temperature=0.3):
                    full_response += chunk_text
                    stream_placeholder.markdown(full_response)

                # 將完整回應存回 session
                st.session_state.stream_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_response}]
                })

    # ----------------
    # (b_v2) 在主體最後：顯示「Persistent Code Editor」位置
    # ----------------
    from streamlit_ace import st_ace  # b_v2 原本就有
    debug_log(f"DEBUG: editor_location = {st.session_state.editor_location}")
    debug_log(f"DEBUG: final st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
    debug_log(f"DEBUG: final st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

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


# -------------------------------------------
# 執行入口
# -------------------------------------------
if __name__ == "__main__":
    main()
