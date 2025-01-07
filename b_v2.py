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
from openai import OpenAI  # 使用自定義的 OpenAI 客戶端
from streamlit_ace import st_ace

# --- Initialize and Settings ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4o",  # 假設可解析圖片的實驗模型
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k"
]

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.write(msg)
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.error(msg)
        print(msg)

def initialize_client(api_key, model_name):
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        model=model_name,
        temperature=0.5
    )

def stream_llm_response(client, model_params): 
    """Stream responses from the LLM model."""
    try:
        for chunk in client.chat.completions.create(
                model=model_params.get("model", "gpt-4o"),
                messages=st.session_state.messages,
                temperature=model_params.get("temperature", 0.3),
                max_tokens=4096,
                stream=True):
            chunk_text = chunk.choices[0].delta.content or ""
            yield chunk_text
    except Exception as e:
        debug_error(f"API request failed: {e}")
        yield ""

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    debug_log(f"DEBUG: saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"DEBUG: files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def execute_code(code, global_vars=None):
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
    pattern = r'```(?:json)?(.*)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json_str
    else:
        return response.strip()

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (With Debug & Deep Analysis)")

    # Initialize session state variables
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
        api_key = st.text_input("OpenAI API Key", type="password")

        selected_model = st.selectbox("選擇模型:", OPENAI_MODELS, index=0)

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("深度分析模式", value=False)

        if "client" not in st.session_state:
            if api_key:
                st.session_state.client = initialize_client(api_key, selected_model)
                debug_log(f"DEBUG: Initialized OpenAI client with model {selected_model}")
            else:
                st.warning("⬅️ 請輸入 API Key 以初始化聊天機器人。")

        if st.session_state.debug_mode and "client" in st.session_state:
            debug_log(f"DEBUG: Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory"):
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

        st.subheader("🧠 Memory State")
        if "messages" in st.session_state:
            memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.text_area("Current Memory", value=memory_content, height=200)

        # --- CSV 上傳 ---
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

        # --- 圖片上傳 ---
        st.subheader("🖼️ Upload an Image")
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

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location

    # --- 顯示歷史訊息 ---
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                if "content" in message:
                    st.write(message["content"])
                if "code" in message:
                    st.code(message["code"], language="python")

    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        if "client" not in st.session_state or st.session_state.client is None:
            st.error("OpenAI client is not initialized. Please enter a valid API Key.")
        else:
            with st.spinner("Thinking..."):
                try:
                    debug_log(f"DEBUG: Currently st.session_state.uploaded_file_path = {st.session_state.uploaded_file_path}")
                    debug_log(f"DEBUG: Currently st.session_state.uploaded_image_path = {st.session_state.uploaded_image_path}")

                    # --- 決定使用哪種 prompt ---
                    if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                        # [情境] 有上傳圖片 -> 只給 user_input 與 圖片 Base64
                        # 避免將 CSV & JSON 格式也混進去
                        prompt = f"User input: {user_input}\nHere is the image data in base64:\n{st.session_state.image_base64[:300]}..."
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

                        if st.session_state.uploaded_file_path is not None and csv_columns != "無上傳檔案":
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
                        else:
                            prompt = f"請全部以繁體中文回答此問題：{user_input}"

                    debug_log(f"DEBUG: Prompt used => {prompt}")

                    # Append the prompt to messages
                    st.session_state.messages.append({"role": "system", "content": prompt})

                    # Get the model parameters
                    model_params = {
                        "model": st.session_state.client.model,
                        "temperature": st.session_state.client.temperature
                    }

                    # Stream the response
                    response_content = ""
                    response_code = ""
                    for chunk in stream_llm_response(st.session_state.client, model_params):
                        response_content += chunk
                        st.chat_message("assistant").write(chunk)
                        st.experimental_rerun()  # To display the streaming content

                    # After streaming is complete, process the full response
                    json_str = extract_json_block(response_content)
                    try:
                        response_json = json.loads(json_str)
                    except Exception as e:
                        debug_log(f"json.loads parsing error: {e}")
                        debug_error(f"json.loads parsing error: {e}")
                        response_json = {"content": response_content, "code": ""}

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

                        deep_model = initialize_client(st.session_state.client.api_key, "gpt-4o")
                        if deep_model:
                            prompt_2 = f"""
這是一張我從剛才的程式碼中產生的圖表，以下是圖表的base64編碼：
{chart_base64[:300]}...

請你為我進行進一步的分析，解釋這張圖表可能代表什麼樣的數據趨勢或觀察。
"""
                            debug_log(f"DEBUG: Deep Analysis Prompt => {prompt_2}")
                            st.session_state.messages.append({"role": "system", "content": prompt_2})

                            # Stream the second response
                            second_raw_response = ""
                            for chunk in stream_llm_response(deep_model, {"model": "gpt-4o", "temperature": 0.5}):
                                second_raw_response += chunk
                                st.chat_message("assistant").write(chunk)
                                st.experimental_rerun()

                            st.session_state.second_response = second_raw_response

                            st.write("#### [深度分析] 圖表解析結果 (第二次回覆) :")
                            st.write(second_raw_response)

                            final_model = initialize_client(st.session_state.client.api_key, "gpt-4o")
                            if final_model:
                                prompt_3 = f"""
第一階段回覆內容：{content}
第二階段圖表解析內容：{second_raw_response}

請你幫我把以上兩階段的內容好好做一個文字總結，並提供額外的建議或見解。
"""
                                debug_log(f"DEBUG: Final Summary Prompt => {prompt_3}")
                                st.session_state.messages.append({"role": "system", "content": prompt_3})

                                # Stream the third response
                                third_raw_response = ""
                                for chunk in stream_llm_response(final_model, {"model": "gpt-4o", "temperature": 0.5}):
                                    third_raw_response += chunk
                                    st.chat_message("assistant").write(chunk)
                                    st.experimental_rerun()

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

if __name__ == "__main__":
    main()
