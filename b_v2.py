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
from openai import OpenAI  # 自定義的 OpenAI 類別
from PIL import Image
from streamlit_ace import st_ace
import openai  # 引入 openai 模組以供內部使用

# --- 定義 OpenAI 類別 ---
class OpenAI:
    def __init__(self, api_key):
        openai.api_key = api_key

    class chat:
        class completions:
            @staticmethod
            def create(model, messages, temperature, max_tokens, stream):
                return openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )

    # 添加 error 屬性以便捕捉異常
    class error:
        RateLimitError = openai.error.RateLimitError
        OpenAIError = openai.error.OpenAIError

# --- 初始化與設置 ---
dotenv.load_dotenv()

UPLOAD_DIR = "uploaded_files"

OPENAI_MODELS = [
    "gpt-4-turbo",  # 使用更穩定的模型
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o"
]

MAX_MESSAGES = 10  # 限制訊息歷史

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.write(f"**DEBUG LOG:** {msg}")
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.error(f"**DEBUG ERROR:** {msg}")
        print(msg)

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image, max_size=(800, 800)):
    """Load image, resize if necessary, and convert to base64."""
    try:
        # 重新調整圖片大小以減少大小
        image.thumbnail(max_size, Image.LANCZOS)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        debug_log("Image has been resized and converted to base64.")
        debug_log(f"Image base64 (first 100 chars): {img_base64[:100]}...")
        return img_base64
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message(role, content):
    """Append a message and ensure the total number of messages does not exceed MAX_MESSAGES."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # 移除最舊的訊息（除了 system prompt）
        # 假設 system prompt 是第一條訊息
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES-1):]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(image):
    """Add an image message to the session state."""
    img_base64 = load_image_base64(image)
    if img_base64:
        image_message = f"Here is the image you uploaded:\n![Uploaded Image](data:image/png;base64,{img_base64})"
        append_message("user", image_message)
        st.success("圖像已上傳!")
    else:
        debug_error("Failed to convert image to base64.")

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.code(code, language="python")

        debug_log(f"Executing code with global_vars: {list(exec_globals.keys())}")
        exec(code, exec_globals)
        output = exec_globals.get("output", "(No output returned)")
        debug_log(f"Execution output: {output}")
        return f"Code executed successfully. Output: {output}"
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log(f"Execution error: {error_msg}")
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode)."

def extract_json_block(response: str) -> str:
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

def stream_llm_response(client, model_params):
    """Stream responses from the LLM model."""
    try:
        response = client.chat.completions.create(
            model=model_params.get("model", "gpt-4-turbo"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=model_params.get("max_tokens", 4096),
            stream=True
        )
        response_content = ""
        assistant_placeholder = st.empty()  # Create a placeholder for assistant's response

        for chunk in response:
            chunk_text = chunk.choices[0].delta.get('content', '')
            if chunk_text:
                response_content += chunk_text
                # Update the assistant's response in the placeholder
                assistant_placeholder.markdown(response_content)
                debug_log(f"Received chunk: {chunk_text[:100]}...")
        return response_content
    except OpenAI.error.RateLimitError as e:
        debug_error(f"Rate limit exceeded: {e}")
        st.error("Rate limit exceeded. Please try again later.")
        return ""
    except OpenAI.error.OpenAIError as e:
        debug_error(f"OpenAI error: {e}")
        st.error(f"An OpenAI error occurred: {e}")
        return ""
    except Exception as e:
        debug_error(f"Error streaming response: {e}")
        st.error(f"An error occurred while streaming the response: {e}")
        return ""

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

        if "memory" not in st.session_state:
            st.session_state.memory = []

        if "conversation_initialized" not in st.session_state:
            if api_key:
                # Initialize OpenAI client
                client = OpenAI(api_key)
                st.session_state.conversation_initialized = True
                st.session_state.messages = []  # Initialize with empty message history
                debug_log("Conversation initialized with empty message history.")
            else:
                st.warning("⬅️ 請輸入 API Key 以初始化聊天機器人。")

        if st.session_state.debug_mode:
            debug_log(f"Currently using model => {selected_model}")

        if st.button("🗑️ Clear Memory"):
            st.session_state.memory = []
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
            debug_log("Memory has been cleared.")

        st.subheader("🧠 Memory State")
        if st.session_state.messages:
            memory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.text_area("Current Memory", value=memory_content, height=200)
            debug_log(f"Current memory content: {memory_content}")
        else:
            st.text_area("Current Memory", value="No messages yet.", height=200)
            debug_log("No messages in memory.")

        # --- CSV 上傳 ---
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview")
                st.dataframe(csv_data)
                debug_log(f"CSV Data Columns: {list(csv_data.columns)}")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading CSV: {e}")
                debug_log(f"Error reading CSV: {e}")

        # --- 圖片上傳 ---
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("選擇一張圖片:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            add_user_image(Image.open(uploaded_image))

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location
        debug_log(f"Editor location set to: {st.session_state.editor_location}")

    # --- 顯示歷史訊息 ---
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict) and "code" in message["content"]:
                st.code(message["content"]["code"], language="python")
                debug_log(f"Displaying code from {message['role']}: {message['content']['code']}")
            else:
                st.write(message["content"])
                debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")

    # --- 用戶輸入 ---
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        append_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)
            debug_log(f"User input added to messages: {user_input}")

        with st.spinner("Thinking..."):
            try:
                # Initialize OpenAI client if not already done
                if api_key:
                    client = OpenAI(api_key)
                else:
                    raise ValueError("OpenAI API Key is not provided.")

                debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
                debug_log(f"Uploaded image path: {st.session_state.uploaded_image_path}")

                # --- 確保 system prompt 僅添加一次 ---
                if not any(msg["role"] == "system" for msg in st.session_state.messages):
                    system_prompt = "你是一個協助進行數據分析的助手。"
                    append_message("system", system_prompt)
                    debug_log("System prompt added to messages.")

                # --- 決定使用哪種 prompt ---
                if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                    # 圖片已上傳，圖片數據已作為單獨的訊息添加
                    prompt = user_input  # 直接使用用戶輸入
                    debug_log("User input with image data already appended.")
                else:
                    # 沒有圖片上傳，使用複雜的 JSON 邏輯
                    if st.session_state.uploaded_file_path is not None:
                        try:
                            df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                            csv_columns = ", ".join(df_temp.columns)
                            debug_log(f"CSV columns: {csv_columns}")
                        except Exception as e:
                            csv_columns = "無法讀取欄位"
                            if st.session_state.debug_mode:
                                st.error(f"Error reading columns: {e}")
                            debug_log(f"Error reading columns: {e}")
                    else:
                        csv_columns = "無上傳檔案"
                        debug_log("No CSV file uploaded.")

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
                        debug_log("Prompt constructed for CSV input with JSON response.")
                        append_message("system", prompt)
                        debug_log("System prompt appended to messages.")
                    else:
                        prompt = f"請全部以繁體中文回答此問題：{user_input}"
                        debug_log("Prompt constructed for plain text input.")
                        append_message("system", prompt)
                        debug_log("Plain text system prompt appended to messages.")

                # Make the API request and stream the response
                model_params = {
                    "model": selected_model,
                    "temperature": 0.5,
                    "max_tokens": 4096
                }
                response_content = stream_llm_response(client, model_params)
                debug_log(f"Full assistant response: {response_content}")

                if response_content:
                    # After streaming is done, append assistant message
                    append_message("assistant", response_content)
                    with st.chat_message("assistant"):
                        st.write(response_content)
                        debug_log(f"Assistant response added to messages: {response_content}")

                    # Extract JSON and code
                    json_str = extract_json_block(response_content)
                    try:
                        response_json = json.loads(json_str)
                        debug_log("JSON parsing successful.")
                    except Exception as e:
                        debug_log(f"json.loads parsing error: {e}")
                        debug_error(f"json.loads parsing error: {e}")
                        response_json = {"content": json_str, "code": ""}
                        debug_log("Fallback to raw response for content.")

                    content = response_json.get("content", "這是我的分析：")
                    append_message("assistant", content)
                    with st.chat_message("assistant"):
                        st.write(content)
                        debug_log(f"Content from JSON appended to messages: {content}")

                    code = response_json.get("code", "")
                    if code:
                        # 將代碼作為獨立的訊息添加
                        append_message("assistant", {"code": code})
                        with st.chat_message("assistant"):
                            st.code(code, language="python")
                            debug_log(f"Code from JSON appended to messages: {code}")
                        st.session_state.ace_code = code
                        debug_log("ace_code updated with new code.")

                    # --- 若勾選深度分析模式 & 有程式碼 -> 執行程式、二次解析圖表 ---
                    if st.session_state.deep_analysis_mode and code:
                        st.write("### [深度分析] 自動執行產生的程式碼並將圖表送至 GPT-4o 解析...")
                        debug_log("Deep analysis mode activated.")

                        global_vars = {
                            "uploaded_file_path": st.session_state.uploaded_file_path,
                            "uploaded_image_path": st.session_state.uploaded_image_path,
                        }
                        exec_result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                        st.write("#### Execution Result")
                        st.text(exec_result)
                        debug_log(f"Execution result: {exec_result}")

                        fig = plt.gcf()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        st.session_state.deep_analysis_image = chart_base64
                        debug_log("Chart has been converted to base64.")

                        # Prepare deep analysis prompt
                        prompt_2 = f"""
這是一張我從剛才的程式碼中產生的圖表，以下是圖表的base64編碼：
![image](data:image/png;base64,{chart_base64})

請你為我進行進一步的分析，解釋這張圖表可能代表什麼樣的數據趨勢或觀察。
"""
                        debug_log(f"Deep Analysis Prompt: {prompt_2}")

                        # Append prompt_2 to messages
                        append_message("user", prompt_2)
                        debug_log("Deep analysis prompt appended to messages.")

                        # Make the API request for deep analysis
                        second_raw_response = stream_llm_response(client, model_params)
                        debug_log(f"Deep analysis response: {second_raw_response}")

                        if second_raw_response:
                            # Append assistant response
                            append_message("assistant", second_raw_response)
                            st.session_state.second_response = second_raw_response
                            with st.chat_message("assistant"):
                                st.write(second_raw_response)
                                debug_log(f"Deep analysis response added to messages: {second_raw_response}")

                            # Prepare final summary prompt
                            prompt_3 = f"""
第一階段回覆內容：{content}
第二階段圖表解析內容：{second_raw_response}

請你幫我把以上兩階段的內容好好做一個文字總結，並提供額外的建議或見解。
"""
                            debug_log(f"Final Summary Prompt: {prompt_3}")

                            # Append prompt_3 to messages
                            append_message("user", prompt_3)
                            debug_log("Final summary prompt appended to messages.")

                            # Make the API request for final summary
                            third_raw_response = stream_llm_response(client, model_params)
                            debug_log(f"Final summary response: {third_raw_response}")

                            if third_raw_response:
                                # Append assistant response
                                append_message("assistant", third_raw_response)
                                st.session_state.third_response = third_raw_response
                                with st.chat_message("assistant"):
                                    st.write(third_raw_response)
                                    debug_log(f"Final summary response added to messages: {third_raw_response}")

                                # Display the chart
                                st.write("#### [深度分析] 圖表：")
                                try:
                                    img_data = base64.b64decode(st.session_state.deep_analysis_image)
                                    st.image(img_data, caption="深度分析產生的圖表", use_column_width=True)
                                    debug_log("Deep analysis chart displayed.")
                                except Exception as e:
                                    if st.session_state.debug_mode:
                                        st.error(f"Error displaying chart: {e}")
                                    debug_log(f"Error displaying chart: {e}")

            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"An error occurred: {e}")
                debug_log(f"An error occurred: {e}")

    # --- Persistent Code Editor ---
    if "ace_code" not in st.session_state:
        st.session_state.ace_code = ""

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
                debug_log("ace_code updated from main editor.")

            if st.button("▶️ Execute Code", key="execute_code_main"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)
                debug_log(f"Code execution result: {result}")

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
                debug_log("ace_code updated from sidebar editor.")

            if st.button("▶️ Execute Code", key="execute_code_sidebar"):
                global_vars = {
                    "uploaded_file_path": st.session_state.uploaded_file_path,
                    "uploaded_image_path": st.session_state.uploaded_image_path,
                }
                debug_log(f"Executing code with uploaded_file_path = {st.session_state.uploaded_file_path}")
                debug_log(f"Executing code with uploaded_image_path = {st.session_state.uploaded_image_path}")

                result = execute_code(st.session_state.ace_code, global_vars=global_vars)
                st.write("### Execution Result")
                st.text(result)
                debug_log(f"Code execution result: {result}")

if __name__ == "__main__":
    main()
