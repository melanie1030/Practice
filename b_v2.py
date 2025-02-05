import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import google.generativeai as genai  # 新增Gemini依赖
from streamlit_ace import st_ace
import time

# --- 初始化设置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
LLM_MODELS = [  # 修改后的模型列表
    "gpt-4-turbo",
    "gpt-3.5-turbo-16k",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

MAX_MESSAGES = 10  # Limit message history

def initialize_client(api_key):
    return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        print(msg)

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
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

def load_image_base64(image):
    """Convert image to Base64 encoding."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # Use PNG for consistency
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message(role, content):
    """Append a message and ensure the total number of messages does not exceed MAX_MESSAGES."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Remove the oldest messages except the system prompt
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES - 1):]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(uploaded_file):
    """Add an image message to the session state using image_url structure and save the file."""
    try:
        # 打開上傳的圖片
        image = Image.open(uploaded_file)
        img_base64 = load_image_base64(image)
        if img_base64:
            # 創建 image_url 結構
            image_content = [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            }]
            append_message("user", image_content)  # 將圖片訊息添加到訊息歷史
            st.session_state.image_base64 = img_base64  # 更新 image_base64
            st.session_state.uploaded_image_path = save_uploaded_file(uploaded_file)  # 保存圖片檔案路徑
            st.success("圖片上傳成功！")
            debug_log("Image uploaded and added to messages.")
        else:
            debug_error("無法將圖片轉換為 base64。")
    except Exception as e:
        debug_error(f"處理上傳圖片時出錯: {e}")

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
        st.success("Memory cleared!")
        debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.session_state.debug_logs.append(f"```python\n{code}\n```")

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

def to_markdown(text: str) -> str:
    """
    将Gemini响应文本转换为兼容Streamlit的Markdown格式
    功能：
    1. 自动检测代码块并添加正确语法
    2. 转换项目符号列表
    3. 处理引文标注
    4. 保留原始换行格式
    """
    import re
    
    # 预处理：保留换行符
    text = text.replace('\n', '<br>')
    
    # 转换代码块
    text = re.sub(
        r'``[(\w+)?\s*(.*?)](cci:1://file:///d:/Practice-main/b_v2.py:295:0-754:61)``', 
        lambda m: f"```{m.group(1) or ''}\n{m.group(2)}\n```", 
        text, 
        flags=re.DOTALL
    )
    
    # 转换项目符号列表
    text = re.sub(r'•', '  *', text)
    
    # 转换编号列表
    text = re.sub(r'(\d+)\.', r'\1\\.', text)
    
    # 处理引文标注（如：^[1]）
    text = re.sub(r'\^\[(\d+)\]', r'<sup>\1</sup>', text)
    
    # 恢复换行符
    text = text.replace('<br>', '\n')
    
    # 添加Markdown引用格式
    lines = []
    for line in text.split('\n'):
        if line.strip().startswith('*') or line.strip().startswith('#'):
            lines.append(line)
        else:
            lines.append(f"> {line}")
    
    return '\n'.join(lines)
    
# =======================================================================================================
#       以下為舊版本的 get_llm_response 函數
# def get_llm_response(client, model_params, max_retries=3):
#     """Get response from the LLM model synchronously with retry logic."""
#     retries = 0
#     wait_time = 5  # Start with 5 seconds

#     while retries < max_retries:
#         try:
#             response = client.chat.completions.create(
#                 model=model_params.get("model", "gpt-4-turbo"),
#                 messages=st.session_state.messages,
#                 temperature=model_params.get("temperature", 0.3),
#                 max_tokens=model_params.get("max_tokens", 4096),
#                 stream=False  # Disable streaming
#             )
#             # Extract the full response content
#             response_content = response.choices[0].message.content.strip()
#             debug_log(f"Full assistant response: {response_content}")
#             return response_content

#         except Exception as e:
#             if 'rate_limit_exceeded' in str(e).lower() or '429' in str(e):
#                 debug_error(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
#                 st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
#                 time.sleep(wait_time)
#                 retries += 1
#                 wait_time *= 2  # Exponential backoff
#             else:
#                 debug_error(f"Error getting response: {e}")
#                 st.error(f"An error occurred while getting the response: {e}")
#                 return ""

#     st.error("Max retries exceeded. Please try again later.")
#     return ""
# =======================================================================================================


# 以下為新版本的 get_llm_response 函數
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_gemini_response(model_params, max_retries=3):
    """整合新版 Gemini 請求方法"""
    # 從環境變數獲取 API 金鑰 (保持原有設定方式)
    api_key = st.session_state.get("gemini_api_key_input")
    if not api_key:
        st.error("未設定 Gemini API 金鑰")
        return ""
    
    # 初始化 Gemini 模型
    genai.configure(api_key=api_key)
    model_name = model_params.get("model", "gemini-1.5-flash")
    
    # 初始化會話 (依用戶提供的程式碼結構)
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = genai.GenerativeModel(model_name)
        st.session_state.gemini_history = []
    
    # 轉換歷史訊息格式
    converted_history = []
    for msg in st.session_state.messages:
        role = map_role(msg["role"])
        parts = []
        
        # 處理多模態內容
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item["type"] == "image_url":
                    parts.append(Part(
                        inline_data={
                            "mime_type": "image/png",
                            "data": item["image_url"]["url"].split(",")[1]
                        }
                    ))
                else:
                    parts.append(Part(text=item))
        else:
            parts.append(Part(text=msg["content"]))
        
        converted_history.append({"role": role, "parts": parts})
    
    # 請求邏輯 (帶重試機制)
    retries = 0
    while retries < max_retries:
        try:
            response = st.session_state.gemini_chat.generate_content(
                contents=converted_history,
                generation_config={
                    "temperature": model_params.get("temperature", 0.3),
                    "max_output_tokens": model_params.get("max_tokens", 4096)
                }
            )
            # 更新歷史記錄 (依用戶程式碼格式)
            st.session_state.gemini_history.extend([
                {"role": "user", "parts": converted_history[-1]["parts"]},
                {"role": "model", "parts": [Part(text=response.text)]}
            ])
            return response.text
            
        except genai.GenerationError as e:
            debug_error(f"生成錯誤: {str(e)}")
            retries += 1
            time.sleep(5 * retries)
        except Exception as e:
            debug_error(f"API請求異常: {str(e)}")
            return ""
    
    return "請求失敗次數過多，請稍後重試"


def get_openai_response(client, model_params, max_retries=3):
    """处理OpenAI API请求"""
    retries = 0
    wait_time = 5  # 初始等待时间5秒
    model_name = model_params.get("model", "gpt-4-turbo")
    
    while retries < max_retries:
        try:
            # 构建请求参数
            request_params = {
                "model": model_name,
                "messages": st.session_state.messages,
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096),
                "stream": False
            }
            
            # 添加图像处理逻辑（如果存在）
            if any(msg.get("content") and isinstance(msg["content"], list) for msg in st.session_state.messages):
                request_params["max_tokens"] = 4096  # 增加token限制
                debug_log("Detected multimodal input, adjusting max_tokens")
            
            # 发送请求
            response = client.chat.completions.create(**request_params)
            
            # 提取并清理响应内容
            response_content = response.choices[0].message.content.strip()
            debug_log(f"OpenAI原始响应：\n{response_content}")
            return response_content
            
        except Exception as e:
            # 处理速率限制错误
            if 'rate limit' in str(e).lower() or '429' in str(e):
                debug_error(f"速率限制错误（尝试 {retries+1}/{max_retries}）：{e}")
                st.warning(f"请求过于频繁，{wait_time}秒后重试...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2  # 指数退避
                
            # 处理认证错误
            elif 'invalid api key' in str(e).lower():
                debug_error(f"API密钥无效：{e}")
                st.error("OpenAI API密钥无效，请检查后重试")
                return ""
                
            # 其他错误处理
            else:
                debug_error(f"OpenAI请求异常：{str(e)}")
                st.error(f"请求发生错误：{str(e)}")
                return ""
    
    # 超过最大重试次数
    debug_error(f"超过最大重试次数（{max_retries}次）")
    st.error("请求失败次数过多，请稍后再试")
    return ""

def get_llm_response(client, model_params, max_retries=3):
    """获取LLM模型响应（支持OpenAI和Gemini）"""
    model_name = model_params.get("model", "gpt-4-turbo")
    
    if "gpt" in model_name:
        return get_openai_response(client, model_params, max_retries)
    elif "gemini" in model_name:
        return get_gemini_response(model_params=model_params, max_retries=max_retries)
    else:
        st.error(f"不支持的模型类型: {model_name}")
        return ""
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state:
        st.session_state.debug_errors = []
    if "thinking_protocol" not in st.session_state:
        st.session_state.thinking_protocol = None  # Initialize thinking_protocol

    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        # 原有的OpenAI金鑰設定
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input("OpenAI API Key", value=default_openai_key, type="password")
        
        # 新增Gemini金鑰輸入
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input("Gemini API Key", 
                                     value=default_gemini_key, 
                                     type="password",
                                     key="gemini_api_key")
        
        # 更新環境變數
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key 

        selected_model = st.selectbox(
            "選擇模型", 
            LLM_MODELS, 
            index=0, 
            key="selected_model"  # 新增key用於session state綁定
        )
        
        # 步驟3：API金鑰狀態檢查 (放在模型選擇之後)
        if "selected_model" in st.session_state:
            current_model = st.session_state.selected_model.lower()
            
            if "gemini" in current_model:
                # 檢查Gemini金鑰 (環境變數或手動輸入)
                gemini_key = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_api_key")
                if not gemini_key:
                    st.error("使用Gemini模型需在下方輸入API金鑰 🔑")
                    st.stop()  # 阻止後續代碼執行
                    
            elif "gpt" in current_model:
                # 檢查OpenAI金鑰
                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
                if not openai_key:
                    st.error("使用OpenAI模型需在下方輸入API金鑰 🔑")
                    st.stop()

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode", value=False)

        if "memory" not in st.session_state:
            st.session_state.memory = []

        if "conversation_initialized" not in st.session_state:
            openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key_input")
            if openai_api_key or gemini_api_key:
                client = initialize_client(openai_api_key)
                st.session_state.conversation_initialized = True
                st.session_state.messages = []
                debug_log("Conversation initialized with OpenAI client.")
            else:
                st.warning("⬅️ 請在側邊欄輸入OpenAI API金鑰以初始化聊天機器人")

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
            st.session_state.debug_logs = []
            st.session_state.debug_errors = []
            st.session_state.thinking_protocol = None  # Clear thinking_protocol
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

        # --- CSV Upload ---
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

        # --- Image Upload ---
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_image:
            add_user_image(uploaded_image)

        # --- Thinking Protocol Upload ---
        st.subheader("🧠 Upload Thinking Protocol")
        uploaded_thinking_protocol = st.file_uploader("Choose a thinking_protocol.md file:", type=["md"], key="thinking_protocol_uploader")
        if uploaded_thinking_protocol:
            try:
                thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
                st.session_state.thinking_protocol = thinking_protocol_content
                append_message("user", thinking_protocol_content)  # 添加为用户消息
                st.success("Thinking Protocol uploaded successfully!")
                debug_log("Thinking Protocol uploaded and added to messages.")
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error reading Thinking Protocol: {e}")
                debug_log(f"Error reading Thinking Protocol: {e}")

        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=0 if st.session_state.editor_location == "Main" else 1
        )
        st.session_state.editor_location = location
        debug_log(f"Editor location set to: {st.session_state.editor_location}")

        # --- 调试区块移动到侧边栏 ---
        with st.expander("🛠️ 调试与会话信息", expanded=False):
            if st.session_state.debug_mode:
                st.subheader("调试日志")
                if st.session_state.debug_logs:
                    debug_logs_combined = "\n".join(st.session_state.debug_logs)
                    st.text_area("Debug Logs", value=debug_logs_combined, height=200)
                else:
                    st.write("没有调试日志。")

                st.subheader("调试错误")
                if st.session_state.debug_errors:
                    debug_errors_combined = "\n".join(st.session_state.debug_errors)
                    st.text_area("Debug Errors", value=debug_errors_combined, height=200)
                else:
                    st.write("没有调试错误。")

            st.subheader("会话信息 (messages.json)")
            if "messages" in st.session_state:
                messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
                st.text_area("messages.json", value=messages_json, height=300)

                # 添加下载按钮
                st.download_button(
                    label="📥 下载 messages.json",
                    data=messages_json,
                    file_name="messages.json",
                    mime="application/json"
                )

                st.markdown("---")  # 添加分隔线

                # 新增按钮：显示原始消息
                if st.button("📄 显示原始消息"):
                    st.subheader("🔍 原始消息内容")
                    st.json(st.session_state.messages)  # 使用 st.json 格式化显示
            else:
                st.write("没有找到 messages。")

    # --- Display Message History ---
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                # 處理列表形式的訊息內容，例如 image_url
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        st.image(image_url, caption="📷 上傳的圖片", use_column_width=True)
                        debug_log(f"Displaying image from {message['role']}: {image_url}")
                    else:
                        st.write(item)
                        debug_log(f"Displaying non-image content from {message['role']}: {item}")
            elif isinstance(message["content"], str) and "```python" in message["content"]:
                # 處理包含 Python 代碼塊的文字訊息
                code_match = re.search(r'```python\s*(.*?)\s*```', message["content"], re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    st.code(code, language="python")
                    debug_log(f"Displaying code from {message['role']}: {code}")
                else:
                    st.write(message["content"])  # 顯示上傳對話
                    debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")
            else:
                # 處理普通的文字訊息
                st.write(message["content"])
                debug_log(f"Displaying message {idx} from {message['role']}: {message['content']}")

    # --- User Input ---
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
                    client = initialize_client(api_key)
                else:
                    raise ValueError("OpenAI API Key is not provided.")

                debug_log(f"Uploaded file path: {st.session_state.uploaded_file_path}")
                debug_log(f"Uploaded image path: {st.session_state.uploaded_image_path}")

                # --- Ensure system prompt is added only once ---
                if not any(msg["role"] == "system" for msg in st.session_state.messages):
                    system_prompt = "You are an assistant that helps with data analysis."
                    append_message("system", system_prompt)
                    debug_log("System prompt added to messages.")

                # --- Decide which prompt to use ---
                if st.session_state.uploaded_image_path is not None and st.session_state.image_base64:
                    # Image uploaded, image data already added as a separate message
                    prompt = user_input  # Use user input directly
                    debug_log("User input with image data already appended.")
                else:
                    # No image uploaded, use complex JSON logic
                    if st.session_state.uploaded_file_path is not None:
                        try:
                            df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                            csv_columns = ", ".join(df_temp.columns)
                            debug_log(f"CSV columns: {csv_columns}")
                        except Exception as e:
                            csv_columns = "Unable to read columns"
                            if st.session_state.debug_mode:
                                st.error(f"Error reading columns: {e}")
                            debug_log(f"Error reading columns: {e}")
                    else:
                        csv_columns = "No file uploaded"
                        debug_log("No CSV file uploaded.")

                    if st.session_state.uploaded_file_path is not None and csv_columns != "No file uploaded":
                        prompt = f"""Please respond with a JSON object in the format:
{{
    "content": "這是我的觀察跟分析: {{analysis}}",
    "code": "import pandas as pd\\nimport streamlit as st\\nimport matplotlib.pyplot as plt\\n# Read CSV file (use st.session_state.uploaded_file_path variable)\\ndata = pd.read_csv(st.session_state.uploaded_file_path)\\n\\n# Add your plotting or analysis logic here\\n\\n# For example, to display a plot using st.pyplot():\\n# fig, ax = plt.subplots()\\n# ax.scatter(data['colA'], data['colB'])\\n# st.pyplot(fig)"
}}
Important:
1) 必須使用 st.session_state.uploaded_file_path 作為 CSV 路徑 (instead of a hardcoded path)
2) Must use st.pyplot() to display any matplotlib figure
3) Return only valid JSON (escape any special characters if needed)

Based on the request: {user_input}.
Available columns: {csv_columns}.
然後請使用繁體中文回應
"""
                        debug_log("Prompt constructed for CSV input with JSON response.")
                        append_message("system", prompt)
                        debug_log("System prompt appended to messages.")
                    else:
                        prompt = f"Please answer this question entirely in Traditional Chinese: {user_input}"
                        debug_log("Prompt constructed for plain text input.")
                        append_message("system", prompt)
                        debug_log("Plain text system prompt appended to messages.")

                # Make the API request and get the response
                model_params = {
                    "model": selected_model,
                    "temperature": 0.5,
                    "max_tokens": 4096
                }

                response_content = get_llm_response(client, model_params)
                debug_log(f"Full assistant response: {response_content}")

                if response_content:
                    # After getting the response, append assistant message
                    append_message("assistant", response_content)
                    with st.chat_message("assistant"):
                        st.write(response_content)  # 避免二次顯示
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

                    content = response_json.get("content", "Here is my analysis:")
                    append_message("assistant", content)
                    # with st.chat_message("assistant"):
                    #     # st.write(content)    # 避免二次顯示
                    #     debug_log(f"Content from JSON appended to messages: {content}")

                    code = response_json.get("code", "")
                    if code:
                        code_block = f"```python\n{code}\n```"
                        append_message("assistant", code_block)
                        with st.chat_message("assistant"):
                            st.code(code, language="python")
                        st.session_state.ace_code = code
                        debug_log("ace_code updated with new code.")

                    # --- If deep analysis mode is checked & code is present -> execute code and re-analyze chart ---
                    if st.session_state.deep_analysis_mode and code:
                        st.write("### [Deep Analysis] Automatically executing the generated code and sending the chart to GPT-4o for analysis...")
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
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        st.session_state.deep_analysis_image = chart_base64
                        debug_log("Chart has been converted to base64.")

                        # Prepare deep analysis prompt
                        prompt_2 = f"""基於圖片給我更多資訊"""
                        debug_log(f"Deep Analysis Prompt: {prompt_2}")

                        # Append prompt_2 to messages
                        append_message("user", prompt_2)
                        debug_log("Deep analysis prompt appended to messages.")

                        # 把圖片加到二次分析裡
                        image_content = [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{chart_base64}"}
                        }]
                        append_message("user", image_content)  # 添加圖片到消息

                        # Make the API request for deep analysis
                        second_raw_response = get_llm_response(client, model_params)
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
First response content: {content}
Second response chart analysis content: {second_raw_response}

請把前兩次的分析內容做分析總結，有數據的話就顯示得漂亮一點，主要是需要讓使用者感到很厲害。並且以繁體中文作為回答用的語言。
另外需要解釋傳給妳的圖表，以一個沒有資料科學背景的小白解釋我所傳的圖表。還有根據第二次的圖表分析得出的結論，直接預測之後的走向，例如:"之後這個數值的走向會呈現向上的趨勢"等...
不要跟使用者說甚麼妳可以使用RFM分析，交叉分析之類的方法。我需要妳直接預測之後的走向，比如往上還是往下。
"""
                            debug_log(f"Final Summary Prompt: {prompt_3}")

                            # Append prompt_3 to messages
                            append_message("user", prompt_3)
                            debug_log("Final summary prompt appended to messages.")

                            # Make the API request for final summary
                            third_raw_response = get_llm_response(client, model_params)
                            debug_log(f"Final summary response: {third_raw_response}")

                            if third_raw_response:
                                # Append assistant response
                                append_message("assistant", third_raw_response)
                                st.session_state.third_response = third_raw_response
                                with st.chat_message("assistant"):
                                    st.write(third_raw_response)
                                    debug_log(f"Final summary response added to messages: {third_raw_response}")

                                # Display the chart
                                st.write("#### [Deep Analysis] Chart:")
                                try:
                                    img_data = base64.b64decode(st.session_state.deep_analysis_image)
                                    st.image(img_data, caption="Chart generated from deep analysis", use_column_width=True)
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
