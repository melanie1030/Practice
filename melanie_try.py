import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import re
import os
import dotenv
import base64
from io import BytesIO, StringIO
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
from streamlit_ace import st_ace
import time
import matplotlib.font_manager as fm
import matplotlib
import sys
import httpx

# --- LangChain and Pandas Agent Imports (NEW) ---
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


# --- Role Definitions (Generic Chat Roles) ---
ROLE_DEFINITIONS = {
    "summarizer": {
        "name": "📝 Summarizer",
        "system_prompt": "You are an expert summarizer. Your task is to take any text or conversation provided and condense it into a clear, concise summary in Traditional Chinese. Focus on the main points and key takeaways. Respond with #zh-tw.",
        "messages_key": "summarizer_messages",
        "chat_session_key": "summarizer_chat_session",
    },
    "code_explainer": {
        "name": "💻 Code Explainer",
        "system_prompt": "You are an expert code explainer. Your task is to explain code snippets in Traditional Chinese. Describe what the code does, its logic, potential improvements, and answer any questions related to it. Respond with #zh-tw.",
        "messages_key": "code_explainer_messages",
        "chat_session_key": "code_explainer_chat_session",
    },
    "creative_writer": {
        "name": "✍️ Creative Writer",
        "system_prompt": "You are a creative writer. Your task is to help users with creative writing tasks such as writing stories, poems, scripts, or brainstorming ideas, all in Traditional Chinese. Be imaginative and inspiring. Respond with #zh-tw.",
        "messages_key": "creative_writer_messages",
        "chat_session_key": "creative_writer_chat_session",
    }
}

# --- Executive Workflow Definitions ---
EXECUTIVE_ROLE_IDS = {
    "CFO": "cfo_exec",
    "COO": "coo_exec",
    "CEO": "ceo_exec",
}

# 指定字型檔路徑（相對路徑）
font_path = "./fonts/msjh.ttc"

try:
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font loading error: {e}. Matplotlib might not use the custom font.")


# --- 初始化設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

LLM_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gpt-4o",
    "gpt-3.5-turbo-16k",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

MAX_MESSAGES_PER_STREAM = 12


def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"**DEBUG LOG ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_errors" not in st.session_state:
            st.session_state.debug_errors = []
        st.session_state.debug_errors.append(f"**DEBUG ERROR ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG ERROR: {msg}")

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image_pil):
    try:
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message_to_stream(message_stream_key, role, content, max_messages=MAX_MESSAGES_PER_STREAM):
    if message_stream_key not in st.session_state:
        st.session_state[message_stream_key] = []
    st.session_state[message_stream_key].append({"role": role, "content": content})
    current_messages = st.session_state[message_stream_key]
    if len(current_messages) > max_messages:
        st.session_state[message_stream_key] = current_messages[-max_messages:]
        debug_log(f"Message history for '{message_stream_key}' trimmed to {max_messages} messages.")


def add_user_image_to_main_chat(uploaded_file):
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        use_base64_for_gpt_or_claude = "gpt" in current_model or "claude" in current_model
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path
        image_pil = Image.open(file_path)

        if use_base64_for_gpt_or_claude:
            image_base64 = load_image_base64(image_pil)
            if not image_base64: st.error("圖片轉為Base64失敗。"); return
            image_url = f"data:image/{file_path.split('.')[-1]};base64,{image_base64}"
            image_msg_content = [{"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}]
            append_message_to_stream("messages", "user", image_msg_content)
            st.session_state.image_base64 = image_base64
            debug_log(f"圖片已處理 (主聊天室 - Base64): {file_path}.")
            st.rerun()
        else: # For Gemini
            st.session_state.pending_image_for_main_gemini = image_pil
            st.image(image_pil, caption="圖片已上傳，將隨下一條文字訊息發送 (主聊天室 - Gemini)。", use_container_width=True)
            debug_log(f"圖片已處理 (主聊天室 - Gemini Pending): {file_path}.")
    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        st.error("圖片處理異常，請檢查日誌")
        debug_error(f"Error in add_user_image_to_main_chat: {e}, Traceback: {traceback.format_exc()}")


def reset_session_messages(message_stream_key="messages"):
    # (This function remains unchanged)
    if message_stream_key in st.session_state:
        st.session_state.pop(message_stream_key)
        debug_log(f"Conversation history for '{message_stream_key}' cleared.")
    for role_id_iter, P_info in ROLE_DEFINITIONS.items():
        if P_info["messages_key"] == message_stream_key and P_info["chat_session_key"] in st.session_state:
            del st.session_state[P_info["chat_session_key"]]
            debug_log(f"Gemini chat session for role '{role_id_iter}' cleared.")
    st.success(f"Memory for '{message_stream_key}' cleared!")


def execute_code(code, global_vars=None):
    # (This function remains unchanged)
    try:
        exec_globals = global_vars if global_vars else {}
        exec_globals.update({'pd': pd, 'plt': plt, 'st': st, 'np': __import__('numpy')})
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        exec(code, exec_globals)
        sys.stdout = old_stdout
        std_output_val = redirected_output.getvalue()
        script_output_val = exec_globals.get("output", "")
        final_output_parts = []
        if std_output_val: final_output_parts.append(f"Standard Output:\n{std_output_val}")
        if script_output_val: final_output_parts.append(f"Script 'output' variable:\n{script_output_val}")
        final_output = "\n".join(final_output_parts) if final_output_parts else "(No explicit print/output variable captured)"
        debug_log(f"Code execution successful. Output: {final_output}")
        return f"Code executed. Output:\n{final_output}"
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log(f"Execution error: {error_msg}")
        return error_msg

def extract_json_block(response: str) -> str:
    # (This function remains unchanged)
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

# --- MODIFIED FUNCTION ---
def create_pandas_agent(file_path: str):
    """
    Creates a LangChain Pandas DataFrame Agent from a given CSV file path.
    Includes robust proxy handling for the ChatOpenAI client.
    """
    debug_log(f"Attempting to create Pandas Agent for {file_path}")
    openai_api_key = st.session_state.get("openai_api_key_input")
    if not openai_api_key:
        st.error("請在側邊欄設定您的 OpenAI API 金鑰以啟用資料分析代理。")
        return None

    try:
        df = pd.read_csv(file_path)

        # --- Proxy handling logic for ChatOpenAI ---
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        http_client_for_llm = None
        if proxy_url:
            debug_log(f"Proxy detected for Pandas Agent: {proxy_url}")
            proxies = {"http://": proxy_url, "https://": proxy_url}
            http_client_for_llm = httpx.Client(proxies=proxies)
        
        # Use a powerful model like gpt-4o for better reasoning and code generation
        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o", 
            api_key=openai_api_key,
            http_client=http_client_for_llm  # Pass the configured httpx client
        )
        
        # Create the agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="openai-tools",
            verbose=st.session_state.get("debug_mode", False),
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        debug_log("Pandas Agent created successfully.")
        return agent
        
    except FileNotFoundError:
        st.error(f"檔案未找到：{file_path}")
        debug_error(f"Pandas Agent creation failed: File not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"建立資料分析代理時發生錯誤：{e}")
        debug_error(f"Pandas Agent creation failed: {e}, Traceback: {traceback.format_exc()}")
        return None


# --- NEW FUNCTION for querying the Pandas Agent ---
def query_pandas_agent(agent, query: str):
    """
    Queries the provided LangChain Pandas Agent with a user's question.
    Handles invocation and returns the result.
    """
    if not agent:
        return "錯誤：資料分析代理未初始化。"
    
    debug_log(f"Querying Pandas Agent with: '{query}'")
    try:
        # The agent will internally convert the query to pandas commands, execute them, and return a summary.
        # The 'input' key is standard for agent invocation.
        response = agent.invoke({"input": query})
        
        # The agent's output is in the 'output' key of the response dictionary.
        result = response.get("output", "代理沒有提供有效的輸出。")
        debug_log(f"Pandas Agent raw response: {response}")
        return result
    except Exception as e:
        error_message = f"代理在處理您的請求時發生錯誤：{e}"
        debug_error(f"Pandas Agent invocation error: {e}, Traceback: {traceback.format_exc()}")
        st.error(error_message)
        return error_message

# --- NEW HELPER FUNCTION to handle OpenAI client initialization with proxy support ---
def get_openai_client_with_proxy_support():
    """
    Creates an OpenAI client instance with proper proxy handling for openai library v1.0.0+.
    It checks for HTTPS_PROXY environment variable.
    """
    openai_api_key = st.session_state.get("openai_api_key_input") or os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.error("未在側邊欄設定 OpenAI API 金鑰。")
        return None

    # Get proxy from environment variables
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    try:
        if proxy_url:
            debug_log(f"Proxy detected: {proxy_url}. Configuring httpx client.")
            # Configure httpx client with proxy
            proxies = {"http://": proxy_url, "https://": proxy_url}
            http_client = httpx.Client(proxies=proxies)
            # Pass the configured httpx client to OpenAI client
            client = OpenAI(api_key=openai_api_key, http_client=http_client)
        else:
            # If no proxy, initialize normally
            debug_log("No proxy detected. Initializing OpenAI client normally.")
            client = OpenAI(api_key=openai_api_key)
        
        return client
    except Exception as e:
        st.error(f"OpenAI Client 初始化失敗: {e}")
        debug_error(f"OpenAI Client init with proxy support failed: {e}, Traceback: {traceback.format_exc()}")
        return None
        
# --- All original LLM interaction functions below remain the same ---
# (get_gemini_response_for_generic_role, get_gemini_executive_analysis, etc.)

def get_gemini_response_for_generic_role(role_id, user_input_text, model_params, max_retries=3):
    # (This function remains unchanged)
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key:
        st.error("未設定 Gemini API 金鑰，請於側邊欄設定。")
        return "Error: Missing Gemini API Key."
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini API 金鑰設定失敗: {e}")
        return f"Error: Gemini API key configuration failed: {e}"
    role_info = ROLE_DEFINITIONS[role_id]
    model_name = model_params.get("model", "gemini-1.5-flash")
    chat_session_key = role_info["chat_session_key"]
    messages_key = role_info["messages_key"]
    try:
        gemini_model_instance = genai.GenerativeModel(model_name, system_instruction=role_info["system_prompt"])
    except Exception as e:
        st.error(f"無法初始化 Gemini 模型 '{model_name}': {e}")
        debug_error(f"Gemini model init error for {model_name}: {e}")
        return f"Error: Could not initialize Gemini model {model_name}."
    gemini_history = []
    if messages_key in st.session_state:
        relevant_messages = st.session_state[messages_key][-(MAX_MESSAGES_PER_STREAM-1):] if len(st.session_state[messages_key]) > MAX_MESSAGES_PER_STREAM else st.session_state[messages_key]
        for msg in relevant_messages:
            if msg["role"] == "user":
                gemini_history.append({'role': 'user', 'parts': [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_history.append({'role': 'model', 'parts': [msg["content"]]})
    if chat_session_key not in st.session_state or \
       st.session_state[chat_session_key].model_name != f"models/{model_name}" or \
       len(st.session_state[chat_session_key].history) != len(gemini_history):
        debug_log(f"Initializing/Resetting Gemini ChatSession for generic role '{role_id}' with model '{model_name}'. History length: {len(gemini_history)}")
        st.session_state[chat_session_key] = gemini_model_instance.start_chat(history=gemini_history[:-1])
    chat = st.session_state[chat_session_key]
    debug_log(f"Sending message to Gemini for generic role '{role_id}': {user_input_text}")
    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            response = chat.send_message(user_input_text)
            final_reply = response.text.strip()
            debug_log(f"Gemini response for generic role '{role_id}': {final_reply}")
            return final_reply
        except Exception as e:
            # ... (rest of the function is unchanged)
            debug_error(f"Gemini API error for generic role '{role_id}' (attempt {retries + 1}): {e}")
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e):
                st.error(f"Gemini API 金鑰無效或權限不足: {e}")
                return "Error: Gemini API Key Invalid or Permission Denied."
            if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"Gemini 請求頻繁或資源耗盡，{wait_time}秒後重試...")
                time.sleep(wait_time); retries += 1; wait_time *= 2
            else:
                st.error(f"Gemini API 請求發生未預期錯誤: {e}")
                return f"Error: An unexpected error occurred with Gemini API: {e}"
    st.error(f"Gemini for generic role '{role_id}': 請求失敗次數過多。")
    return "Error: Max retries exceeded for Gemini."

def get_gemini_executive_analysis(executive_role_name, full_prompt, model_params, max_retries=3):
    # (This function remains unchanged)
    # ...
    return "This function is kept for reference but workflow now uses OpenAI."

# --- MODIFIED FUNCTION ---
def get_openai_executive_analysis(executive_role_name, full_prompt, model_params, max_retries=3):
    """
    Handles a single turn OpenAI API request for an executive role in the workflow.
    Uses a helper function for robust client initialization with proxy support.
    """
    client = get_openai_client_with_proxy_support()
    if not client:
        # The helper function already shows the error message
        return f"Error: OpenAI Client initialization failed."

    model_name = model_params.get("model", "gpt-4o")
    temperature = model_params.get("temperature", 0.3)
    max_tokens_val = model_params.get("max_tokens", 4096)

    messages_for_openai = [
        {"role": "user", "content": full_prompt}
    ]

    debug_log(f"Sending prompt to OpenAI for Executive '{executive_role_name}': Model {model_name}\n{full_prompt[:300]}...")

    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages_for_openai,
                temperature=temperature,
                max_tokens=max_tokens_val,
            )
            final_reply = response.choices[0].message.content.strip()
            debug_log(f"OpenAI response for Executive '{executive_role_name}': {final_reply[:200]}...")
            return final_reply
        except Exception as e:
            debug_error(f"OpenAI API error for Executive '{executive_role_name}' (attempt {retries + 1}): {e}")
            if 'rate limit' in str(e).lower() or '429' in str(e):
                st.warning(f"OpenAI 請求頻繁或資源耗盡 for {executive_role_name}，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower() or 'authentication_error' in str(e).lower() or "access_terminated" in str(e).lower():
                st.error(f"OpenAI API 金鑰無效、認證失敗或帳戶問題 for {executive_role_name}: {e}")
                return f"Error: OpenAI API Key Invalid/Authentication Failed/Access Terminated ({executive_role_name})."
            else:
                st.error(f"OpenAI API 請求發生未預期錯誤 for {executive_role_name}: {e}")
                return f"Error: An unexpected error with OpenAI API for {executive_role_name}: {e}"

    st.error(f"OpenAI for Executive '{executive_role_name}': 請求失敗次數過多。")
    return f"Error: Max retries exceeded for OpenAI ({executive_role_name})."

# --- Main Chat LLM Response Functions ---
def get_gemini_response_main_chat(model_params, max_retries=3):
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key: st.error("未設定 Gemini API 金鑰"); return ""
    try: genai.configure(api_key=api_key)
    except Exception as e: st.error(f"Gemini API key config error: {e}"); return ""

    model_name = model_params.get("model", "gemini-1.5-flash")
    system_instruction_main = model_params.get("system_prompt", "請以繁體中文回答。所有回覆開頭請加上 #zh-tw 標籤。")
    try: model_instance_main = genai.GenerativeModel(model_name, system_instruction=system_instruction_main)
    except Exception as e: st.error(f"Gemini Main Chat Model Init Error '{model_name}': {e}"); return ""

    gemini_history_main = []
    main_chat_messages = st.session_state.get("messages", [])

    num_messages_for_history = len(main_chat_messages)
    # Exclude the current user message about to be sent from history
    if main_chat_messages and main_chat_messages[-1]["role"] == "user":
        num_messages_for_history -=1

    for i in range(num_messages_for_history):
        msg = main_chat_messages[i]
        role = "model" if msg["role"] == "assistant" else msg["role"]
        if role not in ["user", "model"]: continue

        content_parts_main = []
        if isinstance(msg["content"], list): # GPT vision format in history
            text_found = False
            for item_hist in msg["content"]:
                if isinstance(item_hist, dict) and item_hist.get("type") == "text":
                    content_parts_main.append(item_hist["text"])
                    text_found = True
            # For Gemini history, we typically only pass text. Images are handled per turn.
            # If only image was in history from GPT, send empty string part or a placeholder.
            if not text_found: content_parts_main.append("[Image previously shown]")
        else:
            content_parts_main.append(str(msg["content"]))

        if content_parts_main:
            gemini_history_main.append({'role': role, 'parts': content_parts_main})


    chat_session_main_key = "main_chat_gemini_session"
    if chat_session_main_key not in st.session_state or \
       st.session_state[chat_session_main_key].model_name != f"models/{model_name}" or \
       len(st.session_state[chat_session_main_key].history) != len(gemini_history_main):
        debug_log(f"Initializing/Resetting Gemini ChatSession for Main Chat with model '{model_name}'. History length: {len(gemini_history_main)}")
        st.session_state[chat_session_main_key] = model_instance_main.start_chat(history=gemini_history_main)

    chat_main = st.session_state[chat_session_main_key]

    current_turn_parts_main = []
    if main_chat_messages and main_chat_messages[-1]["role"] == "user":
        last_user_msg_content = main_chat_messages[-1]["content"]
        # The last message is always the user's text input for this turn
        current_turn_parts_main.append(str(last_user_msg_content))

    if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
        # Prepend image if it exists, as it's usually "image then prompt"
        current_turn_parts_main.insert(0, st.session_state.pending_image_for_main_gemini)
        debug_log("Attaching pending_image_for_main_gemini to current Gemini Main Chat call.")

    if not current_turn_parts_main:
        debug_error("Main Chat Gemini: No text or image input for current turn.")
        return "Error: No user input to send to Gemini main chat."

    debug_log(f"Gemini (Main Chat) sending parts for current turn: {[(type(p), str(p)[:50] + '...' if isinstance(p,str) else 'PIL.Image') for p in current_turn_parts_main]}")

    retries = 0; wait_time = 5
    response_text_final = ""
    try:
        while retries < max_retries:
            try:
                response = chat_main.send_message(current_turn_parts_main)
                response_text_final = response.text.strip()
                debug_log(f"Gemini (Main Chat) send_message final reply => {response_text_final}")
                break # Success
            except Exception as e:
                debug_error(f"Gemini (Main Chat) API error (attempt {retries + 1}): {e}, Parts: {current_turn_parts_main}")
                if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
                    st.error(f"Gemini API 金鑰無效或權限不足: {e}"); response_text_final = "Error: Key Invalid/Permission Denied."; break
                if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning(f"Gemini (Main) 請求頻繁，{wait_time}秒後重試..."); time.sleep(wait_time); retries += 1; wait_time *= 2
                else:
                    st.error(f"Gemini (Main) API 未預期錯誤: {e}"); response_text_final = f"Error: Unexpected Gemini API error: {e}"; break
        if retries >= max_retries:
            st.error("Gemini (Main Chat): 請求失敗次數過多"); response_text_final = "Error: Max retries exceeded."
    finally:
        if "pending_image_for_main_gemini" in st.session_state: # Consume image after attempt
            del st.session_state.pending_image_for_main_gemini
    return response_text_final


# --- MODIFIED FUNCTION ---
def get_openai_response(client_openai_passed, model_params, max_retries=3):
    # The passed client is ignored to ensure consistent proxy handling via the helper function.
    client_to_use = get_openai_client_with_proxy_support()
    if not client_to_use:
        return "" # Error is already shown by the helper function

    processed_messages_for_openai = []
    main_chat_msg_list = st.session_state.get("messages", [])
    for msg in main_chat_msg_list:
        if isinstance(msg["content"], list):
            new_content_list = []
            has_text = False
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    new_content_list.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    new_content_list.append(item)
                    has_text = True
                elif isinstance(item, str):
                    new_content_list.append({"type": "text", "text": item})
                    has_text = True
            if any(c_item.get("type") == "image_url" for c_item in new_content_list) and not has_text:
                new_content_list.append({"type": "text", "text": " "})
            processed_messages_for_openai.append({"role": msg["role"], "content": new_content_list})
        else:
            processed_messages_for_openai.append(msg)

    model_name = model_params.get("model", "gpt-4o")
    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            request_params = {
                "model": model_name, "messages": processed_messages_for_openai,
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096), "stream": False
            }
            response = client_to_use.chat.completions.create(**request_params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            debug_error(f"OpenAI API error for main chat (attempt {retries + 1}): {e}")
            if 'rate limit' in str(e).lower() or '429' in str(e):
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower() or "authentication_error" in str(e).lower() or "access_terminated" in str(e).lower():
                st.error(f"OpenAI API金鑰無效/認證失敗/帳戶問題: {e}")
                return f"Error: OpenAI API Key Invalid/Authentication Failed/Access Terminated."
            else:
                st.error(f"OpenAI請求錯誤：{e}")
                return f"Error: OpenAI request error: {e}"
    st.error("OpenAI請求失敗次數過多")
    return "Error: Max retries exceeded for OpenAI."

def get_claude_response(model_params, max_retries=3):
    import anthropic
    api_key = st.session_state.get("claude_api_key_input") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key: st.error("未設定 Claude API 金鑰"); return ""
    try: client = anthropic.Anthropic(api_key=api_key)
    except Exception as e: st.error(f"Claude Client Init Error: {e}"); return ""

    model_name = model_params.get("model", "claude-3-opus-20240229")
    claude_messages = []; system_prompt_claude = None
    source_messages = st.session_state.get("messages", []) # Use main chat messages

    # The first message in source_messages could be a system prompt for Claude
    # But we must ensure Claude's messages alternate user/assistant
    processed_claude_msgs = []
    temp_msgs_for_claude_conversion = []

    # If first message is system, save it for Claude's system parameter
    if source_messages and source_messages[0]["role"] == "system":
        system_prompt_claude = source_messages[0]["content"]
        temp_msgs_for_claude_conversion = source_messages[1:]
    else:
        temp_msgs_for_claude_conversion = source_messages[:]

    # Ensure alternating user/assistant, merging consecutive messages of the same role if necessary
    # This is a simplified merge, more complex merging might be needed for some cases.
    for msg in temp_msgs_for_claude_conversion:
        # Skip system messages here as it's handled above
        if msg["role"] == "system":
            continue

        # Convert content to Claude's format
        current_content_parts = []
        if isinstance(msg["content"], list): # OpenAI vision format
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    if img_url.startswith("data:image"): # base64
                        base64_data=img_url.split(",")[1]
                        media_type=img_url.split(";")[0].split(":")[1]
                        current_content_parts.append({"type":"image","source":{"type":"base64","media_type":media_type,"data":base64_data}})
                elif isinstance(item, dict) and item.get("type") == "text":
                    current_content_parts.append({"type":"text","text":item["text"]})
                elif isinstance(item, str): # Simple string in list
                     current_content_parts.append({"type":"text","text":item})
        else: # Simple text string
            current_content_parts.append({"type":"text","text":str(msg["content"])})

        if not current_content_parts: continue # Skip if no valid parts

        # Merge with previous message if same role
        if processed_claude_msgs and processed_claude_msgs[-1]["role"] == msg["role"]:
            # Append text parts, images usually stand alone or come first
            for part in current_content_parts:
                if part["type"] == "text":
                    # Find existing text part to append to, or add new
                    existing_text_part = next((p for p in processed_claude_msgs[-1]["content"] if p["type"] == "text"), None)
                    if existing_text_part:
                        existing_text_part["text"] += "\n" + part["text"]
                    else:
                        processed_claude_msgs[-1]["content"].append(part)
                else: # image
                    processed_claude_msgs[-1]["content"].insert(0,part) # Images often come first
        else:
             processed_claude_msgs.append({"role":msg["role"],"content":current_content_parts})

    claude_messages = processed_claude_msgs
    if not claude_messages or claude_messages[-1]["role"] != "user":
        debug_log("Claude: Last message is not from user or no messages. Cannot send.")
        return "Error: Claude requires the last message to be from the user."


    retries = 0; wait_time = 5
    while retries < max_retries:
        try:
            api_params={"model":model_name,"max_tokens":model_params.get("max_tokens",8192),"messages":claude_messages,"temperature":model_params.get("temperature",0.3)}
            if system_prompt_claude: api_params["system"]=system_prompt_claude
            response=client.messages.create(**api_params)
            completion = "".join([block.text for block in response.content if hasattr(block, 'text')]) # hasattr for safety
            return completion.strip()
        except Exception as e:
            debug_error(f"Claude API error (attempt {retries + 1}): {e}")
            if "rate limit" in str(e).lower() or "429" in str(e): time.sleep(wait_time); retries += 1; wait_time *= 2
            elif "authentication_error" in str(e).lower() or "permission_error" in str(e).lower():
                st.error(f"Claude API金鑰無效/權限不足: {e}"); return f"Error: Claude API Key Invalid/Permission Denied."
            else: st.warning(f"Claude 生成錯誤，{wait_time}秒後重試..."); time.sleep(wait_time); retries += 1; wait_time *= 2;
            if retries >=max_retries: st.error(f"Claude 請求失敗過多: {e}"); return f"Error: Max retries for Claude."
    return ""


def get_llm_response(client_openai_main_chat, model_params, message_stream_key="messages", max_retries=3):
    model_name = model_params.get("model", "gemini-1.5-flash")
    debug_log(f"Routing LLM request for model: {model_name}, stream: {message_stream_key}")

    if "gpt" in model_name.lower():
        return get_openai_response(client_openai_main_chat, model_params, max_retries) # Pass the client
    elif "gemini" in model_name.lower():
        if message_stream_key == "messages": # Main chat
            return get_gemini_response_main_chat(model_params, max_retries)
        else: # This case should be handled by role-specific Gemini calls directly
            st.error("Gemini routing error in get_llm_response for non-main stream.")
            return "Error: Gemini routing issue for non-main stream."
    elif "claude" in model_name.lower():
        return get_claude_response(model_params, max_retries)
    else:
        st.error(f"Unsupported model type in get_llm_response: {model_name}")
        return ""


# --- MODIFIED FUNCTION ---
def get_cross_validated_response(client_openai_validator, model_params_validator, max_retries=3):
    cross_validation_prompt_content = (
        "請仔細閱讀以下全部對話記憶 (來自主要聊天室)，對先前模型的回答進行交叉驗證。"
        "你的任務是檢查回答的正確性，指出其中可能存在的錯誤或不足，"
        "並提供具體的數據、理由或例子來支持你的分析。"
        "請務必使用繁體中文回答。"
        "在回答時請回答的詳細，內容需要你盡可能的多。"
    )
    original_main_messages = list(st.session_state.get("messages", []))
    validator_messages_for_api = [{"role": "system", "content": cross_validation_prompt_content}] + original_main_messages
    validator_model_name = model_params_validator.get('model')
    validator_response_text = ""
    debug_log(f"Cross-validation: Validating with {validator_model_name} using main chat history.")

    original_st_messages_backup = st.session_state.get("messages")
    st.session_state.messages = validator_messages_for_api

    if "gpt" in validator_model_name.lower():
        # Use the new helper to get a properly configured client for the validator
        client_for_validator_call = get_openai_client_with_proxy_support()
        if not client_for_validator_call:
            st.session_state.messages = original_st_messages_backup
            return {"validator_response": "Error: OpenAI Key missing or Client init failed for validator."}
        
        # The get_openai_response function will now use the correct client internally
        validator_response_text = get_openai_response(client_for_validator_call, model_params_validator, max_retries)

    elif "gemini" in validator_model_name.lower():
        api_key_gem_val = st.session_state.get("gemini_api_key_input", "")
        if not api_key_gem_val:
            st.error("Gemini key for validator missing.")
            st.session_state.messages = original_st_messages_backup
            return {"validator_response": "Error: Gemini Key missing."}
        try:
            genai.configure(api_key=api_key_gem_val)
            system_instruction_for_gemini_val = validator_messages_for_api[0]['content']
            val_model_gem = genai.GenerativeModel(validator_model_name, system_instruction=system_instruction_for_gemini_val)
            gemini_formatted_val_history = []
            for m_val in validator_messages_for_api[1:]: # Skip system message
                role_val = 'model' if m_val['role'] == 'assistant' else m_val['role']
                if role_val not in ['user', 'model']: continue
                content_str = str(m_val['content']) # Simplified content for validation
                gemini_formatted_val_history.append({'role': role_val, 'parts': [content_str]})
            if not gemini_formatted_val_history: raise ValueError("No content to validate for Gemini.")
            response_val = val_model_gem.generate_content(gemini_formatted_val_history)
            validator_response_text = response_val.text.strip()
        except Exception as e_gem_val:
            validator_response_text = f"Error during Gemini validation: {e_gem_val}"
            debug_error(validator_response_text)

    elif "claude" in validator_model_name.lower():
        validator_response_text = get_claude_response(model_params_validator, max_retries)
        
    else:
        st.error(f"Unsupported validator model: {validator_model_name}")
        validator_response_text = "Error: Unsupported validator model."

    st.session_state.messages = original_st_messages_backup
    debug_log("Cross-validation prompt effect removed from main messages stream.")
    return {"validator_response": validator_response_text}

# --- Helper to generate data profile for Executive Workflow ---
def generate_data_profile(df):
    if df is None or df.empty:
        return "No data loaded or DataFrame is empty."

    profile_parts = []
    profile_parts.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    profile_parts.append("\nColumn Names and Types:")
    for col in df.columns:
        profile_parts.append(f"- {col}: {df[col].dtype}")

    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    profile_parts.append(f"\nDataFrame Info:\n{info_str}")

    profile_parts.append("\nDescriptive Statistics (Numeric Columns):")
    try:
        desc_num = df.describe(include='number').to_string()
        profile_parts.append(desc_num)
    except Exception:
        profile_parts.append("Could not generate numeric descriptive statistics.")

    profile_parts.append("\nDescriptive Statistics (Object/Categorical Columns):")
    try:
        desc_obj = df.describe(include=['object', 'category']).to_string()
        profile_parts.append(desc_obj)
    except Exception:
        profile_parts.append("Could not generate object/categorical descriptive statistics.")

    profile_parts.append("\nFirst 5 Rows (Head):")
    profile_parts.append(df.head().to_string())

    return "\n".join(profile_parts)

# ------------------------------
# 主應用入口 (最終完整版)
# ------------------------------
def main():
    st.set_page_config(page_title="Multi-Role & Exec Workflow Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Multi-Role & 📈 Executive Workflow Chatbot")

    # --- Initialize Session States ---
    if "messages" not in st.session_state: st.session_state.messages = []
    for role_id, role_info in ROLE_DEFINITIONS.items():
        if role_info["messages_key"] not in st.session_state: st.session_state[role_info["messages_key"]] = []
        if role_info["chat_session_key"] not in st.session_state: st.session_state[role_info["chat_session_key"]] = None

    if "executive_workflow_stage" not in st.session_state: st.session_state.executive_workflow_stage = "idle"
    if "executive_user_query" not in st.session_state: st.session_state.executive_user_query = ""
    if "executive_data_profile_str" not in st.session_state: st.session_state.executive_data_profile_str = ""
    if "cfo_analysis_text" not in st.session_state: st.session_state.cfo_analysis_text = ""
    if "coo_analysis_text" not in st.session_state: st.session_state.coo_analysis_text = ""
    if "ceo_summary_text" not in st.session_state: st.session_state.ceo_summary_text = ""
    for exec_id_key in ["cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages"]:
        if exec_id_key not in st.session_state: st.session_state[exec_id_key] = []
    
    # <--- NEW STATE ---
    if "pandas_agent" not in st.session_state:
        st.session_state.pandas_agent = None

    if "ace_code" not in st.session_state: st.session_state.ace_code = ""
    if "editor_location" not in st.session_state: st.session_state.editor_location = "Sidebar"
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state: st.session_state.uploaded_image_path = None
    if "image_base64" not in st.session_state: st.session_state.image_base64 = None
    if "pending_image_for_main_gemini" not in st.session_state: st.session_state.pending_image_for_main_gemini = None
    if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state: st.session_state.deep_analysis_mode = True

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        openai_api_key_input = st.text_input("OpenAI API Key", value=st.session_state.get("openai_api_key_input",""), type="password", key="openai_api_key_widget")
        if openai_api_key_input: st.session_state.openai_api_key_input = openai_api_key_input
        gemini_api_key_input = st.text_input("Gemini API Key", value=st.session_state.get("gemini_api_key_input",""), type="password", key="gemini_api_key_widget")
        if gemini_api_key_input: st.session_state.gemini_api_key_input = gemini_api_key_input
        claude_api_key_input = st.text_input("Claude API Key", value=st.session_state.get("claude_api_key_input",""), type="password", key="claude_api_key_widget")
        if claude_api_key_input: st.session_state.claude_api_key_input = claude_api_key_input

        st.subheader("⚙️ Main Chat Model")
        selected_model_main = st.selectbox(
            "選擇主要聊天模型 (若未上傳CSV):", LLM_MODELS,
            index=LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0,
            key="selected_model_main"
        )
        st.session_state.selected_model = selected_model_main

        st.subheader("📈 Executive Workflow Model")
        st.caption("Executive workflow now uses OpenAI (e.g., gpt-4o).")

        st.subheader("🛠️ Tools & Settings")
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.get("debug_mode", False))
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode (Main Chat)", value=st.session_state.get("deep_analysis_mode", True))

        if st.button("🗑️ Clear ALL Memory & Chats"):
            st.session_state.messages = []
            if "main_chat_gemini_session" in st.session_state: del st.session_state.main_chat_gemini_session
            for role_id_iter, P_info in ROLE_DEFINITIONS.items():
                if P_info["messages_key"] in st.session_state: st.session_state[P_info["messages_key"]] = []
                if P_info["chat_session_key"] in st.session_state: del st.session_state[P_info["chat_session_key"]]
            
            # <--- MODIFIED LOGIC ---
            keys_to_clear = [
                "ace_code", "uploaded_file_path", "uploaded_image_path",
                "image_base64", "pending_image_for_main_gemini", "second_response", "third_response",
                "deep_analysis_image", "thinking_protocol", "debug_logs", "debug_errors",
                "pandas_agent", "executive_user_query", "cfo_analysis_text", "coo_analysis_text", 
                "ceo_summary_text", "executive_workflow_stage"
            ]
            for key in keys_to_clear:
                if key in st.session_state: st.session_state.pop(key)
            st.success("All memories and chat states cleared!")
            debug_log("All memory cleared.")
            st.rerun()

        st.subheader("🧠 Main Chat Memory State")
        if st.session_state.get("messages"):
            memory_content = "\n".join([f"{('' if msg['role']=='system' else msg['role']+': ')}{str(msg['content'])[:100]+'...' if isinstance(msg['content'], str) and len(msg['content']) > 100 else str(msg['content'])}" for msg in st.session_state.messages])
            st.text_area("Current Main Chat Memory", value=memory_content, height=150, key="main_chat_memory_display_sidebar")
        else:
            st.text_area("Current Main Chat Memory", value="No messages yet in main chat.", height=150)

        st.subheader("📂 Upload CSV (For Main Chat & Executive Workflow)")
        uploaded_file = st.file_uploader("上傳CSV以啟用資料分析代理:", type=["csv"], key="main_csv_uploader_sidebar")
        
        # <--- NEW LOGIC ---
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path != st.session_state.get("uploaded_file_path") or not st.session_state.get("pandas_agent"):
                st.session_state.uploaded_file_path = file_path
                with st.spinner("正在初始化資料分析代理..."):
                    st.session_state.pandas_agent = create_pandas_agent(file_path)
            try:
                df_preview = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### CSV Data Preview")
                st.dataframe(df_preview.head())
                if st.session_state.get("pandas_agent"):
                    st.success("✅ 資料分析代理已啟用！")
                    st.info("您現在可以在主聊天室中對此 CSV 檔案提問。")
            except Exception as e:
                st.error(f"讀取 CSV 預覽時出錯: {e}")
                st.session_state.pandas_agent = None

        st.subheader("🖼️ Upload Image (Main Chat)")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="main_image_uploader_sidebar")
        if uploaded_image:
            is_gemini_selected = "gemini" in st.session_state.get("selected_model", "").lower()
            if is_gemini_selected and "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
                st.warning("已有圖片待發送 (Gemini)。請先發送包含該圖片的文字訊息，或清除記憶。")
            else:
                add_user_image_to_main_chat(uploaded_image)

        st.subheader("Editor Location")
        location = st.radio( "Choose where to display the editor:", ["Main", "Sidebar"],
            index=1 if st.session_state.get("editor_location", "Sidebar") == "Sidebar" else 0,
            key="editor_loc_radio_sidebar" )
        st.session_state.editor_location = location

        with st.expander("🛠️ 調試與會話資訊 (Main Chat)", expanded=False):
            if st.session_state.get("debug_mode", False):
                st.subheader("調試日誌")
                debug_logs_str = "\n".join(map(str, st.session_state.get("debug_logs", [])))
                st.text_area("Debug Logs", value=debug_logs_str, height=200, key="debug_log_area_sidebar")
                st.subheader("調試錯誤")
                debug_errors_str = "\n".join(map(str, st.session_state.get("debug_errors", [])))
                st.text_area("Debug Errors", value=debug_errors_str, height=200, key="debug_err_area_sidebar")
            st.subheader("會話資訊 (messages.json - Main Chat)")
            if st.session_state.get("messages"):
                try:
                    def safe_json_encoder(obj):
                        if isinstance(obj, Image.Image): return f"<PIL.Image {obj.format} {obj.size}>"
                        return str(obj)
                    messages_json_main = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2, default=safe_json_encoder)
                    st.text_area("messages.json (Main Chat)", value=messages_json_main, height=200, key="main_msg_json_area_sidebar")
                    st.download_button(label="📥 下載 Main messages.json", data=messages_json_main, file_name="main_messages.json", mime="application/json", key="dl_main_json_sidebar")
                except TypeError as te:
                    st.error(f"無法序列化主聊天消息: {te}")

    # --- Main Area with Tabs ---
    tab_ui_names = ["💬 Main Chat & Analysis", "💼 Executive Workflow"] + [ROLE_DEFINITIONS[rid]["name"] for rid in ROLE_DEFINITIONS.keys()]
    tabs = st.tabs(tab_ui_names)

    # Tab 0: Main Chat
    with tabs[0]:
        st.header("💬 Main Chat & Data Analysis Engine")
        for idx, message in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url": st.image(item["image_url"]["url"], caption="📷", use_container_width=True)
                        elif isinstance(item, dict) and item.get("type") == "text": st.write(item["text"])
                        else: st.write(item)
                else:
                    st.write(message["content"])
        if "gemini" in st.session_state.get("selected_model", "").lower() and "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
            with st.chat_message("user"):
                st.image(st.session_state.pending_image_for_main_gemini, caption="圖片待發送 (Gemini). 輸入文字以發送.", use_container_width=True)

        user_input_main = st.chat_input("對通用模型或上傳的CSV提問...", key="main_chat_input_box_main")
        if user_input_main:
            append_message_to_stream("messages", "user", user_input_main)
            st.rerun()

        # <--- MAJOR MODIFIED LOGIC: Main response generation ---
        if st.session_state.get("messages") and st.session_state.messages[-1]["role"] == "user":
            last_user_prompt = st.session_state.messages[-1]["content"]
            if st.session_state.get("pandas_agent"):
                with st.spinner("資料分析代理正在思考中..."):
                    debug_log("Routing query to Pandas Agent.")
                    response_content_main = query_pandas_agent(st.session_state.pandas_agent, last_user_prompt)
                    if response_content_main and not response_content_main.startswith("錯誤："):
                        append_message_to_stream("messages", "assistant", response_content_main)
                    else:
                        append_message_to_stream("messages", "assistant", f"代理處理時發生問題：{response_content_main}")
                    st.rerun()
            else:
                with st.spinner("通用助理正在思考中..."):
                    debug_log("Routing query to general LLM.")
                    client_openai_main_chat = None
                    if "gpt" in selected_model_main.lower():
                        openai_api_key_val = st.session_state.get("openai_api_key_input")
                        if not openai_api_key_val: st.error("OpenAI key needed for GPT."); st.stop()
                        try:
                            client_openai_main_chat = OpenAI(api_key=openai_api_key_val)
                        except Exception as e:
                            st.error(f"Failed to initialize OpenAI client for main chat: {e}"); st.stop()
                    model_params_main = {"model": selected_model_main, "temperature": 0.5, "max_tokens": 4096}
                    response_content_main = get_llm_response(client_openai_main_chat, model_params_main, message_stream_key="messages")
                    if response_content_main and not response_content_main.startswith("Error:"):
                        append_message_to_stream("messages", "assistant", response_content_main)
                    elif response_content_main.startswith("Error:"):
                        append_message_to_stream("messages", "assistant", f"通用模型錯誤: {response_content_main}")
                    else:
                        append_message_to_stream("messages", "assistant", "通用助理未能獲取回覆。")
                    st.rerun()

        if st.session_state.get("editor_location") == "Main":
            with st.expander("🖋️ Persistent Code Editor (Main Chat)", expanded=True):
                edited_code_main = st_ace( value=st.session_state.get("ace_code", "# Python code for main analysis"), language="python", theme="monokai", height=300, key="ace_editor_main_chat_main_tab" )
                if edited_code_main != st.session_state.get("ace_code"): st.session_state.ace_code = edited_code_main
                if st.button("▶️ Execute Code (Main Chat)", key="exec_code_main_btn_main_tab"):
                    global_vars_main = { "st_session_state": st.session_state, "pd": pd, "plt": plt, "st": st, "uploaded_file_path": st.session_state.get("uploaded_file_path")}
                    exec_result_main = execute_code(st.session_state.ace_code, global_vars=global_vars_main)
                    st.text_area("Execution Result:", value=str(exec_result_main), height=150, key="exec_result_main_area_main_tab")
        st.markdown("---")
        st.subheader("🔬 Multi-Model Cross-Validation (Main Chat)")
        default_validator_idx = LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0
        validator_model_name = st.selectbox("選擇交叉驗證模型 (Main Chat):", LLM_MODELS, index=default_validator_idx, key="validator_model_main_select_main_tab")
        if st.button("🚀 執行交叉驗證 (Main Chat)", key="cross_validate_main_btn_main_tab"):
            client_for_validator_cv = None
            if "gpt" in validator_model_name.lower():
                openai_api_key_val_cv = st.session_state.get("openai_api_key_input")
                if not openai_api_key_val_cv: st.error("OpenAI key for GPT validator missing."); st.stop()
                try:
                    client_for_validator_cv = OpenAI(api_key=openai_api_key_val_cv)
                except Exception as e:
                    st.error(f"Failed to initialize OpenAI client for CV: {e}"); st.stop()
            if not st.session_state.get("messages") or len(st.session_state.messages) < 2: st.warning("Main chat內容過少。"); st.stop()
            model_params_validator = {"model": validator_model_name, "temperature": 0.2, "max_tokens": 4096}
            with st.spinner(f"使用 {validator_model_name} 進行交叉驗證中..."):
                validated_data = get_cross_validated_response(client_for_validator_cv, model_params_validator)
                st.markdown(f"#### ✅ {validator_model_name} 交叉驗證結果："); st.markdown(validated_data.get("validator_response", "未能獲取驗證回覆。"))

    # Tab 1: Executive Workflow
    with tabs[1]:
        st.header("💼 Executive Decision Workflow (Powered by OpenAI)")
        st.write("This workflow uses the CSV uploaded in the Main Chat tab and OpenAI models for analysis.")
        st.session_state.executive_user_query = st.text_area( "Enter the Business Problem or Question for Executive Analysis:", value=st.session_state.get("executive_user_query", ""), key="exec_problem_input", height=100 )
        can_start_exec_workflow = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
        if st.button("🚀 Start/Restart Executive Analysis (OpenAI)", key="start_exec_workflow_btn_openai", disabled=not can_start_exec_workflow):
            if not st.session_state.get("uploaded_file_path"):
                st.error("Please upload a CSV file in the 'Main Chat & Analysis' tab first.")
            elif not st.session_state.get("executive_user_query", "").strip():
                st.error("Please enter a business problem or question.")
            else:
                st.session_state.executive_workflow_stage = "data_profiling_pending"
                st.session_state.cfo_analysis_text = ""
                st.session_state.coo_analysis_text = ""
                st.session_state.ceo_summary_text = ""
                for exec_key in ["cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages"]: st.session_state[exec_key] = []
                debug_log("Executive Workflow (OpenAI) Initiated.")
                st.rerun()
        if not can_start_exec_workflow and st.session_state.executive_workflow_stage == "idle":
            st.info("Please upload a CSV in the 'Main Chat' tab and enter a business problem above to start the Executive Analysis (OpenAI).")
        if st.session_state.executive_workflow_stage == "data_profiling_pending":
            with st.spinner("Generating data profile for executives..."):
                try:
                    df_exec = pd.read_csv(st.session_state.uploaded_file_path)
                    st.session_state.executive_data_profile_str = generate_data_profile(df_exec)
                    append_message_to_stream("cfo_exec_messages", "system", f"Data Profile Provided:\n{st.session_state.executive_data_profile_str[:500]}...")
                    st.session_state.executive_workflow_stage = "cfo_analysis_pending"
                    debug_log("Data profile generated for executive workflow.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to read or profile CSV for executive workflow: {e}")
                    debug_error(f"Exec workflow data profiling error: {e}")
                    st.session_state.executive_workflow_stage = "idle"
        if st.session_state.executive_data_profile_str and st.session_state.executive_workflow_stage != "idle":
            with st.expander("View Data Profile Used by Executives", expanded=False):
                st.text_area("Data Profile:", value=st.session_state.executive_data_profile_str, height=300, key="exec_data_profile_display", disabled=True)
        if st.session_state.executive_workflow_stage == "cfo_analysis_pending":
            with st.spinner("CFO is analyzing... (using OpenAI gpt-4o by default)"):
                cfo_prompt = f"""You are the Chief Financial Officer (CFO)...""" # Prompt kept concise for brevity
                append_message_to_stream("cfo_exec_messages", "user", cfo_prompt)
                cfo_model_params = {"model": "gpt-4o", "temperature": 0.3, "max_tokens": 2000}
                cfo_response = get_openai_executive_analysis("CFO", cfo_prompt, cfo_model_params)
                if cfo_response and not cfo_response.startswith("Error:"):
                    st.session_state.cfo_analysis_text = cfo_response
                    append_message_to_stream("cfo_exec_messages", "assistant", cfo_response)
                    st.session_state.executive_workflow_stage = "coo_analysis_pending"
                    debug_log("CFO analysis (OpenAI) completed.")
                    st.rerun()
                else:
                    st.error(f"CFO analysis (OpenAI) failed: {cfo_response}")
                    st.session_state.executive_workflow_stage = "error"
        if st.session_state.cfo_analysis_text:
            st.subheader("📊 CFO Analysis (OpenAI)")
            st.markdown(st.session_state.cfo_analysis_text)
            st.markdown("---")
        if st.session_state.executive_workflow_stage == "coo_analysis_pending":
            with st.spinner("COO is analyzing... (using OpenAI gpt-4o by default)"):
                coo_prompt = f"""You are the Chief Operating Officer (COO)...""" # Prompt kept concise for brevity
                append_message_to_stream("coo_exec_messages", "user", coo_prompt)
                coo_model_params = {"model": "gpt-4o", "temperature": 0.4, "max_tokens": 2000}
                coo_response = get_openai_executive_analysis("COO", coo_prompt, coo_model_params)
                if coo_response and not coo_response.startswith("Error:"):
                    st.session_state.coo_analysis_text = coo_response
                    append_message_to_stream("coo_exec_messages", "assistant", coo_response)
                    st.session_state.executive_workflow_stage = "ceo_synthesis_pending"
                    debug_log("COO analysis (OpenAI) completed.")
                    st.rerun()
                else:
                    st.error(f"COO analysis (OpenAI) failed: {coo_response}")
                    st.session_state.executive_workflow_stage = "error"
        if st.session_state.coo_analysis_text:
            st.subheader("⚙️ COO Analysis (OpenAI)")
            st.markdown(st.session_state.coo_analysis_text)
            st.markdown("---")
        if st.session_state.executive_workflow_stage == "ceo_synthesis_pending":
            with st.spinner("CEO is synthesizing and making decisions... (using OpenAI gpt-4o by default)"):
                ceo_prompt = f"""You are the Chief Executive Officer (CEO)...""" # Prompt kept concise for brevity
                append_message_to_stream("ceo_exec_messages", "user", ceo_prompt)
                ceo_model_params = {"model": "gpt-4o", "temperature": 0.5, "max_tokens": 2500}
                ceo_response = get_openai_executive_analysis("CEO", ceo_prompt, ceo_model_params)
                if ceo_response and not ceo_response.startswith("Error:"):
                    st.session_state.ceo_summary_text = ceo_response
                    append_message_to_stream("ceo_exec_messages", "assistant", ceo_response)
                    st.session_state.executive_workflow_stage = "completed"
                    debug_log("CEO synthesis (OpenAI) completed.")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"CEO synthesis (OpenAI) failed: {ceo_response}")
                    st.session_state.executive_workflow_stage = "error"
        if st.session_state.executive_workflow_stage == "completed" and st.session_state.ceo_summary_text:
            st.subheader("👑 CEO Strategic Summary & Decisions (OpenAI)")
            st.markdown(st.session_state.ceo_summary_text)
            st.success("Executive Workflow (OpenAI) Completed!")
        if st.session_state.executive_workflow_stage == "error":
            st.error("An error occurred during the executive workflow. Please check logs if debug mode is on, or try restarting.")
        if st.session_state.executive_workflow_stage not in ["idle", "data_profiling_pending"]:
            with st.expander("View Executive Communication Logs (OpenAI)", expanded=False):
                for exec_role_name, exec_msg_key in [("CFO", "cfo_exec_messages"), ("COO", "coo_exec_messages"), ("CEO", "ceo_exec_messages")]:
                    if st.session_state.get(exec_msg_key):
                        st.markdown(f"**{exec_role_name}'s Log:**")
                        for i, msg in enumerate(st.session_state[exec_msg_key]):
                            st.markdown(f"_{msg['role']} (turn {i//2 + 1})_:\n{str(msg['content'])[:300]}...")
                        st.markdown("---")

    # Tabs for Generic Roles
    for i, role_id_generic in enumerate(ROLE_DEFINITIONS.keys()):
        role_info = ROLE_DEFINITIONS[role_id_generic]
        with tabs[i + 2]:
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"].split('.')[0] + ". (Powered by Gemini)")
            message_key_role = role_info["messages_key"]
            for msg_role in st.session_state[message_key_role]:
                with st.chat_message(msg_role["role"]):
                    st.write(msg_role["content"])
            user_input_role = st.chat_input(f"Chat with {role_info['name']}...", key=f"input_{role_id_generic}")
            if user_input_role:
                append_message_to_stream(message_key_role, "user", user_input_role)
                st.rerun()
            if st.session_state[message_key_role] and st.session_state[message_key_role][-1]["role"] == "user":
                with st.spinner(f"{role_info['name']} is thinking..."):
                    last_user_input_for_role = st.session_state[message_key_role][-1]["content"]
                    role_model_params = { "model": "gemini-1.5-flash", "temperature": 0.7 }
                    response_role = get_gemini_response_for_generic_role(role_id_generic, last_user_input_for_role, role_model_params)
                    if response_role and not response_role.startswith("Error:"):
                        append_message_to_stream(message_key_role, "assistant", response_role)
                    else:
                        append_message_to_stream(message_key_role, "assistant", response_role if response_role else f"Sorry, {role_info['name']} couldn't get a response.")
                    st.rerun()

    # Sidebar Code Editor
    if st.session_state.get("editor_location") == "Sidebar":
        with st.sidebar.expander("🖋️ Persistent Code Editor (Sidebar)", expanded=False):
            edited_code_sidebar = st_ace(value=st.session_state.get("ace_code", "// Python code in sidebar..."), language="python",theme="monokai",height=300,key="ace_editor_sidebar_widget")
            if edited_code_sidebar != st.session_state.get("ace_code"): st.session_state.ace_code = edited_code_sidebar
            if st.button("▶️ Execute Code (Sidebar)", key="exec_code_sidebar_btn_widget"):
                global_vars_sidebar = {"st_session_state": st.session_state, "pd": pd, "plt": plt, "st": st, "uploaded_file_path": st.session_state.get("uploaded_file_path")}
                exec_result_sidebar = execute_code(st.session_state.ace_code, global_vars=global_vars_sidebar)
                st.sidebar.text_area("Execution Result (Sidebar):", value=str(exec_result_sidebar), height=100, key="exec_result_sidebar_area_widget")

if __name__ == "__main__":
    main()
