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
# from openai import OpenAI # OpenAI parts will be kept but Gemini is the focus for new roles
from PIL import Image
import google.generativeai as genai
from streamlit_ace import st_ace
import time
import matplotlib.font_manager as fm
import matplotlib
import sys # For capturing print output from exec

# --- Role Definitions (Generic Chat Roles) ---
ROLE_DEFINITIONS = {
    "summarizer": {
        "name": "📝 Summarizer",
        "system_prompt": "You are an expert summarizer. Your task is to take any text or conversation provided and condense it into a clear, concise summary in Traditional Chinese. Focus on the main points and key takeaways. Respond with #zh-tw.",
        "messages_key": "summarizer_messages",
        "chat_session_key": "summarizer_chat_session", # For Gemini ChatSession
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
font_path = "./fonts/msjh.ttc" # Make sure this file exists

# 加入字型並設定為預設字型
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
    "claude-3-opus-20240229", # Switched to official model names
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

MAX_MESSAGES_PER_STREAM = 12  # Increased slightly


def debug_log(msg):
    # ... (existing function)
    if st.session_state.get("debug_mode", False):
        if "debug_logs" not in st.session_state: # Ensure list exists
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"**DEBUG LOG ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    # ... (existing function)
    if st.session_state.get("debug_mode", False):
        if "debug_errors" not in st.session_state: # Ensure list exists
            st.session_state.debug_errors = []
        st.session_state.debug_errors.append(f"**DEBUG ERROR ({time.strftime('%H:%M:%S')}):** {msg}")
        print(f"DEBUG ERROR: {msg}")

def save_uploaded_file(uploaded_file):
    # ... (existing function)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image_pil): # Take PIL image as input
    # ... (existing function)
    try:
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")  # Use PNG for consistency
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message_to_stream(message_stream_key, role, content, max_messages=MAX_MESSAGES_PER_STREAM):
    # ... (existing function, slightly improved trimming)
    if message_stream_key not in st.session_state:
        st.session_state[message_stream_key] = []

    st.session_state[message_stream_key].append({"role": role, "content": content})

    current_messages = st.session_state[message_stream_key]
    if len(current_messages) > max_messages:
        # Simple trim: keep the last 'max_messages' items.
        # If a system prompt is the first, this might remove it.
        # For roles with system_instruction in Gemini, this is less of an issue for the API call itself.
        st.session_state[message_stream_key] = current_messages[-max_messages:]
        debug_log(f"Message history for '{message_stream_key}' trimmed to {max_messages} messages. Count: {len(st.session_state[message_stream_key])}")


def add_user_image_to_main_chat(uploaded_file):
    # ... (existing function)
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        use_base64_for_gpt = "gpt" in current_model

        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path 
        image_pil = Image.open(file_path)

        if use_base64_for_gpt:
            image_base64 = load_image_base64(image_pil)
            if not image_base64: st.error("圖片轉為Base64失敗。"); return
            image_url = f"data:image/{file_path.split('.')[-1]};base64,{image_base64}"
            image_msg_content = [{"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}]
            append_message_to_stream("messages", "user", image_msg_content) 
            st.session_state.image_base64 = image_base64 
        else: 
            st.session_state.pending_image_for_main_gemini = image_pil
            st.image(image_pil, caption="圖片已上傳，將隨下一條文字訊息發送 (主聊天室)。", use_container_width=True)

        debug_log(f"圖片已處理 (主聊天室): {file_path}. Base64 for GPT: {use_base64_for_gpt}")
        if use_base64_for_gpt: st.rerun()

    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        st.error("圖片處理異常，請檢查日誌")
        debug_error(f"Error in add_user_image_to_main_chat: {e}, Traceback: {traceback.format_exc()}")


def reset_session_messages(message_stream_key="messages"):
    # ... (existing function, adapted for generic roles too)
    if message_stream_key in st.session_state:
        st.session_state.pop(message_stream_key)
        debug_log(f"Conversation history for '{message_stream_key}' cleared.")
    
    # Clear related chat session for generic roles
    for role_id_iter, P_info in ROLE_DEFINITIONS.items(): # Renamed role_id to role_id_iter
        if P_info["messages_key"] == message_stream_key and P_info["chat_session_key"] in st.session_state:
            del st.session_state[P_info["chat_session_key"]]
            debug_log(f"Gemini chat session for role '{role_id_iter}' cleared.")
    st.success(f"Memory for '{message_stream_key}' cleared!")


def execute_code(code, global_vars=None):
    # ... (existing function from previous version - ensure it's robust)
    try:
        exec_globals = global_vars if global_vars else {}
        exec_globals.update({'pd': pd, 'plt': plt, 'st': st, 'np': __import__('numpy')})
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(code, exec_globals)
        
        sys.stdout = old_stdout # Restore stdout
        
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
        return error_msg # Always return full error

def extract_json_block(response: str) -> str:
    # ... (existing function)
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip()

# --- Gemini Interaction for Generic ROLES (Summarizer, Code Explainer, etc.) ---
def get_gemini_response_for_generic_role(role_id, user_input_text, model_params, max_retries=3):
    # ... (Function from previous response: get_gemini_response_for_role)
    # Renamed for clarity to distinguish from executive roles if logic differs.
    # This function uses ChatSession and its history.
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
        gemini_model_instance = genai.GenerativeModel(
            model_name,
            system_instruction=role_info["system_prompt"]
        )
    except Exception as e:
        st.error(f"無法初始化 Gemini 模型 '{model_name}': {e}")
        debug_error(f"Gemini model init error for {model_name}: {e}")
        return f"Error: Could not initialize Gemini model {model_name}."

    gemini_history = []
    if messages_key in st.session_state:
        for msg in st.session_state[messages_key]: # Use only user/assistant for history
            if msg["role"] == "user":
                gemini_history.append({'role': 'user', 'parts': [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_history.append({'role': 'model', 'parts': [msg["content"]]})
    
    # Initialize or get existing chat session
    if chat_session_key not in st.session_state or \
       st.session_state[chat_session_key].model_name != f"models/{model_name}" or \
       len(st.session_state[chat_session_key].history) != len(gemini_history) : # Basic check if history might have diverged
        debug_log(f"Initializing/Resetting Gemini ChatSession for generic role '{role_id}' with model '{model_name}'. History length: {len(gemini_history)}")
        st.session_state[chat_session_key] = gemini_model_instance.start_chat(history=gemini_history)
    
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
            debug_error(f"Gemini API error for generic role '{role_id}' (attempt {retries + 1}): {e}")
            # ... (error handling from previous version) ...
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
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

# --- Gemini Interaction for EXECUTIVE WORKFLOW (New Function) ---
def get_gemini_executive_analysis(executive_role_name, full_prompt, model_params, max_retries=3):
    """
    Handles a single turn Gemini API request for an executive role in the workflow.
    This does not use ChatSession history explicitly, as each turn gets a full context.
    """
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key:
        st.error("未設定 Gemini API 金鑰，請於側邊欄設定。")
        return "Error: Missing Gemini API Key."
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini API 金鑰設定失敗: {e}")
        return f"Error: Gemini API key configuration failed: {e}"

    model_name = model_params.get("model", "gemini-1.5-pro") # Executives might need more capable model
    
    try:
        # For single, comprehensive prompts, system_instruction can still be useful.
        # Or, incorporate the "You are CFO/COO/CEO" into the full_prompt itself.
        # Let's assume the full_prompt already contains the role instruction.
        model_instance = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"無法初始化 Gemini 模型 '{model_name}' for {executive_role_name}: {e}")
        return f"Error: Could not initialize Gemini model {model_name}."

    debug_log(f"Sending prompt to Gemini for Executive '{executive_role_name}': \n{full_prompt[:300]}...") # Log beginning of prompt

    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            response = model_instance.generate_content(full_prompt) # Single turn generation
            final_reply = response.text.strip()
            debug_log(f"Gemini response for Executive '{executive_role_name}': {final_reply[:200]}...")
            return final_reply
        except Exception as e:
            debug_error(f"Gemini API error for Executive '{executive_role_name}' (attempt {retries + 1}): {e}")
            # ... (similar error handling as other Gemini functions) ...
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
                 st.error(f"Gemini API 金鑰無效或權限不足: {e}")
                 return "Error: Gemini API Key Invalid or Permission Denied."
            if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"Gemini 請求頻繁或資源耗盡，{wait_time}秒後重試...")
                time.sleep(wait_time); retries += 1; wait_time *= 2
            else:
                st.error(f"Gemini API 請求發生未預期錯誤 for {executive_role_name}: {e}")
                return f"Error: An unexpected error with Gemini API for {executive_role_name}: {e}"
    
    st.error(f"Gemini for Executive '{executive_role_name}': 請求失敗次數過多。")
    return f"Error: Max retries exceeded for Gemini ({executive_role_name})."


# --- Main Chat LLM Response Functions (Existing, slightly adapted for clarity) ---
def get_gemini_response_main_chat(model_params, max_retries=3):
    # ... (existing function from previous response, ensure it's robust) ...
    # This should be largely okay but verify image and history handling.
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key: st.error("未設定 Gemini API 金鑰"); return ""
    try: genai.configure(api_key=api_key)
    except Exception as e: st.error(f"Gemini API key config error: {e}"); return ""

    model_name = model_params.get("model", "gemini-1.5-flash")
    system_instruction_main = model_params.get("system_prompt", "請以繁體中文回答。所有回覆開頭請加上 #zh-tw 標籤。") # Allow custom system prompt
    
    try: model_instance_main = genai.GenerativeModel(model_name, system_instruction=system_instruction_main)
    except Exception as e: st.error(f"Gemini Main Chat Model Init Error '{model_name}': {e}"); return ""

    # Reconstruct history for main chat for ChatSession
    gemini_history_main = []
    main_chat_messages = st.session_state.get("messages", [])
    
    # The last message in main_chat_messages is the current user input if just added.
    # History for ChatSession should exclude the very last user message that's about to be sent.
    num_messages_for_history = len(main_chat_messages)
    if main_chat_messages and main_chat_messages[-1]["role"] == "user" and \
       ("pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini is not None \
        or not ("pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini is not None)): # if it's a text-only new message
        num_messages_for_history -=1


    for i in range(num_messages_for_history):
        msg = main_chat_messages[i]
        role = "model" if msg["role"] == "assistant" else msg["role"]
        if role not in ["user", "model"]: continue
        
        content_parts_main = []
        if isinstance(msg["content"], list): # GPT vision format in history
            text_found = False
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_parts_main.append(item["text"])
                    text_found = True
                # Skipping images from history for simplicity, current turn image handled below
            if not text_found: content_parts_main.append("") # Gemini requires some text part if list
        else:
            content_parts_main.append(str(msg["content"]))
        if content_parts_main:
             gemini_history_main.append({'role': role, 'parts': content_parts_main})


    chat_session_main_key = "main_chat_gemini_session"
    # Re-start chat session if model changes or history seems to diverge (simple check)
    if chat_session_main_key not in st.session_state or \
       st.session_state[chat_session_main_key].model_name != f"models/{model_name}" or \
       len(st.session_state[chat_session_main_key].history) != len(gemini_history_main): # Basic check
        debug_log(f"Initializing/Resetting Gemini ChatSession for Main Chat with model '{model_name}'. History length: {len(gemini_history_main)}")
        st.session_state[chat_session_main_key] = model_instance_main.start_chat(history=gemini_history_main)
    
    chat_main = st.session_state[chat_session_main_key]

    # Prepare current turn message parts
    current_turn_parts_main = []
    if main_chat_messages and main_chat_messages[-1]["role"] == "user":
        last_user_msg_content = main_chat_messages[-1]["content"]
        if isinstance(last_user_msg_content, list): # GPT vision format
            for item in last_user_msg_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    current_turn_parts_main.append(item["text"])
                    break # Assuming one text part
            else: # No text part, add empty for Gemini if image is present
                 if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
                     current_turn_parts_main.append("")
        else:
            current_turn_parts_main.append(str(last_user_msg_content))
    
    if not current_turn_parts_main and not ("pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini):
        debug_error("Main Chat Gemini: No text input for current turn.")
        # This can happen if an image was uploaded but no subsequent text was entered by user yet to trigger send
        # Or if the last message was assistant. The calling logic should ensure user input is present.
        return "Error: No user input to send to Gemini main chat."


    if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
        current_turn_parts_main.append(st.session_state.pending_image_for_main_gemini)
        debug_log("Attaching pending_image_for_main_gemini to current Gemini Main Chat call.")
        # Consume after successful send or clear on error to avoid resending.
        # For now, let the caller manage clearing pending_image_for_main_gemini after response.

    debug_log(f"Gemini (Main Chat) sending parts for current turn: {[(type(p), str(p)[:50] + '...' if isinstance(p,str) else '') for p in current_turn_parts_main]}")
    if not current_turn_parts_main: # Final check
        return "Error: No content to send for current turn in main Gemini chat."

    retries = 0; wait_time = 5
    while retries < max_retries:
        try:
            response = chat_main.send_message(current_turn_parts_main)
            if "pending_image_for_main_gemini" in st.session_state: # Consume image after attempt
                del st.session_state.pending_image_for_main_gemini
            final_reply = response.text.strip()
            debug_log(f"Gemini (Main Chat) send_message final reply => {final_reply}")
            return final_reply
        except Exception as e:
            if "pending_image_for_main_gemini" in st.session_state: # Consume image after attempt
                del st.session_state.pending_image_for_main_gemini
            debug_error(f"Gemini (Main Chat) API error (attempt {retries + 1}): {e}, Parts: {current_turn_parts_main}")
            # ... (error handling logic from previous version) ...
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
                 st.error(f"Gemini API 金鑰無效或權限不足: {e}"); return "Error: Key Invalid/Permission Denied."
            if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"Gemini (Main) 請求頻繁，{wait_time}秒後重試..."); time.sleep(wait_time); retries += 1; wait_time *= 2
            else:
                st.error(f"Gemini (Main) API 未預期錯誤: {e}"); return f"Error: Unexpected Gemini API error: {e}"
    st.error("Gemini (Main Chat): 請求失敗次數過多"); return "Error: Max retries exceeded."

def get_openai_response(client, model_params, max_retries=3):
    # ... (existing function, ensure client is passed correctly if GPT is chosen) ...
    # This function uses st.session_state.messages for main chat
    if not client:
        openai_api_key_curr = st.session_state.get("openai_api_key_input") or os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key_curr: st.error("OpenAI API Key not set."); return ""
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key_curr)
    # ... (rest of the function from previous version, ensure processed_messages_for_openai uses "messages") ...
    processed_messages_for_openai = [] # From st.session_state.messages
    main_chat_msg_list = st.session_state.get("messages", [])
    for msg in main_chat_msg_list: # Explicitly use main chat messages
        if isinstance(msg["content"], list):
            new_content_list = []; has_text = False
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url": new_content_list.append(item)
                elif isinstance(item, dict) and item.get("type") == "text": new_content_list.append(item); has_text = True
                elif isinstance(item, str): new_content_list.append({"type": "text", "text": item}); has_text = True
            # Ensure text part if image is present (OpenAI might require this for some models or message structures)
            if any(item.get("type") == "image_url" for item in new_content_list) and not has_text:
                 new_content_list.append({"type": "text", "text": " "}) # Add a minimal text part
            processed_messages_for_openai.append({"role": msg["role"], "content": new_content_list})
        else:
            processed_messages_for_openai.append(msg) # Assumes msg is already in correct OpenAI format if not list

    model_name = model_params.get("model", "gpt-4o")
    retries = 0; wait_time = 5
    while retries < max_retries:
        try:
            request_params = {
                "model": model_name, "messages": processed_messages_for_openai,
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096), "stream": False
            }
            response = client.chat.completions.create(**request_params)
            return response.choices[0].message.content.strip()
        except Exception as e: # ... (OpenAI error handling from before) ...
            if 'rate limit' in str(e).lower() or '429' in str(e): time.sleep(wait_time); retries += 1; wait_time *= 2
            elif 'invalid api key' in str(e).lower(): st.error("OpenAI API金鑰無效"); return ""
            else: st.error(f"OpenAI請求錯誤：{e}"); return ""
    st.error("OpenAI請求失敗次數過多"); return ""

def get_claude_response(model_params, max_retries=3):
    # ... (existing function, ensure it uses st.session_state.messages for main chat) ...
    # This function uses st.session_state.messages for main chat
    import anthropic 
    api_key = st.session_state.get("claude_api_key_input") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key: st.error("未設定 Claude API 金鑰"); return ""
    try: client = anthropic.Anthropic(api_key=api_key)
    except Exception as e: st.error(f"Claude Client Init Error: {e}"); return ""
    
    model_name = model_params.get("model", "claude-3-opus-20240229")
    # ... (rest of Claude message prep and API call logic from previous version, ensure source_messages is "messages") ...
    claude_messages = []; system_prompt_claude = None
    source_messages = st.session_state.get("messages", []) # Use main chat messages
    for i, msg in enumerate(source_messages):
        if msg["role"] == "system" and i == 0: system_prompt_claude = msg["content"]; continue
        # ... (Claude message conversion logic from previous version)
        if isinstance(msg["content"], list): 
            content_parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    if img_url.startswith("data:image"):
                        base64_data=img_url.split(",")[1]; media_type=img_url.split(";")[0].split(":")[1]
                        content_parts.append({"type":"image","source":{"type":"base64","media_type":media_type,"data":base64_data}})
                elif isinstance(item,dict) and item.get("type")=="text": content_parts.append({"type":"text","text":item["text"]})
                elif isinstance(item,str): content_parts.append({"type":"text","text":item})
            if content_parts: claude_messages.append({"role":msg["role"],"content":content_parts})
        else: claude_messages.append({"role":msg["role"],"content":str(msg["content"])})
    
    retries = 0; wait_time = 5
    while retries < max_retries:
        try:
            api_params={"model":model_name,"max_tokens":model_params.get("max_tokens",8192),"messages":claude_messages,"temperature":model_params.get("temperature",0.3)}
            if system_prompt_claude: api_params["system"]=system_prompt_claude
            response=client.messages.create(**api_params)
            completion = "".join([block.text for block in response.content if block.type=="text"])
            return completion.strip()
        except Exception as e: # ... (Claude error handling from before) ...
            if "rate limit" in str(e).lower() or "429" in str(e): time.sleep(wait_time); retries += 1; wait_time *= 2
            elif "authentication_error" in str(e).lower(): st.error(f"Claude API金鑰無效: {e}"); return ""
            else: st.warning(f"Claude 生成錯誤，{wait_time}秒後重試..."); time.sleep(wait_time); retries += 1; wait_time *= 2;
            if retries >=max_retries: st.error(f"Claude 請求失敗過多: {e}"); return ""
    return ""


def get_llm_response(client_openai, model_params, message_stream_key="messages", max_retries=3): # client_openai is specific
    # ... (existing function, routing logic)
    model_name = model_params.get("model", "gemini-1.5-flash") # Default to Gemini
    debug_log(f"Routing LLM request for model: {model_name}, stream: {message_stream_key}")

    if "gpt" in model_name.lower():
        return get_openai_response(client_openai, model_params, max_retries)
    elif "gemini" in model_name.lower():
        # This assumes it's for the main chat if message_stream_key is "messages"
        # Role-specific Gemini calls are handled by get_gemini_response_for_generic_role or executive
        if message_stream_key == "messages":
            return get_gemini_response_main_chat(model_params, max_retries)
        else: # Should not happen if generic roles use their own function
            st.error("Gemini routing error in get_llm_response for non-main stream.")
            return "Error: Gemini routing issue."
    elif "claude" in model_name.lower():
        return get_claude_response(model_params, max_retries)
    else:
        st.error(f"Unsupported model type in get_llm_response: {model_name}")
        return ""


def get_cross_validated_response(client_openai, model_params_validator, max_retries=3):
    # ... (existing function, ensure client_openai is used if validator is GPT)
    # ... and that it correctly calls the right LLM provider for validator_model_name
    cross_validation_prompt_content = (
        "請仔細閱讀以下全部對話記憶 (來自主要聊天室)，對先前模型的回答進行交叉驗證。"
        "你的任務是檢查回答的正確性，指出其中可能存在的錯誤或不足，"
        "並提供具體的數據、理由或例子來支持你的分析。"
        "請務必使用繁體中文回答。"
        "在回答時請回答的詳細，內容需要你盡可能的多。"
    )
    original_main_messages = list(st.session_state.get("messages", []))
    
    # Temporarily construct messages for validator, including the system prompt
    validator_messages_with_system_prompt = [{"role": "system", "content": cross_validation_prompt_content}] + original_main_messages

    validator_model_name = model_params_validator.get('model')
    validator_response_text = ""
    debug_log(f"Cross-validation: Validating with {validator_model_name} using main chat history.")

    # Create a temporary st.session_state.messages for the validator call if needed by sub-functions
    # This is a bit of a hack; ideally, sub-functions would take messages as an argument.
    original_st_messages = st.session_state.get("messages")
    st.session_state.messages = validator_messages_with_system_prompt # Temporarily set for the call

    if "gpt" in validator_model_name.lower():
        openai_api_key_val = st.session_state.get("openai_api_key_input")
        if not openai_api_key_val: st.error("OpenAI key for GPT validator missing."); st.session_state.messages = original_st_messages; return {"validator_response": "Error: OpenAI Key missing."}
        from openai import OpenAI
        temp_openai_client = OpenAI(api_key=openai_api_key_val)
        validator_response_text = get_openai_response(temp_openai_client, model_params_validator, max_retries)
    elif "gemini" in validator_model_name.lower():
        # get_gemini_response_main_chat uses st.session_state.messages, which now has the system prompt at the start
        # and it handles its own system_instruction, so this might double-prompt or override.
        # For cross-val, it might be better to use a single generate_content with the full history.
        # Let's adapt to a single generate_content for validator to ensure prompt is used.
        api_key = st.session_state.get("gemini_api_key_input", "")
        if not api_key: st.error("Gemini key for validator missing."); st.session_state.messages = original_st_messages; return {"validator_response": "Error: Gemini Key missing."}
        genai.configure(api_key=api_key)
        val_model_gem = genai.GenerativeModel(validator_model_name) # No system_instruction here, it's in validator_messages_with_system_prompt
        
        # Convert validator_messages_with_system_prompt to Gemini's content format
        gemini_val_contents = []
        for msg in validator_messages_with_system_prompt:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            if role not in ["user", "model", "system"]: continue # System handled if first
            # System prompt as first user message if not directly supported by generate_content in this way
            if msg["role"] == "system": # Gemini doesn't use "system" role in contents like this.
                                        # We'll prepend it to the first user message or treat it as a user turn.
                                        # Or, if using system_instruction with GenerativeModel, it's handled there.
                                        # For now, let's use the system_instruction of the model.
                pass # System prompt already added if we are passing a 'system' role.
                     # For generate_content, system prompt is usually a separate param or part of model config.
                     # The cross_validation_prompt_content should be the primary content.
            else:
                 gemini_val_contents.append({'role': role, 'parts': [str(msg['content'])]}) # Simplified content
        
        # The validator_messages_with_system_prompt already has the system instruction as the first message.
        # For generate_content, the first message is often taken as the primary instruction.
        try:
            # Combine all messages into a single prompt for generate_content
            # The system prompt is the first message.
            # This is a simplified way for a single turn validation.
            full_validation_prompt_for_gemini = "\n\n".join([f"{m['role']}: {m['content']}" for m in validator_messages_with_system_prompt])
            # Or, more correctly for Gemini generate_content:
            gemini_formatted_val_history = []
            sys_instr_val = None
            if validator_messages_with_system_prompt[0]['role'] == 'system':
                sys_instr_val = validator_messages_with_system_prompt[0]['content']
                # val_model_gem = genai.GenerativeModel(validator_model_name, system_instruction=sys_instr_val) # Re-init with SI
            
            for m_val in validator_messages_with_system_prompt:
                if m_val['role'] == 'system' and sys_instr_val: continue # Handled by system_instruction
                role_val = 'model' if m_val['role'] == 'assistant' else m_val['role']
                if role_val not in ['user', 'model']: continue
                gemini_formatted_val_history.append({'role': role_val, 'parts': [str(m_val['content'])]}) # Keep it simple text for val

            val_model_gem = genai.GenerativeModel(validator_model_name, system_instruction=sys_instr_val)
            response_val = val_model_gem.generate_content(gemini_formatted_val_history)
            validator_response_text = response_val.text.strip()
        except Exception as e_gem_val:
            validator_response_text = f"Error during Gemini validation: {e_gem_val}"
            debug_error(validator_response_text)


    elif "claude" in validator_model_name.lower():
        # get_claude_response uses st.session_state.messages
        validator_response_text = get_claude_response(model_params_validator, max_retries)
    else:
        st.error(f"Unsupported validator model: {validator_model_name}")
        validator_response_text = "Error: Unsupported validator model."

    st.session_state.messages = original_st_messages # Restore main messages
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
# 主應用入口
# ------------------------------
def main():
    st.set_page_config(page_title="Multi-Role & Exec Workflow Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Multi-Role & 📈 Executive Workflow Chatbot")

    # --- Initialize Session States ---
    if "messages" not in st.session_state: st.session_state.messages = []
    for role_id, role_info in ROLE_DEFINITIONS.items(): # Generic roles
        if role_info["messages_key"] not in st.session_state: st.session_state[role_info["messages_key"]] = []
        if role_info["chat_session_key"] not in st.session_state: st.session_state[role_info["chat_session_key"]] = None
    
    # Executive Workflow States
    if "executive_workflow_stage" not in st.session_state: st.session_state.executive_workflow_stage = "idle" # idle, cfo_pending, coo_pending, ceo_pending, completed
    if "executive_user_query" not in st.session_state: st.session_state.executive_user_query = ""
    if "executive_data_profile_str" not in st.session_state: st.session_state.executive_data_profile_str = ""
    if "cfo_analysis_text" not in st.session_state: st.session_state.cfo_analysis_text = ""
    if "coo_analysis_text" not in st.session_state: st.session_state.coo_analysis_text = ""
    if "ceo_summary_text" not in st.session_state: st.session_state.ceo_summary_text = ""
    for exec_id_key in ["cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages"]:
        if exec_id_key not in st.session_state: st.session_state[exec_id_key] = []


    # ... other initializations from your original main ...
    if "ace_code" not in st.session_state: st.session_state.ace_code = ""
    if "editor_location" not in st.session_state: st.session_state.editor_location = "Sidebar"
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state: st.session_state.uploaded_image_path = None 
    if "image_base64" not in st.session_state: st.session_state.image_base64 = None 
    if "pending_image_for_main_gemini" not in st.session_state: st.session_state.pending_image_for_main_gemini = None
    if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state: st.session_state.deep_analysis_mode = True
    # ... (rest of your session state initializations)

    # --- Sidebar ---
    with st.sidebar:
        # ... (API Key inputs - good as is, ensure they store to st.session_state.xxx_api_key_input) ...
        st.subheader("🔑 API Key Settings")
        openai_api_key_input = st.text_input("OpenAI API Key", value=st.session_state.get("openai_api_key_input",""), type="password", key="openai_api_key_widget")
        if openai_api_key_input: st.session_state.openai_api_key_input = openai_api_key_input
        
        gemini_api_key_input = st.text_input("Gemini API Key", value=st.session_state.get("gemini_api_key_input",""), type="password", key="gemini_api_key_widget")
        if gemini_api_key_input: st.session_state.gemini_api_key_input = gemini_api_key_input

        claude_api_key_input = st.text_input("Claude API Key", value=st.session_state.get("claude_api_key_input",""), type="password", key="claude_api_key_widget")
        if claude_api_key_input: st.session_state.claude_api_key_input = claude_api_key_input

        st.subheader("⚙️ Main Chat Model")
        # ... (selected_model_main - good as is) ...
        selected_model_main = st.selectbox(
            "選擇主要聊天模型:", LLM_MODELS, 
            index=LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0, 
            key="selected_model_main"
        )
        st.session_state.selected_model = selected_model_main

        st.subheader("🛠️ Tools & Settings")
        # ... (debug_mode, deep_analysis_mode checkboxes - good as is) ...
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.get("debug_mode", False))
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode (Main Chat)", value=st.session_state.get("deep_analysis_mode", True))


        if st.button("🗑️ Clear ALL Memory & Chats"):
            # ... (Clear main chat messages, ROLE_DEFINITIONS messages & chat_sessions) ...
            st.session_state.messages = []
            if "main_chat_gemini_session" in st.session_state: del st.session_state.main_chat_gemini_session
            for role_id_iter, P_info in ROLE_DEFINITIONS.items(): # Renamed role_id to role_id_iter
                if P_info["messages_key"] in st.session_state: st.session_state[P_info["messages_key"]] = []
                if P_info["chat_session_key"] in st.session_state: del st.session_state[P_info["chat_session_key"]]
            
            # Clear Executive Workflow states
            st.session_state.executive_workflow_stage = "idle"
            st.session_state.executive_user_query = ""
            st.session_state.executive_data_profile_str = ""
            st.session_state.cfo_analysis_text = ""
            st.session_state.coo_analysis_text = ""
            st.session_state.ceo_summary_text = ""
            for exec_key in ["cfo_exec_messages", "coo_exec_messages", "ceo_exec_messages"]:
                st.session_state[exec_key] = []

            keys_to_clear = ["ace_code", "uploaded_file_path", "uploaded_image_path", 
                             "image_base64", "pending_image_for_main_gemini", "second_response", "third_response",
                             "deep_analysis_image", "thinking_protocol", "debug_logs", "debug_errors"]
            for key in keys_to_clear:
                if key in st.session_state: st.session_state.pop(key)
            st.success("All memories and chat states cleared!")
            debug_log("All memory cleared.")
            st.rerun()

        # ... (Rest of sidebar from previous version: Main Chat Memory, Uploads, Editor, Debug Info)
        st.subheader("🧠 Main Chat Memory State") # ...
        if st.session_state.get("messages"):
            memory_content = "\n".join([f"{('' if msg['role']=='system' else msg['role']+': ')}{str(msg['content'])[:100]+'...' if isinstance(msg['content'], str) and len(msg['content']) > 100 else str(msg['content'])}" for msg in st.session_state.messages])
            st.text_area("Current Main Chat Memory", value=memory_content, height=150, key="main_chat_memory_display_sidebar")
        else:
            st.text_area("Current Main Chat Memory", value="No messages yet in main chat.", height=150)

        st.subheader("📂 Upload CSV (For Main Chat & Executive Workflow)") # Clarified use
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"], key="main_csv_uploader_sidebar")
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            # ... (CSV preview logic)
            try:
                # Create a temp df for preview without storing large df in session_state unless needed
                df_preview = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### CSV Data Preview (First 5 rows)")
                st.dataframe(df_preview.head())
            except Exception as e:
                st.error(f"Error reading CSV for preview: {e}")


        st.subheader("🖼️ Upload Image (Main Chat)") # ...
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="main_image_uploader_sidebar")
        if uploaded_image:
            if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
                 st.warning("已有圖片待發送。請先發送包含該圖片的文字訊息，或清除記憶。")
            else:
                add_user_image_to_main_chat(uploaded_image)


        st.subheader("Editor Location") # ...
        location = st.radio( "Choose where to display the editor:", ["Main", "Sidebar"],
            index=1 if st.session_state.get("editor_location", "Sidebar") == "Sidebar" else 0,
            key="editor_loc_radio_sidebar" )
        st.session_state.editor_location = location

        with st.expander("🛠️ 調試與會話資訊 (Main Chat)", expanded=False): # ...
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
                    messages_json_main = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2, default=str) # Added default=str
                    st.text_area("messages.json (Main Chat)", value=messages_json_main, height=200, key="main_msg_json_area_sidebar")
                    st.download_button(label="📥 下載 Main messages.json", data=messages_json_main, file_name="main_messages.json", mime="application/json", key="dl_main_json_sidebar")
                except TypeError as te:
                    st.error(f"無法序列化主聊天消息: {te}. 可能包含無法轉換為JSON的對象 (例如 PIL Image)。")


    # --- Main Area with Tabs ---
    # Add Executive Workflow to tab names and keys
    tab_ui_names = ["💬 Main Chat & Analysis", "💼 Executive Workflow"] + [ROLE_DEFINITIONS[rid]["name"] for rid in ROLE_DEFINITIONS.keys()]
    
    tabs = st.tabs(tab_ui_names)

    # Tab 0: Main Chat
    with tabs[0]:
        # ... (Existing Main Chat UI and logic from previous response) ...
        st.header("💬 Main Chat & Data Analysis Engine")
        for idx, message in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(message["role"]):
                # ... (message display logic from previous, ensure PIL image display if stored)
                if isinstance(message["content"], list): 
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url": st.image(item["image_url"]["url"], caption="📷", use_container_width=True)
                        elif isinstance(item, dict) and item.get("type") == "text": st.write(item["text"])
                        else: st.write(item) 
                else: st.write(message["content"])

        user_input_main = st.chat_input("Ask the main assistant (supports CSV/Image)...", key="main_chat_input_box_main")
        if user_input_main:
            append_message_to_stream("messages", "user", user_input_main)
            # If image was pending, it's now associated with this text.
            # The get_gemini_response_main_chat will handle sending image + text.
            st.rerun() 
        
        # LLM Call for Main Chat (if last message is user)
        if st.session_state.get("messages") and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Main Assistant thinking..."):
                client_openai_main = None
                if "gpt" in selected_model_main.lower():
                    openai_api_key_val = st.session_state.get("openai_api_key_input")
                    if not openai_api_key_val: st.error("OpenAI key needed for GPT."); st.stop()
                    from openai import OpenAI
                    client_openai_main = OpenAI(api_key=openai_api_key_val)

                model_params_main = {"model": selected_model_main, "temperature": 0.5, "max_tokens": 4096}
                response_content_main = get_llm_response(client_openai_main, model_params_main, message_stream_key="messages")

                if response_content_main and not response_content_main.startswith("Error:"):
                    append_message_to_stream("messages", "assistant", response_content_main)
                    if "pending_image_for_main_gemini" in st.session_state: # Clear after successful send
                        del st.session_state.pending_image_for_main_gemini
                elif response_content_main.startswith("Error:"):
                    append_message_to_stream("messages", "assistant", f"Main LLM Error: {response_content_main}")
                else:
                    append_message_to_stream("messages", "assistant", "Main assistant failed to get a response.")
                st.rerun()
        
        # ... (Main Chat Code Editor & Cross-Validation UI from previous response) ...
        if st.session_state.get("editor_location") == "Main":
            with st.expander("🖋️ Persistent Code Editor (Main Chat)", expanded=True): # Keep expanded
                # ... ACE editor and execute button for main ...
                edited_code_main = st_ace( value=st.session_state.get("ace_code", "# Python code for main analysis"), language="python", theme="monokai", height=300, key="ace_editor_main_chat_main_tab" )
                if edited_code_main != st.session_state.get("ace_code"): st.session_state.ace_code = edited_code_main
                if st.button("▶️ Execute Code (Main Chat)", key="exec_code_main_btn_main_tab"):
                    global_vars_main = { "st_session_state": st.session_state, "pd": pd, "plt": plt, "st": st, "uploaded_file_path": st.session_state.get("uploaded_file_path")}
                    exec_result_main = execute_code(st.session_state.ace_code, global_vars=global_vars_main)
                    st.text_area("Execution Result:", value=str(exec_result_main), height=150, key="exec_result_main_area_main_tab")
        st.markdown("---")
        st.subheader("🔬 Multi-Model Cross-Validation (Main Chat)") # ...
        # ... (Cross-validation UI and logic for main chat from previous response) ...
        default_validator_idx = LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0
        validator_model_name = st.selectbox("選擇交叉驗證模型 (Main Chat):", LLM_MODELS, index=default_validator_idx, key="validator_model_main_select_main_tab")
        if st.button("🚀 執行交叉驗證 (Main Chat)", key="cross_validate_main_btn_main_tab"):
            client_for_validator = None # ... (OpenAI client init if GPT validator) ...
            if "gpt" in validator_model_name.lower(): # ...
                openai_api_key_val = st.session_state.get("openai_api_key_input")
                if not openai_api_key_val: st.error("OpenAI key for GPT validator missing."); st.stop()
                from openai import OpenAI; client_for_validator = OpenAI(api_key=openai_api_key_val)

            if not st.session_state.get("messages") or len(st.session_state.messages) < 2: st.warning("Main chat內容過少。"); st.stop()
            model_params_validator = {"model": validator_model_name, "temperature": 0.2, "max_tokens": 4096}
            with st.spinner(f"使用 {validator_model_name} 進行交叉驗證中..."):
                validated_data = get_cross_validated_response(client_for_validator, model_params_validator)
                st.markdown(f"#### ✅ {validator_model_name} 交叉驗證結果："); st.markdown(validated_data.get("validator_response", "未能獲取驗證回覆。"))


    # Tab 1: Executive Workflow
    with tabs[1]:
        st.header("💼 Executive Decision Workflow")
        st.write("This workflow uses the CSV uploaded in the Main Chat tab.")

        st.session_state.executive_user_query = st.text_area(
            "Enter the Business Problem or Question for Executive Analysis:",
            value=st.session_state.get("executive_user_query", ""),
            key="exec_problem_input",
            height=100
        )

        can_start_exec_workflow = bool(st.session_state.get("uploaded_file_path") and st.session_state.get("executive_user_query"))
        
        if st.button("🚀 Start/Restart Executive Analysis", key="start_exec_workflow_btn", disabled=not can_start_exec_workflow):
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
                debug_log("Executive Workflow Initiated.")
                st.rerun()
        
        if not can_start_exec_workflow and st.session_state.executive_workflow_stage == "idle":
             st.info("Please upload a CSV in the 'Main Chat' tab and enter a business problem above to start the Executive Analysis.")


        # --- Data Profiling Stage ---
        if st.session_state.executive_workflow_stage == "data_profiling_pending":
            with st.spinner("Generating data profile for executives..."):
                try:
                    df_exec = pd.read_csv(st.session_state.uploaded_file_path)
                    st.session_state.executive_data_profile_str = generate_data_profile(df_exec)
                    append_message_to_stream("cfo_exec_messages", "system", f"Data Profile Provided:\n{st.session_state.executive_data_profile_str[:500]}...") # Log snippet
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

        # --- CFO Analysis Stage ---
        if st.session_state.executive_workflow_stage == "cfo_analysis_pending":
            with st.spinner("CFO is analyzing... (using Gemini Pro by default)"):
                cfo_prompt = f"""You are the Chief Financial Officer (CFO).
                Business Problem/Question: {st.session_state.executive_user_query}

                Provided Data Profile:
                ---
                {st.session_state.executive_data_profile_str}
                ---
                Please provide a concise financial analysis based on this information. Focus on:
                1. Key financial insights or questions that arise from the data profile.
                2. Potential financial risks or opportunities highlighted.
                3. What key financial metrics should be investigated further if the full dataset were available?
                Respond in Traditional Chinese. Start your response with "CFO Financial Analysis:".
                """
                append_message_to_stream("cfo_exec_messages", "user", cfo_prompt) # Log the constructed prompt
                cfo_model_params = {"model": "gemini-1.5-pro", "temperature": 0.3} # CFO more analytical
                
                cfo_response = get_gemini_executive_analysis("CFO", cfo_prompt, cfo_model_params)
                
                if cfo_response and not cfo_response.startswith("Error:"):
                    st.session_state.cfo_analysis_text = cfo_response
                    append_message_to_stream("cfo_exec_messages", "assistant", cfo_response)
                    st.session_state.executive_workflow_stage = "coo_analysis_pending" # Move to next stage
                    debug_log("CFO analysis completed.")
                    st.rerun()
                else:
                    st.error(f"CFO analysis failed: {cfo_response}")
                    st.session_state.executive_workflow_stage = "error" # Or back to idle

        if st.session_state.cfo_analysis_text:
            st.subheader("📊 CFO Analysis")
            st.markdown(st.session_state.cfo_analysis_text)
            st.markdown("---")

        # --- COO Analysis Stage ---
        if st.session_state.executive_workflow_stage == "coo_analysis_pending":
            with st.spinner("COO is analyzing... (using Gemini Pro by default)"):
                coo_prompt = f"""You are the Chief Operating Officer (COO).
                Business Problem/Question: {st.session_state.executive_user_query}

                Provided Data Profile:
                ---
                {st.session_state.executive_data_profile_str}
                ---
                CFO's Financial Analysis (for context):
                ---
                {st.session_state.cfo_analysis_text}
                ---
                Please provide a concise operational analysis. Focus on:
                1. Key operational insights or questions based on the data profile and CFO's input.
                2. Potential operational efficiencies, bottlenecks, or resource allocation issues.
                3. What key operational metrics should be investigated further?
                Respond in Traditional Chinese. Start your response with "COO Operational Analysis:".
                """
                append_message_to_stream("coo_exec_messages", "user", coo_prompt)
                coo_model_params = {"model": "gemini-1.5-pro", "temperature": 0.4}
                
                coo_response = get_gemini_executive_analysis("COO", coo_prompt, coo_model_params)

                if coo_response and not coo_response.startswith("Error:"):
                    st.session_state.coo_analysis_text = coo_response
                    append_message_to_stream("coo_exec_messages", "assistant", coo_response)
                    st.session_state.executive_workflow_stage = "ceo_synthesis_pending"
                    debug_log("COO analysis completed.")
                    st.rerun()
                else:
                    st.error(f"COO analysis failed: {coo_response}")
                    st.session_state.executive_workflow_stage = "error"

        if st.session_state.coo_analysis_text:
            st.subheader("⚙️ COO Analysis")
            st.markdown(st.session_state.coo_analysis_text)
            st.markdown("---")

        # --- CEO Synthesis Stage ---
        if st.session_state.executive_workflow_stage == "ceo_synthesis_pending":
            with st.spinner("CEO is synthesizing and making decisions... (using Gemini Pro by default)"):
                ceo_prompt = f"""You are the Chief Executive Officer (CEO).
                Business Problem/Question: {st.session_state.executive_user_query}

                Provided Data Profile:
                ---
                {st.session_state.executive_data_profile_str}
                ---
                CFO's Financial Analysis:
                ---
                {st.session_state.cfo_analysis_text}
                ---
                COO's Operational Analysis:
                ---
                {st.session_state.coo_analysis_text}
                ---
                Please synthesize these analyses and provide a strategic summary. Include:
                1. Overall assessment of the situation based on all inputs.
                2. Key strategic priorities or decisions to consider.
                3. Any conflicting points between CFO/COO analysis and how you might reconcile them.
                Respond in Traditional Chinese. Start your response with "CEO Strategic Summary & Decisions:".
                """
                append_message_to_stream("ceo_exec_messages", "user", ceo_prompt)
                ceo_model_params = {"model": "gemini-1.5-pro", "temperature": 0.5}
                
                ceo_response = get_gemini_executive_analysis("CEO", ceo_prompt, ceo_model_params)

                if ceo_response and not ceo_response.startswith("Error:"):
                    st.session_state.ceo_summary_text = ceo_response
                    append_message_to_stream("ceo_exec_messages", "assistant", ceo_response)
                    st.session_state.executive_workflow_stage = "completed"
                    debug_log("CEO synthesis completed.")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"CEO synthesis failed: {ceo_response}")
                    st.session_state.executive_workflow_stage = "error"
        
        if st.session_state.executive_workflow_stage == "completed" and st.session_state.ceo_summary_text:
            st.subheader("👑 CEO Strategic Summary & Decisions")
            st.markdown(st.session_state.ceo_summary_text)
            st.success("Executive Workflow Completed!")
        
        if st.session_state.executive_workflow_stage == "error":
            st.error("An error occurred during the executive workflow. Please check logs if debug mode is on, or try restarting.")

        # Option to view executive "memory streams" (logged prompts and responses)
        if st.session_state.executive_workflow_stage not in ["idle", "data_profiling_pending"]:
            with st.expander("View Executive Communication Logs", expanded=False):
                for exec_role_name, exec_msg_key in [("CFO", "cfo_exec_messages"), ("COO", "coo_exec_messages"), ("CEO", "ceo_exec_messages")]:
                    if st.session_state.get(exec_msg_key):
                        st.markdown(f"**{exec_role_name}'s Log:**")
                        for i, msg in enumerate(st.session_state[exec_msg_key]):
                            st.markdown(f"_{msg['role']} (turn {i//2 + 1})_:\n{msg['content'][:300]}...") # Show snippet
                        st.markdown("---")


    # Tabs for Generic Roles (Summarizer, Code Explainer, Creative Writer)
    for i, role_id_generic in enumerate(ROLE_DEFINITIONS.keys()): # Renamed role_id to role_id_generic
        role_info = ROLE_DEFINITIONS[role_id_generic]
        with tabs[i + 2]: # Starts from index 2 (Main Chat, Exec Workflow, then generic roles)
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"].split('.')[0] + ".")

            message_key_role = role_info["messages_key"]
            if message_key_role not in st.session_state: st.session_state[message_key_role] = []
            
            for msg_role in st.session_state[message_key_role]:
                with st.chat_message(msg_role["role"]):
                    st.write(msg_role["content"])

            user_input_role = st.chat_input(f"Chat with {role_info['name']}...", key=f"input_{role_id_generic}")

            if user_input_role:
                append_message_to_stream(message_key_role, "user", user_input_role)
                # The call to LLM will happen on rerun if last message is user
                st.rerun()
            
            # LLM Call for Generic Role (if last message is user)
            if st.session_state[message_key_role] and st.session_state[message_key_role][-1]["role"] == "user":
                with st.spinner(f"{role_info['name']} is thinking..."):
                    last_user_input_for_role = st.session_state[message_key_role][-1]["content"]
                    role_model_params = { "model": "gemini-1.5-flash", "temperature": 0.7 } # Generic roles can be more creative
                    
                    response_role = get_gemini_response_for_generic_role(role_id_generic, last_user_input_for_role, role_model_params)
                    
                    if response_role and not response_role.startswith("Error:"):
                        append_message_to_stream(message_key_role, "assistant", response_role)
                    else:
                        append_message_to_stream(message_key_role, "assistant", response_role if response_role else f"Sorry, {role_info['name']} couldn't get a response.")
                    st.rerun()


    # Sidebar Code Editor (if selected)
    # ... (Code editor logic from previous response - ensure keys are unique if also in main tab) ...
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
