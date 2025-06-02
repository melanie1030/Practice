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
# from openai import OpenAI # OpenAI parts will be kept but Gemini is the focus for new roles
from PIL import Image
import google.generativeai as genai
from streamlit_ace import st_ace
import time
import matplotlib.font_manager as fm
import matplotlib

# --- Role Definitions ---
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


# 指定字型檔路徑（相對路徑）
font_path = "./fonts/msjh.ttc"

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
LLM_MODELS = [
    "gemini-1.5-flash", # Default for new roles
    "gemini-1.5-pro",
    "gpt-4o",
    "gpt-3.5-turbo-16k",
    # "models/gemini-2.0-flash", # Assuming this is a custom or less common identifier, verify if needed
    "claude-3-7-sonnet-20250219", # Renamed to match a more standard format, assuming it was a typo for Claude 3 Sonnet
    "claude-3-5-haiku-20240729" # Assuming it was a typo for Claude 3.5 Haiku
]

MAX_MESSAGES_PER_STREAM = 10  # Limit message history for each stream (main chat and each role)

# def initialize_client(api_key): # For OpenAI
#     return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_logs" not in st.session_state: # Ensure list exists
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        # st.write(msg) # Avoid writing directly during processing, can clutter UI
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        if "debug_errors" not in st.session_state: # Ensure list exists
            st.session_state.debug_errors = []
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
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

def load_image_base64(image_pil): # Take PIL image as input
    """Convert PIL image to Base64 encoding."""
    try:
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")  # Use PNG for consistency
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return ""

def append_message_to_stream(message_stream_key, role, content, max_messages=MAX_MESSAGES_PER_STREAM):
    """Append a message to a specific message stream and ensure it doesn't exceed max_messages."""
    if message_stream_key not in st.session_state:
        st.session_state[message_stream_key] = []

    # For Gemini, user and model roles are standard for history
    # System prompts are handled differently (usually at the start or via generation_config)
    # We will store system prompts separately and prepend them when making the API call if needed,
    # or rely on setting it once in the chat session if the SDK supports it well.
    # For display purposes, we can keep the system prompt in the message list if it's part of the initial setup.

    st.session_state[message_stream_key].append({"role": role, "content": content})

    # Trim history: keep the first message (often a system prompt) and the last (max_messages - 1) messages.
    # This logic might need adjustment based on how system prompts are handled for roles.
    # For now, if the first message is a system prompt, we try to preserve it.
    current_messages = st.session_state[message_stream_key]
    if len(current_messages) > max_messages:
        num_to_remove = len(current_messages) - max_messages
        # A simple strategy: remove the oldest non-system messages after the very first message (if system)
        # If first is system, remove from index 1. Otherwise, remove from index 0.
        start_index_for_removal = 0
        if current_messages[0]["role"] == "system":
            start_index_for_removal = 1
        
        del current_messages[start_index_for_removal : start_index_for_removal + num_to_remove]
        st.session_state[message_stream_key] = current_messages
        debug_log(f"Message history for '{message_stream_key}' trimmed to maintain token limits. Count: {len(st.session_state[message_stream_key])}")


def add_user_image_to_main_chat(uploaded_file): # Specific to main chat's handling
    """添加用戶圖片消息到session state (for main chat)"""
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        # Base64 for GPT, file path/direct PIL for Gemini (Gemini SDK handles file paths or PIL.Image)
        use_base64_for_gpt = "gpt" in current_model

        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path # Store path for potential direct use by Gemini
        image_pil = Image.open(file_path)

        if use_base64_for_gpt:
            image_base64 = load_image_base64(image_pil)
            if not image_base64:
                st.error("圖片轉為Base64失敗。")
                return
            image_url = f"data:image/{file_path.split('.')[-1]};base64,{image_base64}"
            image_msg_content = [{"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}]
            append_message_to_stream("messages", "user", image_msg_content) # Main chat uses "messages"
            st.session_state.image_base64 = image_base64 # For OpenAI
        else: # For Gemini or other models that might take paths or PIL objects
            # Gemini SDK's send_message can take [string, Image]
            # We will store the PIL image and pass it along with text prompt
            st.session_state.pending_image_for_main_gemini = image_pil
            # Display image in chat UI, the actual image data will be sent with the next text prompt
            st.image(image_pil, caption="圖片已上傳，將隨下一條文字訊息發送。", use_container_width=True)
            # We don't add to st.session_state.messages yet for Gemini, it's combined with text.

        debug_log(f"圖片已處理: {file_path}. Base64 for GPT: {use_base64_for_gpt}")
        # For GPT, rerun to display. For Gemini, wait for text.
        if use_base64_for_gpt:
            st.rerun()

    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        st.error("圖片處理異常，請檢查日誌")
        debug_error(f"Error in add_user_image_to_main_chat: {e}, Traceback: {traceback.format_exc()}")


def reset_session_messages(message_stream_key="messages"):
    """Clear conversation history from a specific session stream."""
    if message_stream_key in st.session_state:
        st.session_state.pop(message_stream_key)
        st.success(f"Memory for '{message_stream_key}' cleared!")
        debug_log(f"Conversation history for '{message_stream_key}' cleared.")
    # Also clear related chat session for roles
    for role_id, P_info in ROLE_DEFINITIONS.items():
        if P_info["messages_key"] == message_stream_key and P_info["chat_session_key"] in st.session_state:
            del st.session_state[P_info["chat_session_key"]]
            debug_log(f"Gemini chat session for role '{role_id}' cleared.")


def execute_code(code, global_vars=None):
    # ... (keep existing execute_code function)
    try:
        exec_globals = global_vars if global_vars else {}
        # Ensure pandas, streamlit, and matplotlib are available if code uses them
        exec_globals.update({'pd': pd, 'st': st, 'plt': plt, 'np': __import__('numpy')})
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.session_state.debug_logs.append(f"```python\n{code}\n```")
        debug_log(f"Executing code with global_vars: {list(exec_globals.keys())}")
        
        # Capture stdout
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(code, exec_globals)
        
        sys.stdout = old_stdout # Restore stdout
        
        # Get output from various possible sources
        std_output_val = redirected_output.getvalue()
        script_output_val = exec_globals.get("output", "") # if script explicitly sets 'output'

        final_output = ""
        if std_output_val:
            final_output += f"Standard Output:\n{std_output_val}\n"
        if script_output_val:
            final_output += f"Script 'output' variable:\n{script_output_val}\n"
        
        if not final_output:
            final_output = "(No explicit output captured from print or 'output' variable)"
            
        debug_log(f"Execution output: {final_output}")
        return f"Code executed. Output:\n{final_output}"
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_log(f"Execution error: {error_msg}")
        # if st.session_state.get("debug_mode", False): # Always show full error for code execution
        return error_msg
        # else:
        #     return "Error executing code (full details hidden in non-debug mode)."


def extract_json_block(response: str) -> str:
    # ... (keep existing extract_json_block function)
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip() # Return original if no JSON block

# --- Gemini Interaction for ROLES (New Function) ---
def get_gemini_response_for_role(role_id, user_input_text, model_params, max_retries=3):
    """
    Handles Gemini API requests for a specific role, maintaining its unique chat history.
    """
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key:
        st.error("未設定 Gemini API 金鑰，請於側邊欄設定。")
        return "Error: Missing Gemini API Key."
    genai.configure(api_key=api_key)

    role_info = ROLE_DEFINITIONS[role_id]
    model_name = model_params.get("model", "gemini-1.5-flash") # Default to flash for roles
    chat_session_key = role_info["chat_session_key"]
    messages_key = role_info["messages_key"]

    # Initialize chat session if not exists or if model changed for the role
    # For simplicity, we re-initialize if the model in model_params is different from what might have been used last time
    # A more robust solution would store the model used with the session, but this is often fine.
    gemini_model_instance = genai.GenerativeModel(
        model_name,
        system_instruction=role_info["system_prompt"] # Set system prompt here
    )

    # Load history for the role from st.session_state[messages_key]
    # Convert to Gemini's expected format: list of {'role': 'user'/'model', 'parts': [text]}
    # The system prompt is handled by system_instruction in GenerativeModel
    gemini_history = []
    if messages_key in st.session_state:
        for msg in st.session_state[messages_key]:
            if msg["role"] == "user":
                gemini_history.append({'role': 'user', 'parts': [msg["content"]]})
            elif msg["role"] == "assistant": # Gemini uses 'model' for assistant
                gemini_history.append({'role': 'model', 'parts': [msg["content"]]})
            # System prompts are now part of the model's configuration

    if chat_session_key not in st.session_state or st.session_state[chat_session_key].model_name != f"models/{model_name}":
        debug_log(f"Initializing new Gemini chat session for role '{role_id}' with model '{model_name}' and history.")
        st.session_state[chat_session_key] = gemini_model_instance.start_chat(history=gemini_history)
    else:
        # If session exists, ensure its history is up-to-date (if we manually modify gemini_history)
        # However, if we only append to st.session_state[messages_key] and then reconstruct gemini_history,
        # starting a new chat with the updated history is safer.
        # For now, let's re-start_chat to pass the most current history.
        # This is less efficient than truly continuing a session if the SDK manages history perfectly with send_message alone.
        debug_log(f"Re-starting Gemini chat session for role '{role_id}' with updated history.")
        st.session_state[chat_session_key] = gemini_model_instance.start_chat(history=gemini_history)


    chat = st.session_state[chat_session_key]
    
    # Append current user message to the display history (will be sent to Gemini next)
    # append_message_to_stream(messages_key, "user", user_input_text) # Already done by the caller

    debug_log(f"Sending message to Gemini for role '{role_id}': {user_input_text}")
    debug_log(f"History for role '{role_id}' being sent (implicitly by chat object or explicitly if generate_content): {chat.history}")


    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            response = chat.send_message(user_input_text) # Send only the new message
            final_reply = response.text.strip()
            debug_log(f"Gemini response for role '{role_id}': {final_reply}")
            
            # After successful response, update the main message list for this role
            # The user message was already added by the calling UI function. Now add assistant's.
            # append_message_to_stream(messages_key, "assistant", final_reply) # Done by caller

            return final_reply
        except Exception as e: # Catch broader exceptions, including API errors, rate limits etc.
            debug_error(f"Gemini API error for role '{role_id}' (attempt {retries + 1}): {e}")
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
                 st.error(f"Gemini API 金鑰無效或權限不足: {e}")
                 return "Error: Gemini API Key Invalid or Permission Denied."
            if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"Gemini 請求頻繁或資源耗盡，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            else:
                st.error(f"Gemini API 請求發生未預期錯誤: {e}")
                return f"Error: An unexpected error occurred with Gemini API: {e}"
    
    st.error(f"Gemini for role '{role_id}': 請求失敗次數過多，請稍後重試。")
    return "Error: Max retries exceeded for Gemini."


# --- Main Chat LLM Response Functions (Existing, slightly adapted for clarity) ---
def get_gemini_response_main_chat(model_params, max_retries=3):
    api_key = st.session_state.get("gemini_api_key_input", "")
    if not api_key:
        st.error("未設定 Gemini API 金鑰")
        return ""
    genai.configure(api_key=api_key)
    model_name = model_params.get("model", "gemini-1.5-flash")
    debug_log(f"Gemini (Main Chat) model: {model_name}")
    
    # For main chat, we reconstruct messages each time for now, including potential image.
    # The system prompt is added if not present.
    current_main_messages = st.session_state.get("messages", [])
    
    # Prepare content for Gemini API
    # Gemini expects a list where each item is a dict like:
    # {'role': 'user'|'model', 'parts': [text_part, image_part_if_any]}
    gemini_content_history = []
    system_instruction_main = "請以繁體中文回答，並且所有回覆必須以 #zh-tw 回覆還有回覆時不用在開頭加上#zh-tw。" # Default for main

    # Check for a specific system prompt in the main messages
    # The very first message if it's from 'system' role could be used as system_instruction.
    # However, the current code appends a generic "You are an assistant..." if no system prompt.
    # For this refactor, let's make the default explicit.
    # If users upload a thinking_protocol.md, it gets added as a "user" message, which is fine.

    for msg in current_main_messages:
        role = "model" if msg["role"] == "assistant" else msg["role"] # Gemini uses 'model'
        if role not in ["user", "model"]: continue # Skip system prompts from history if any for now

        if isinstance(msg["content"], list): # Multimodal content (likely text + image for GPT format)
            parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    # This was for GPT. For Gemini, image handling is different if sent via SDK path.
                    # If st.session_state.pending_image_for_main_gemini exists, it should be used.
                    # This part of history reconstruction might be complex if mixing GPT/Gemini formats.
                    # For now, assume text parts are primary for history.
                    # The `pending_image_for_main_gemini` is for the *current* turn.
                    debug_log("Image in history for main chat - skipping for Gemini history reconstruction, will be sent if current turn.")
                else: # Simple text
                    parts.append(str(item))
            if parts:
              gemini_content_history.append({'role': role, 'parts': parts})
        else: # Simple text message
            gemini_content_history.append({'role': role, 'parts': [str(msg["content"])]})

    # Add pending image for the current turn if it exists
    current_turn_parts = []
    last_user_message_text = ""
    if gemini_content_history and gemini_content_history[-1]['role'] == 'user':
        last_user_message_text_parts = gemini_content_history[-1]['parts']
        # We need to pop it from history if we are sending it as the *current* message
        # and it has an image.
        # The last message in gemini_content_history is the one we're about to send.

    # The last message in `current_main_messages` is the current user input.
    # Let's assume the image is associated with the *very last* user message.
    if current_main_messages and current_main_messages[-1]["role"] == "user":
        current_user_text_input = str(current_main_messages[-1]["content"]) # This might be list for GPT images
        if isinstance(current_main_messages[-1]["content"], list): # if it was GPT image format
            for item_u in current_main_messages[-1]["content"]:
                if isinstance(item_u, dict) and item_u.get("type") == "text":
                    current_user_text_input = item_u["text"]
                    break
        current_turn_parts.append(current_user_text_input)

    if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
        current_turn_parts.append(st.session_state.pending_image_for_main_gemini)
        debug_log("Attaching pending_image_for_main_gemini to current Gemini call.")
        del st.session_state.pending_image_for_main_gemini # Consume it

    # The history for start_chat should not include the current message to be sent.
    history_for_session = gemini_content_history[:-1] if gemini_content_history and gemini_content_history[-1]['role'] == 'user' and current_turn_parts else gemini_content_history


    model_instance_main = genai.GenerativeModel(model_name, system_instruction=system_instruction_main)
    chat_session_main = model_instance_main.start_chat(history=history_for_session)
    debug_log(f"Gemini (Main Chat) session created/restarted. History length: {len(history_for_session)}")
    debug_log(f"Gemini (Main Chat) sending parts for current turn: {current_turn_parts}")


    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            if not current_turn_parts: # Should not happen if user sent text
                 debug_error("No content to send for current turn in main Gemini chat.")
                 return "Error: No content to send."

            response = chat_session_main.send_message(current_turn_parts)
            final_reply = response.text.strip()
            debug_log(f"Gemini (Main Chat) send_message final reply => {final_reply}")
            return final_reply
        except Exception as e:
            debug_error(f"Gemini (Main Chat) API error (attempt {retries + 1}): {e}")
            if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) :
                 st.error(f"Gemini API 金鑰無效或權限不足: {e}")
                 return "Error: Gemini API Key Invalid or Permission Denied."
            if "rate limit" in str(e).lower() or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"Gemini (Main Chat) 請求頻繁或資源耗盡，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            else:
                st.error(f"Gemini (Main Chat) API 請求發生未預期錯誤: {e}")
                return f"Error: An unexpected error occurred with Gemini API for main chat: {e}"
    st.error("Gemini (Main Chat): 請求失敗次數過多，請稍後重試")
    return ""


def get_openai_response(client, model_params, max_retries=3):
    # ... (keep existing get_openai_response, ensure client is passed correctly)
    # This function uses st.session_state.messages directly.
    # Minor: Ensure OpenAI client is initialized if this function is called.
    if not client:
        openai_api_key_current = st.session_state.get("openai_api_key_input") or os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key_current:
            st.error("OpenAI API Key not available for OpenAI model.")
            return ""
        from openai import OpenAI # Ensure OpenAI class is imported
        client = OpenAI(api_key=openai_api_key_current)

    retries = 0
    wait_time = 5
    model_name = model_params.get("model", "gpt-4o") # gpt-4-turbo is old name
    
    # Filter out any non-string content items if a list is passed, specific for OpenAI
    processed_messages_for_openai = []
    for msg in st.session_state.get("messages",[]): # Assuming this is for main chat
        if isinstance(msg["content"], list):
            new_content_list = []
            has_text = False
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    new_content_list.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    new_content_list.append(item)
                    has_text = True
                elif isinstance(item, str): # if a simple string was mixed in
                    new_content_list.append({"type": "text", "text": item})
                    has_text = True
            # OpenAI requires at least one text part in a multimodal message list
            if not has_text and new_content_list: 
                 # Add an empty text part if only image was there and it's expected to have text.
                 # Or, better, ensure prompts always include text. For now, let's assume text exists.
                 pass
            processed_messages_for_openai.append({"role": msg["role"], "content": new_content_list})
        else:
            processed_messages_for_openai.append(msg)


    while retries < max_retries:
        try:
            request_params = {
                "model": model_name,
                "messages": processed_messages_for_openai, # Use main chat messages
                "temperature": model_params.get("temperature", 0.3),
                "max_tokens": model_params.get("max_tokens", 4096), # GPT-4o typical context, but output is less
                "stream": False
            }
            # if any(isinstance(msg.get("content"), list) for msg in processed_messages_for_openai):
            #     # Vision models might have different token needs for the image itself vs output
            #     # request_params["max_tokens"] = 4096 # Max output tokens for vision
            #     debug_log("OpenAI: Detected multimodal input.")

            response = client.chat.completions.create(**request_params)
            response_content = response.choices[0].message.content.strip()
            debug_log(f"OpenAI原始回應：\n{response_content}")
            return response_content
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                debug_error(f"速率限制錯誤（嘗試 {retries+1}/{max_retries}）：{e}")
                st.warning(f"請求過於頻繁，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower():
                debug_error(f"API金鑰無效：{e}")
                st.error("OpenAI API金鑰無效，請檢查後重試")
                return ""
            else:
                debug_error(f"OpenAI請求異常：{str(e)}")
                st.error(f"請求發生錯誤：{str(e)}")
                return ""
    debug_error(f"超過最大重試次數（{max_retries}次）")
    st.error("請求失敗次數過多，請稍後再試")
    return ""

def get_claude_response(model_params, max_retries=3):
    # ... (keep existing get_claude_response, ensure it uses st.session_state.messages for main chat)
    debug_log("Claude loading for main chat")
    import anthropic # Keep import local if not used elsewhere often

    api_key = st.session_state.get("claude_api_key_input") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("未設定 Claude API 金鑰，請在側邊欄輸入")
        return ""
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Claude Client 初始化失敗: {e}")
        return ""
        
    model_name = model_params.get("model", "claude-3-opus-20240229") # Default to Opus
    max_tokens_to_sample = model_params.get("max_tokens", 8192) # Claude uses max_tokens_to_sample
    temperature = model_params.get("temperature", 0.3) # Claude uses 0.0-1.0

    # Claude messages: handle system prompt and multimodal content correctly
    claude_messages = []
    system_prompt_claude = None
    
    # Use main messages for Claude in this generic function
    # Role-specific calls would need to pass their own message list
    source_messages = st.session_state.get("messages", [])

    for i, msg in enumerate(source_messages):
        if msg["role"] == "system" and i == 0: # Anthropic suggests system prompt as a top-level param
            system_prompt_claude = msg["content"]
            continue

        if isinstance(msg["content"], list): # Multimodal for Claude
            content_parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    # Claude needs base64 data directly for images
                    img_url = item["image_url"]["url"]
                    if img_url.startswith("data:image"):
                        base64_data = img_url.split(",")[1]
                        media_type = img_url.split(";")[0].split(":")[1]
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        })
                elif isinstance(item, dict) and item.get("type") == "text":
                     content_parts.append({"type": "text", "text": item["text"]})
                elif isinstance(item, str): # simple text part in list
                     content_parts.append({"type": "text", "text": item})
            if content_parts:
                 claude_messages.append({"role": msg["role"], "content": content_parts})
        else: # Simple text message
            claude_messages.append({"role": msg["role"], "content": str(msg["content"])})
    
    # Ensure user role is last if messages exist
    if claude_messages and claude_messages[-1]["role"] != "user":
        # This might happen if last message was an assistant response and we are re-generating
        # Or if the logic for appending messages needs adjustment.
        # For now, proceed, but this could be an issue for Claude if it expects user turn.
        debug_log("Claude: Last message in history is not 'user'. This might be unexpected.")


    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            api_params = {
                "model": model_name,
                "max_tokens": max_tokens_to_sample,
                "messages": claude_messages,
                "temperature": temperature,
            }
            if system_prompt_claude:
                api_params["system"] = system_prompt_claude
            
            debug_log(f"Calling Claude with params: model={model_name}, max_tokens={max_tokens_to_sample}, temp={temperature}, system_prompt_present={bool(system_prompt_claude)}")
            # debug_log(f"Claude messages being sent: {json.dumps(claude_messages, indent=2)}")


            response = client.messages.create(**api_params)
            
            completion = ""
            if response.content and isinstance(response.content, list):
                for block in response.content:
                    if block.type == "text":
                        completion += block.text
            
            debug_log(f"Claude 回應：{completion}")
            return completion.strip()

        except Exception as e:
            debug_error(f"Claude API 請求異常（嘗試 {retries+1}/{max_retries}）：{e}")
            if "rate limit" in str(e).lower() or "429" in str(e):
                st.warning(f"Claude 請求過於頻繁，{wait_time}秒後重試...")
            elif "authentication_error" in str(e).lower() or "invalid_api_key" in str(e).lower():
                 st.error(f"Claude API金鑰無效或驗證失敗: {e}")
                 return "" # Stop retrying for auth errors
            else:
                 st.warning(f"Claude 生成錯誤，{wait_time}秒後重試...")
            
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2
            if retries >= max_retries:
                 st.error(f"Claude 請求失敗次數過多: {e}")
                 return ""
    return ""


def get_llm_response(client, model_params, message_stream_key="messages", current_user_input_for_gemini=None, max_retries=3):
    """獲取LLM模型回覆（支持OpenAI, Gemini for main chat, and Claude for main chat）"""
    model_name = model_params.get("model", "gpt-4o") # Default if not specified elsewhere
    debug_log(f"Starting get_llm_response for model: {model_name}, stream: {message_stream_key}")

    if "gpt" in model_name.lower():
        debug_log("Routing to OpenAI for main chat")
        return get_openai_response(client, model_params, max_retries) # Assumes client is OpenAI client
    elif "gemini" in model_name.lower():
        # This is for the main chat's Gemini logic
        debug_log(f"Routing to Gemini for main chat. User input: {current_user_input_for_gemini}")
        return get_gemini_response_main_chat(model_params=model_params, max_retries=max_retries)
    elif "claude" in model_name.lower():
        debug_log("Routing to Claude for main chat")
        return get_claude_response(model_params, max_retries)
    else:
        st.error(f"不支援的模型類型: {model_name}")
        return ""

# --- Cross Validation (operates on main chat messages) ---
def get_cross_validated_response(client, model_params_validator, max_retries=3):
    # ... (keep existing, but ensure it uses the main 'messages' stream)
    # And ensure it can call the correct LLM provider for the validator model
    cross_validation_prompt_content = (
        "請仔細閱讀以下全部對話記憶，對先前模型的回答進行交叉驗證。"
        "你的任務是檢查回答的正確性，指出其中可能存在的錯誤或不足，"
        "並提供具體的數據、理由或例子來支持你的分析。"
        "請務必使用繁體中文回答。"
        "在回答時請回答的詳細，內容需要你盡可能的多。"
        "並且越漂亮越好。"
    )
    
    # Temporarily add system prompt for validation to the main message stream
    original_main_messages = list(st.session_state.get("messages", [])) # Make a copy
    st.session_state.messages.insert(0, {"role": "system", "content": cross_validation_prompt_content})
    debug_log(f"Cross-validation prompt added to main messages. Validating with model: {model_params_validator.get('model')}")

    # The get_llm_response needs to correctly dispatch based on validator_model_name
    # The 'client' parameter here is potentially the OpenAI client, which is only relevant if validator is GPT
    validator_model_name = model_params_validator.get('model')
    validator_response_text = ""

    if "gpt" in validator_model_name.lower():
        # Ensure OpenAI client is available
        openai_api_key_val = st.session_state.get("openai_api_key_input") or os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key_val:
            st.error("OpenAI API Key required for GPT validator.")
            st.session_state.messages = original_main_messages # Restore original messages
            return {"validator_response": "Error: OpenAI API Key missing."}
        from openai import OpenAI # Local import to avoid issues if not used
        validator_client_openai = OpenAI(api_key=openai_api_key_val)
        validator_response_text = get_openai_response(validator_client_openai, model_params_validator, max_retries)
    elif "gemini" in validator_model_name.lower():
         # For Gemini, get_gemini_response_main_chat uses st.session_state.messages
         # The system prompt we added will be picked up by it.
        validator_response_text = get_gemini_response_main_chat(model_params_validator, max_retries)
    elif "claude" in validator_model_name.lower():
        # For Claude, get_claude_response uses st.session_state.messages
        validator_response_text = get_claude_response(model_params_validator, max_retries)
    else:
        st.error(f"Unsupported validator model: {validator_model_name}")
        validator_response_text = "Error: Unsupported validator model."

    # Restore original main messages
    st.session_state.messages = original_main_messages
    debug_log("Cross-validation prompt removed from main messages.")

    final_response = {
        "validator_response": validator_response_text
    }
    return final_response


# ------------------------------
# 主應用入口
# ------------------------------
def main():
    st.set_page_config(page_title="Multi-Role Chatbot + Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Multi-Role Chatbot + 📊 Data Analysis")

    # --- Initialize Session States ---
    if "messages" not in st.session_state: st.session_state.messages = [] # Main chat
    for role_id, role_info in ROLE_DEFINITIONS.items():
        if role_info["messages_key"] not in st.session_state:
            st.session_state[role_info["messages_key"]] = []
            # Optionally add initial system message for display, but it's better handled by Gemini's system_instruction
            # st.session_state[role_info["messages_key"]].append({"role": "system", "content": role_info["system_prompt"]})
        if role_info["chat_session_key"] not in st.session_state:
             st.session_state[role_info["chat_session_key"]] = None


    if "ace_code" not in st.session_state: st.session_state.ace_code = ""
    if "editor_location" not in st.session_state: st.session_state.editor_location = "Sidebar"
    if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
    if "uploaded_image_path" not in st.session_state: st.session_state.uploaded_image_path = None # For main chat
    if "image_base64" not in st.session_state: st.session_state.image_base64 = None # For main chat OpenAI
    if "pending_image_for_main_gemini" not in st.session_state: st.session_state.pending_image_for_main_gemini = None


    if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
    if "deep_analysis_mode" not in st.session_state: st.session_state.deep_analysis_mode = True # Default from original
    # ... other initializations from your original main ...
    if "second_response" not in st.session_state: st.session_state.second_response = ""
    if "third_response" not in st.session_state: st.session_state.third_response = ""
    if "deep_analysis_image" not in st.session_state: st.session_state.deep_analysis_image = None
    if "debug_logs" not in st.session_state: st.session_state.debug_logs = []
    if "debug_errors" not in st.session_state: st.session_state.debug_errors = []
    if "thinking_protocol" not in st.session_state: st.session_state.thinking_protocol = None
    # Remove gemini_ai_chat and gemini_ai_history if using role-based sessions
    # if "gemini_ai_chat" in st.session_state: del st.session_state["gemini_ai_chat"]
    # if "gemini_ai_history" in st.session_state: del st.session_state["gemini_ai_history"]

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        # OpenAI API Key
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key_input = st.text_input("OpenAI API Key", value=default_openai_key, type="password", key="openai_api_key_input_widget")
        if openai_api_key_input:
            st.session_state.openai_api_key_input = openai_api_key_input # Store in session state

        # Gemini API Key
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key_input = st.text_input("Gemini API Key", value=default_gemini_key, type="password", key="gemini_api_key_input_widget")
        if gemini_api_key_input:
            st.session_state.gemini_api_key_input = gemini_api_key_input

        # Claude API Key
        default_claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        claude_api_key_input = st.text_input("Claude API Key", value=default_claude_key, type="password", key="claude_api_key_input_widget")
        if claude_api_key_input:
            st.session_state.claude_api_key_input = claude_api_key_input


        st.subheader("⚙️ Main Chat Model")
        selected_model_main = st.selectbox(
            "選擇主要聊天模型:",
            LLM_MODELS,
            index=LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0, # Default to Gemini Flash for main
            key="selected_model_main"
        )
        st.session_state.selected_model = selected_model_main # Keep for compatibility if other parts use it

        st.subheader("🛠️ Tools & Settings")
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.get("debug_mode", False))
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode (Main Chat)", value=st.session_state.get("deep_analysis_mode", True))

        if st.button("🗑️ Clear ALL Memory & Chats"):
            st.session_state.messages = []
            for role_id, role_info in ROLE_DEFINITIONS.items():
                st.session_state[role_info["messages_key"]] = []
                if role_info["chat_session_key"] in st.session_state:
                    del st.session_state[role_info["chat_session_key"]]
            
            # Clear other relevant session states
            keys_to_clear = ["ace_code", "uploaded_file_path", "uploaded_image_path", 
                             "image_base64", "pending_image_for_main_gemini", "second_response", "third_response",
                             "deep_analysis_image", "thinking_protocol", "debug_logs", "debug_errors"]
            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state.pop(key)
            st.success("All memories and chat states cleared!")
            debug_log("All memory cleared.")
            st.rerun()
        
        # ... (rest of your original sidebar: Memory State, Uploads, Editor Location, Debug Info) ...
        # Memory State display should be adapted if you want to show individual role memories here,
        # or keep it for the main chat. For now, let's assume it shows main chat memory.
        st.subheader("🧠 Main Chat Memory State")
        if st.session_state.get("messages"):
            memory_content = "\n".join([f"{('' if msg['role']=='system' else msg['role']+': ')}{msg['content']}" for msg in st.session_state.messages])
            st.text_area("Current Main Chat Memory", value=memory_content, height=150, key="main_chat_memory_display")
        else:
            st.text_area("Current Main Chat Memory", value="No messages yet in main chat.", height=150)

        st.subheader("📂 Upload CSV (Main Chat)")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"], key="main_csv_uploader")
        if uploaded_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### CSV Data Preview")
                st.dataframe(csv_data.head())
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                debug_error(f"Error reading CSV: {e}")

        st.subheader("🖼️ Upload Image (Main Chat)")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="main_image_uploader")
        if uploaded_image:
            if "pending_image_for_main_gemini" in st.session_state and st.session_state.pending_image_for_main_gemini:
                 st.warning("已有圖片待發送。請先發送包含該圖片的文字訊息，或清除記憶。")
            else:
                add_user_image_to_main_chat(uploaded_image) # Handles both GPT and Gemini prep for main chat

        st.subheader("🧠 Upload Thinking Protocol (Main Chat)")
        # ... (keep your thinking protocol upload logic, ensure it appends to "messages") ...
        uploaded_thinking_protocol = st.file_uploader("Choose a thinking_protocol.md file:", type=["md"], key="thinking_protocol_uploader")
        if uploaded_thinking_protocol:
            try:
                thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
                st.session_state.thinking_protocol = thinking_protocol_content
                append_message_to_stream("messages", "user", f"Thinking Protocol Loaded:\n{thinking_protocol_content}") # Add to main chat
                st.success("Thinking Protocol uploaded successfully to main chat!")
                debug_log("Thinking Protocol uploaded and added to main messages.")
            except Exception as e:
                st.error(f"Error reading Thinking Protocol: {e}")
                debug_error(f"Error reading Thinking Protocol: {e}")


        st.subheader("Editor Location")
        location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=1 if st.session_state.get("editor_location", "Sidebar") == "Sidebar" else 0,
            key="editor_loc_radio"
        )
        st.session_state.editor_location = location

        with st.expander("🛠️ 調試與會話資訊 (Main Chat)", expanded=False):
             # ... (Keep your debug info display for main chat - messages.json etc.)
            if st.session_state.get("debug_mode", False):
                st.subheader("調試日誌")
                # Safely join, ensuring all elements are strings
                debug_logs_str = "\n".join(map(str, st.session_state.get("debug_logs", [])))
                st.text_area("Debug Logs", value=debug_logs_str, height=200, key="debug_log_area")
                
                st.subheader("調試錯誤")
                debug_errors_str = "\n".join(map(str, st.session_state.get("debug_errors", [])))
                st.text_area("Debug Errors", value=debug_errors_str, height=200, key="debug_err_area")

            st.subheader("會話資訊 (messages.json - Main Chat)")
            if "messages" in st.session_state and st.session_state.messages:
                try:
                    messages_json_main = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
                    st.text_area("messages.json (Main Chat)", value=messages_json_main, height=200, key="main_msg_json_area")
                    st.download_button(label="📥 下載 Main messages.json", data=messages_json_main, file_name="main_messages.json", mime="application/json")
                except TypeError as te:
                    st.error(f"無法序列化主聊天消息: {te}. 可能包含無法轉換為JSON的對象 (例如 PIL Image)。")
                    debug_error(f"JSON serialization error for main messages: {te}")


    # --- Main Area with Tabs ---
    tab_keys = ["main_chat"] + list(ROLE_DEFINITIONS.keys())
    tab_names = ["💬 Main Chat & Analysis"] + [ROLE_DEFINITIONS[rid]["name"] for rid in ROLE_DEFINITIONS.keys()]
    
    tabs = st.tabs(tab_names)

    # Tab 0: Main Chat
    with tabs[0]:
        st.header("💬 Main Chat & Data Analysis Engine")
        # Display main chat messages
        for idx, message in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], list): # For GPT-style multimodal
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            st.image(item["image_url"]["url"], caption="📷 Uploaded Image", use_container_width=True)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            st.write(item["text"])
                        else:
                            st.write(item) # Fallback
                # elif isinstance(message["content"], Image.Image): # If we store PIL Image directly
                #     st.image(message["content"], caption="📷 Uploaded Image (PIL)", use_container_width=True)
                else: # Simple string
                    st.write(message["content"])

        user_input_main = st.chat_input("Ask the main assistant (supports CSV/Image)...", key="main_chat_input_box")
        if user_input_main:
            # Handle appending user message (text part)
            # If there's a pending image for Gemini, it will be combined by get_gemini_response_main_chat
            append_message_to_stream("messages", "user", user_input_main)
            st.rerun() # Rerun to display user message immediately

            with st.spinner("Main Assistant thinking..."):
                # Determine client (only OpenAI client is explicitly initialized like this for now)
                # Gemini and Claude are handled within their respective functions using API keys from session_state
                client_openai = None
                if "gpt" in selected_model_main.lower():
                    openai_api_key_main = st.session_state.get("openai_api_key_input")
                    if not openai_api_key_main:
                        st.error("OpenAI API Key not provided for the selected GPT model.")
                        st.stop()
                    from openai import OpenAI # Local import
                    client_openai = OpenAI(api_key=openai_api_key_main)

                model_params_main = {
                    "model": selected_model_main, # From sidebar
                    "temperature": 0.5, # Example
                    "max_tokens": 4096 if "gpt" in selected_model_main.lower() else 8192 # Example
                }
                
                # The get_llm_response function will route to the correct provider
                # For Gemini main chat, current_user_input_for_gemini is important if image handling is complex
                # but get_gemini_response_main_chat now derives current turn from st.session_state.messages[-1]
                # and pending_image_for_main_gemini
                response_content_main = get_llm_response(client_openai, model_params_main, message_stream_key="messages")

                if response_content_main:
                    append_message_to_stream("messages", "assistant", response_content_main)
                    # Deep analysis logic from your original code can go here if triggered
                    # ... (Your existing deep analysis, code execution, JSON parsing for main chat)
                    # This part is complex and was tied to the old prompt structure.
                    # For now, just display the response. Refactoring this is a larger step.
                    debug_log(f"Main assistant response: {response_content_main}")
                    st.rerun() # Rerun to display assistant's response
                else:
                    st.error("Main assistant failed to get a response.")
        
        # --- Your existing Code Editor and Execution logic for Main tab ---
        if st.session_state.get("editor_location") == "Main":
            with st.expander("🖋️ Persistent Code Editor (Main Chat)", expanded=True):
                edited_code_main = st_ace(
                    value=st.session_state.get("ace_code", "// Your Python code for main chat analysis here..."),
                    language="python", theme="monokai", height=300, key="ace_editor_main_chat"
                )
                if edited_code_main != st.session_state.get("ace_code"):
                    st.session_state.ace_code = edited_code_main
                
                if st.button("▶️ Execute Code (Main Chat)", key="exec_code_main_btn"):
                    global_vars_main = {
                        "st_session_state": st.session_state, # Pass session state carefully
                        "pd": pd, "plt": plt, "st": st,
                        "uploaded_file_path": st.session_state.get("uploaded_file_path"),
                        # Note: uploaded_image_path might be less useful directly in exec unless code reads it
                    }
                    exec_result_main = execute_code(st.session_state.ace_code, global_vars=global_vars_main)
                    st.text_area("Execution Result:", value=str(exec_result_main), height=150, key="exec_result_main_area")
                    # If exec_result_main contains a plot, it might be displayed via st.pyplot() in the executed code

        # --- Your existing Multi-Model Cross-Validation for Main Tab ---
        st.markdown("---")
        st.subheader("🔬 Multi-Model Cross-Validation (Main Chat)")
        # ... (keep your cross-validation UI and logic here, ensure it uses main chat "messages") ...
        default_validator_idx = LLM_MODELS.index("gemini-1.5-flash") if "gemini-1.5-flash" in LLM_MODELS else 0
        validator_model_name = st.selectbox(
            "選擇交叉驗證模型 (Main Chat):", LLM_MODELS, index=default_validator_idx, key="validator_model_main_select"
        )
        if st.button("🚀 執行交叉驗證 (Main Chat)", key="cross_validate_main_btn"):
            # API key check for validator model
            # ... (similar to your original key check logic for the validator) ...
            # Ensure client for OpenAI is passed if validator is GPT
            client_for_validator = None
            if "gpt" in validator_model_name.lower():
                openai_api_key_val = st.session_state.get("openai_api_key_input")
                if not openai_api_key_val: st.error("OpenAI key needed for GPT validator."); st.stop()
                from openai import OpenAI
                client_for_validator = OpenAI(api_key=openai_api_key_val)

            if not st.session_state.get("messages") or len(st.session_state.messages) < 2:
                st.warning("Main chat內容過少，無法進行交叉驗證。"); st.stop()

            model_params_validator = {"model": validator_model_name, "temperature": 0.2, "max_tokens": 4096}
            with st.spinner(f"使用 {validator_model_name} 進行交叉驗證中..."):
                validated_data = get_cross_validated_response(client_for_validator, model_params_validator)
                st.markdown(f"#### ✅ {validator_model_name} 交叉驗證結果：")
                st.markdown(validated_data.get("validator_response", "未能獲取驗證回覆。"))


    # Tabs for New Roles
    for i, role_id in enumerate(ROLE_DEFINITIONS.keys()):
        role_info = ROLE_DEFINITIONS[role_id]
        with tabs[i + 1]: # i+1 because main chat is tab 0
            st.header(role_info["name"])
            st.caption(role_info["system_prompt"].split('.')[0]) # Show first sentence of system prompt

            # Display messages for this role
            message_key_role = role_info["messages_key"]
            if message_key_role not in st.session_state: st.session_state[message_key_role] = []
            
            for msg_role in st.session_state[message_key_role]:
                with st.chat_message(msg_role["role"]):
                    st.write(msg_role["content"])

            user_input_role = st.chat_input(f"Chat with {role_info['name']}...", key=f"input_{role_id}")

            if user_input_role:
                append_message_to_stream(message_key_role, "user", user_input_role)
                st.rerun() # Display user message

                with st.spinner(f"{role_info['name']} is thinking..."):
                    # For roles, we'll default to Gemini Flash unless a different model is specified per role
                    # For now, let's use a fixed fast model for roles.
                    role_model_params = {
                        "model": "gemini-1.5-flash", # Roles use Gemini Flash by default
                        "temperature": 0.6, # Slightly more creative for roles
                        "max_tokens": 2048
                    }
                    response_role = get_gemini_response_for_role(role_id, user_input_role, role_model_params)
                    
                    if response_role and not response_role.startswith("Error:"):
                        append_message_to_stream(message_key_role, "assistant", response_role)
                    else:
                        # Error message already shown by get_gemini_response_for_role or it returned an error string
                        append_message_to_stream(message_key_role, "assistant", response_role if response_role else "Sorry, I couldn't get a response.")
                    st.rerun() # Display assistant response


    # Sidebar Code Editor (if selected)
    if st.session_state.get("editor_location") == "Sidebar":
        with st.sidebar.expander("🖋️ Persistent Code Editor (Sidebar)", expanded=False):
            edited_code_sidebar = st_ace(
                value=st.session_state.get("ace_code", "// Your Python code here..."),
                language="python", theme="monokai", height=300, key="ace_editor_sidebar"
            )
            if edited_code_sidebar != st.session_state.get("ace_code"):
                st.session_state.ace_code = edited_code_sidebar
            
            if st.button("▶️ Execute Code (Sidebar)", key="exec_code_sidebar_btn"):
                global_vars_sidebar = {
                     "st_session_state": st.session_state,
                     "pd": pd, "plt": plt, "st": st,
                     "uploaded_file_path": st.session_state.get("uploaded_file_path"),
                }
                exec_result_sidebar = execute_code(st.session_state.ace_code, global_vars=global_vars_sidebar)
                # Display result in sidebar or a dedicated main area expander
                st.sidebar.text_area("Execution Result (Sidebar):", value=str(exec_result_sidebar), height=100, key="exec_result_sidebar_area")


if __name__ == "__main__":
    main()
