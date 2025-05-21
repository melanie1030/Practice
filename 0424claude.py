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
import google.generativeai as genai # 新增Gemini依賴
from streamlit_ace import st_ace
import time
import matplotlib.font_manager as fm
import matplotlib

# 指定字型檔路徑（相對路徑）
font_path = "./fonts/msjh.ttc"

# 加入字型並設定為預設字型
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    st.warning(f"字型檔案 {font_path} 未找到，部分圖表中文可能無法正常顯示。")
matplotlib.rcParams['axes.unicode_minus'] = False


# --- 初始化設置 ---
dotenv.load_dotenv()
UPLOAD_DIR = "uploaded_files"
LLM_MODELS = [  # 修改後的模型列表
    "gpt-4o",
    "gpt-3.5-turbo-16k",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-1.5-flash", # Kept original name for compatibility if used elsewhere
    "claude-3-opus-20240229", # Example Claude model, adjust as needed
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

MAX_MESSAGES = 10  # Limit message history

def initialize_client(api_key):
    return OpenAI(api_key=api_key) if api_key else None

def debug_log(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_logs.append(f"**DEBUG LOG:** {msg}")
        # st.write(msg) # Avoid writing to main UI for every debug log
        print(f"DEBUG LOG: {msg}")

def debug_error(msg):
    if st.session_state.get("debug_mode", False):
        st.session_state.debug_errors.append(f"**DEBUG ERROR:** {msg}")
        print(f"DEBUG ERROR: {msg}")
    # st.error(msg) # Display critical errors to user even if not in debug

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    debug_log(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_log(f"Files in {UPLOAD_DIR}: {os.listdir(UPLOAD_DIR)}")
    return file_path

def load_image_base64(image_path_or_obj):
    """Convert image (from path or PIL Image object) to Base64 encoding."""
    try:
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else: # Assumes PIL Image object
            image = image_path_or_obj
        
        buffer = BytesIO()
        # Try to preserve original format if known, else default to PNG
        img_format = image.format if image.format else "PNG"
        if img_format.upper() == "JPEG":
             image.save(buffer, format="JPEG")
             mime_type = "image/jpeg"
        else:
            image.save(buffer, format="PNG") 
            mime_type = "image/png"

        return base64.b64encode(buffer.getvalue()).decode('utf-8'), mime_type
    except Exception as e:
        debug_error(f"Error converting image to base64: {e}")
        return "", "image/png" # Fallback

def append_message(role, content):
    """Append a message and ensure the total number of messages does not exceed MAX_MESSAGES."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Remove the oldest messages except the system prompt (if first message is system)
        if st.session_state.messages[0]["role"] == "system":
            st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_MESSAGES - 1):]
        else:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
        debug_log("Message history trimmed to maintain token limits.")

def add_user_image(uploaded_file):
    """添加用戶圖片消息到session state"""
    try:
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        current_model = st.session_state.get("selected_model", "").lower()
        
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.uploaded_image_path = file_path
        
        image = Image.open(file_path)
        image_base64, mime_type = load_image_base64(image) # Use the modified function

        # For OpenAI models (GPT-4o, etc.), a base64 data URL is typically used.
        # For Gemini with google-generativeai, you often upload the file and use its resource name,
        # or pass Parts. For simplicity in a unified chat, we might standardize on base64 if the SDK call needs it,
        # or handle it differently in get_gemini_response.
        # The current get_gemini_response tries to use a path, so let's see.

        # The OpenAI API expects image content in a specific list format for user messages.
        # Content can be a mix of text and image URLs.
        image_url_for_openai = f"data:{mime_type};base64,{image_base64}"
        
        # For Gemini, the structure in get_gemini_response will handle the image path.
        # We add it to messages in a way that both can potentially parse or ignore.
        # Standardizing on OpenAI's vision format for messages for simplicity here:
        image_msg_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url_for_openai, # This is for OpenAI
                    "detail": "auto"
                }
            },
            # Store path for Gemini if needed, or just rely on uploaded_image_path
            # For now, the get_gemini_response will look at st.session_state.uploaded_image_path
        ]
        # append_message("user", image_msg_content) # This adds *only* the image
        # It's better to add image along with text prompt if any.
        # This function will now just prepare the image path and base64.
        # The image will be combined with text in the main input logic.

        st.session_state.image_base64_for_display = image_base64 # For displaying in UI
        st.session_state.image_mime_type_for_display = mime_type
        
        # debug_log(f"圖片消息準備添加 (OpenAI format): {image_url_for_openai[:100]}...")
        # No rerun here, let main input logic handle it.
        
    except Exception as e:
        st.write(f"添加圖片消息失敗：{str(e)}")
        debug_error(f"圖片處理異常: {traceback.format_exc()}")


def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")
    # Keep other states unless explicitly cleared by "Clear Memory"
    st.success("Memory cleared!")
    debug_log("Conversation history cleared.")

def execute_code(code, global_vars=None):
    try:
        exec_globals = global_vars if global_vars else {}
        # Ensure matplotlib is available for plotting in executed code
        exec_globals['plt'] = plt
        exec_globals['pd'] = pd
        exec_globals['st'] = st # Allow executed code to use st.pyplot, etc.
        
        debug_log("Ready to execute the following code:")
        if st.session_state.get("debug_mode", False):
            st.session_state.debug_logs.append(f"```python\n{code}\n```")
        debug_log(f"Executing code with global_vars: {list(exec_globals.keys())}")
        
        # Capture stdout for print statements in exec
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(code, exec_globals)
        
        sys.stdout = old_stdout # Restore stdout
        
        captured_prints = redirected_output.getvalue()
        
        # Check if a figure was generated and not shown by st.pyplot in exec'd code
        # This part can be tricky if exec'd code already calls st.pyplot()
        # For now, rely on the exec'd code to call st.pyplot(fig)
        
        output_from_exec = exec_globals.get("output", "") # If exec specifically sets 'output'
        
        final_output = "Code executed."
        if captured_prints:
            final_output += f"\nPrint output:\n{captured_prints}"
        if output_from_exec:
            final_output += f"\nReturned output variable: {output_from_exec}"

        # If st.pyplot was used, the plot is already on the page.
        # If a fig object was created but not shown, we could try to show it,
        # but it's better to enforce `st.pyplot(fig)` in the generated code.

        debug_log(f"Execution output: {final_output}")
        return final_output
    except Exception as e:
        error_msg = f"Error executing code:\n{traceback.format_exc()}"
        debug_error(f"Execution error: {error_msg}") # Use debug_error for consistency
        if st.session_state.get("debug_mode", False):
            return error_msg
        else:
            return "Error executing code (hidden in non-debug mode). Check debug logs if enabled."


def extract_json_block(response: str) -> str:
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        debug_log(f"Extracted JSON block: {json_str}")
        return json_str
    else:
        debug_log("No JSON block found in response.")
        return response.strip() # Return original if no JSON block

# ------------------------------
# LLM Response Methods
# ------------------------------

def get_gemini_response(model_params, max_retries=3):
    api_key = st.session_state.get("gemini_api_key_input", "") or os.getenv("GEMINI_API_KEY")
    debug_log(f"gemini api key found: {'Yes' if api_key else 'No'}")
    if not api_key:
        st.error("未設定 Gemini API 金鑰")
        return "Gemini API 金鑰未設定。"
    
    genai.configure(api_key=api_key)
    model_name = model_params.get("model", "gemini-1.5-flash")
    debug_log(f"gemini model: {model_name}")
    
    # For Gemini, system prompts are handled differently.
    # We'll prepend it to the user's first message or treat it as part of the history.
    # The genai.GenerativeModel().start_chat(history=...) is one way.
    # For generate_content, it's simpler: just pass the content list.

    gemini_messages = []
    has_system_prompt = False
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model" # Gemini uses 'user' and 'model'
        if msg["role"] == "system": # Convert system prompt
            # Gemini's `generate_content` doesn't have a separate system role in the same way as OpenAI.
            # Prepend to the next user message or handle as special instruction.
            # For chat history, it can be included. Let's try to include it as a 'user' turn with a note.
            # Or, use the system_instruction parameter if available for the model version.
            # For now, let's try to pass it. The `start_chat` method might handle it.
            # If we use `generate_content` directly, system prompt needs to be part of `contents`.
            # Let's assume for now system prompt is an instruction within the user message or handled by chat.
            # This part needs careful mapping to Gemini's API structure.
            # For simplicity, let's add a general instruction for Gemini.
            if not has_system_prompt: # Add a general one-time instruction
                 gemini_messages.append({"role": "user", "parts": ["請始終以繁體中文回答，並且所有回覆不用在開頭加上 #zh-tw。"]})
                 has_system_prompt = True

        parts = []
        if isinstance(msg["content"], list): # Likely image content for OpenAI
            text_content_parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content_parts.append(item["text"])
                # Image handling for Gemini happens here if we pass base64
                # However, the current code uploads the image separately if st.session_state.uploaded_image_path exists
            if text_content_parts:
                 parts.append(" ".join(text_content_parts))

        elif isinstance(msg["content"], str):
            parts.append(msg["content"])
        
        if parts: # Only add if there's content for this role
             gemini_messages.append({"role": role, "parts": parts})

    # Image handling: If an image was uploaded, Gemini prefers it as a "Part"
    # The `add_user_image` function saves to `st.session_state.uploaded_image_path`
    # Let's check the last user message and if an image path is set.
    current_content_for_gemini = []
    
    # Add text from the last user message (which is the current prompt)
    # The `gemini_messages` already contains history. We are constructing the *current* call.
    last_user_prompt_text = ""
    if gemini_messages and gemini_messages[-1]["role"] == "user":
        last_user_prompt_text = " ".join(gemini_messages[-1]["parts"]) # Assuming parts are strings
        # We'll use the history as context and send the last prompt with image if present

    # Prepare content for the API call
    api_call_contents = []
    # Add historical messages (excluding the very last user prompt if we're sending it separately with an image)
    # For a simple generate_content call with history:
    history_for_api = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        content_parts = []
        if isinstance(msg["content"], str):
            content_parts.append(msg["content"])
        elif isinstance(msg["content"], list): # OpenAI image format
            temp_text = ""
            has_image_in_this_message = False
            for item_part in msg["content"]:
                if item_part["type"] == "text":
                    temp_text += item_part["text"] + " "
                elif item_part["type"] == "image_url":
                    has_image_in_this_message = True # Mark that this message had an image
            if temp_text:
                content_parts.append(temp_text.strip())
            if has_image_in_this_message and st.session_state.get("uploaded_image_path"):
                 # If this historical message had an image, and we have a path, try to load it
                 # This part is complex if images are in history; for now, focus on current image
                 pass # Simplified: only current image is actively processed this way

        if content_parts:
            history_for_api.append({"role": role, "parts": content_parts})


    # Current request: text + optional image
    current_request_parts = []
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_input_content = st.session_state.messages[-1]["content"]
        if isinstance(user_input_content, str):
            current_request_parts.append(user_input_content)
        elif isinstance(user_input_content, list): # OpenAI multimodal format
            for item in user_input_content:
                if item["type"] == "text":
                    current_request_parts.append(item["text"])
                # Image for current prompt is handled below via uploaded_image_path

    if st.session_state.get("uploaded_image_path"):
        try:
            debug_log(f"Gemini: Adding image from path: {st.session_state.uploaded_image_path}")
            # Check if the image path is still valid, e.g., from current session upload
            if os.path.exists(st.session_state.uploaded_image_path):
                image_part = genai.upload_file(path=st.session_state.uploaded_image_path)
                # Prepend image part to the current request parts
                current_request_parts.insert(0, image_part) 
                debug_log(f"Gemini: Image part prepared: {image_part.uri}")
                 # Crucially, clear uploaded_image_path after processing to avoid re-sending with next message
                st.session_state.uploaded_image_path = None
            else:
                debug_log(f"Gemini: Image path {st.session_state.uploaded_image_path} not found. Skipping image.")
        except Exception as e:
            debug_error(f"Gemini: Failed to upload image: {e}")
            st.warning("無法處理上傳的圖片給Gemini。")

    # Construct final content for Gemini API
    # Option 1: Use start_chat and send_message (better for conversational context)
    # Option 2: Use generate_content with a list of all messages formatted for Gemini
    
    # Using generate_content approach for simplicity with history + current request
    final_api_payload = []
    # Add all but the last message from history_for_api, as current_request_parts is the last user turn
    if len(history_for_api) > 1 :
        final_api_payload.extend(history_for_api[:-1])
    
    if current_request_parts: # Add the current user turn (text + image if any)
        final_api_payload.append({"role": "user", "parts": current_request_parts})
    else: # Fallback if current_request_parts is empty but history_for_api had the last turn
        if history_for_api:
             final_api_payload.append(history_for_api[-1])


    if not final_api_payload:
        debug_log("Gemini: No content to send.")
        return "沒有內容可以發送給 Gemini。"

    # Add system instruction if not already present and model supports it
    system_instruction_content = "請始終以繁體中文回答，並且所有回覆不用在開頭加上 #zh-tw。"
    # For "gemini-1.5-pro" and "gemini-1.5-flash" (and newer), system_instruction is a param for GenerativeModel
    # For older models, it might need to be part of the user message.
    model_obj = None
    if "1.5" in model_name: # Assuming 1.5 and newer support system_instruction
        model_obj = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction_content,
            generation_config=genai.types.GenerationConfig(
                temperature=model_params.get("temperature", 0.5),
                max_output_tokens=model_params.get("max_tokens", 8192) # Gemini 1.5 Flash has 8192 default
            )
        )
    else: # Older models might not have system_instruction
        model_obj = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=model_params.get("temperature", 0.5),
                max_output_tokens=model_params.get("max_tokens", 2048) # Older models might have smaller limits
            )
        )
        # Prepend system instruction to the payload if model doesn't support it directly
        if final_api_payload and final_api_payload[0]['role'] == 'user':
            if isinstance(final_api_payload[0]['parts'], list):
                 final_api_payload[0]['parts'].insert(0, system_instruction_content)
            else: # should be list
                 final_api_payload[0]['parts'] = [system_instruction_content, final_api_payload[0]['parts']]

        elif not final_api_payload:
            final_api_payload.append({'role':'user', 'parts':[system_instruction_content]})
        else: # first message is not user, insert new user message
            final_api_payload.insert(0,{'role':'user', 'parts':[system_instruction_content]})



    debug_log(f"Gemini final_api_payload: {json.dumps(final_api_payload, default=lambda o: '<non-serializable>', indent=2)}")

    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            debug_log(f"Calling Gemini model.generate_content (attempt {retries + 1})")
            response = model_obj.generate_content(final_api_payload)
            final_reply = response.text.strip()
            debug_log(f"Gemini raw response: {response}")
            debug_log(f"Gemini final reply: {final_reply}")
            return final_reply
        except Exception as e:
            debug_error(f"Gemini API request error: {e}")
            if "API key not valid" in str(e):
                st.error("Gemini API 金鑰無效。")
                return "Gemini API 金鑰無效。"
            if "rate limit" in str(e).lower() or "429" in str(e):
                st.warning(f"Gemini 請求頻繁，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            else:
                st.error(f"Gemini API 請求異常: {e}")
                return f"Gemini API 請求異常: {e}"
    st.error("Gemini 請求失敗次數過多，請稍後重試。")
    return "Gemini 請求失敗次數過多。"


def get_openai_response(client, model_params, max_retries=3):
    """處理OpenAI API請求"""
    if not client:
        st.error("OpenAI client 未初始化。請檢查 API 金鑰。")
        return "OpenAI client 未初始化。"

    retries = 0
    wait_time = 5
    model_name = model_params.get("model", "gpt-4o") # Default to gpt-4o
    
    # Prepare messages for OpenAI, ensuring multimodal content is correctly formatted
    openai_messages = []
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user" and st.session_state.get("uploaded_image_path_for_openai"): # Check for image to send with this user message
            # This logic assumes add_user_image stores the base64 data appropriately
            # And that the main loop combines text with this image data.
            # Let's refine how image is added to messages for OpenAI:
            # The `append_message("user", user_input_content)` in main needs to build this structure.
            # For now, assume `msg["content"]` can be a list for multimodal.
            pass # The message content should already be formatted by the input logic

        openai_messages.append({"role": role, "content": content})

    debug_log(f"OpenAI request messages: {json.dumps(openai_messages, default=lambda o: '<non-serializable>', indent=2)}")

    while retries < max_retries:
        try:
            request_params = {
                "model": model_name,
                "messages": openai_messages, # Use the processed messages
                "temperature": model_params.get("temperature", 0.5), # Adjusted default
                "max_tokens": model_params.get("max_tokens", 4096), # GPT-4o can handle more, but let's be reasonable
                "stream": False
            }
            # Check if any message content is a list (indicates multimodal)
            # if any(isinstance(msg.get("content"), list) for msg in openai_messages):
            #     request_params["max_tokens"] = 4096 # Vision models might have different pricing / token needs
            #     debug_log("Detected multimodal input for OpenAI, ensuring max_tokens is appropriate.")

            response = client.chat.completions.create(**request_params)
            response_content = response.choices[0].message.content.strip()
            debug_log(f"OpenAI原始回應：\n{response_content}")
            return response_content
        except Exception as e:
            debug_error(f"OpenAI請求異常 (嘗試 {retries+1}/{max_retries})：{e}")
            if 'rate limit' in str(e).lower() or '429' in str(e):
                st.warning(f"OpenAI 請求過於頻繁，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif 'invalid api key' in str(e).lower():
                st.error("OpenAI API金鑰無效，請檢查後重試")
                return "OpenAI API金鑰無效。"
            elif "maximum context length" in str(e).lower():
                st.error("OpenAI 模型達到最大內容長度限制。請嘗試清除記憶或縮短提示。")
                return "OpenAI 內容長度超出限制。"
            else:
                st.error(f"OpenAI 請求發生錯誤：{str(e)}")
                return f"OpenAI 請求錯誤: {str(e)}"
    debug_error(f"OpenAI 超過最大重試次數（{max_retries}次）")
    st.error("OpenAI 請求失敗次數過多，請稍後再試")
    return "OpenAI 請求失敗次數過多。"

def get_claude_response(model_params, max_retries=3):
    debug_log("Claude loading")
    try:
        import anthropic # Ensure anthropic is imported
    except ImportError:
        st.error("Anthropic SDK 未安裝。請執行 `pip install anthropic`。")
        return "Anthropic SDK 未安裝。"

    api_key = st.session_state.get("claude_api_key_input") or os.getenv("ANTHROPIC_API_KEY")
    debug_log(f"Claude API key found: {'Yes' if api_key else 'No'}")
    if not api_key:
        st.error("未設定 Claude API 金鑰，請在側邊欄輸入")
        return "Claude API 金鑰未設定。"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"初始化 Claude client 失敗: {e}")
        debug_error(f"Claude client init error: {e}")
        return "初始化 Claude client 失敗。"
        
    model_name = model_params.get("model", "claude-3-haiku-20240307") # Default to Haiku for speed
    max_tokens = model_params.get("max_tokens", 4096) # Claude 3 models support large contexts
    temperature = model_params.get("temperature", 0.5) # Adjusted default
    
    # Claude messages format:
    # [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there."}]
    # System prompt is a separate parameter in `client.messages.create`
    claude_messages = []
    system_prompt_claude = "請始終以繁體中文回答。" # Default system prompt

    # Extract system prompt and convert messages
    temp_messages_for_claude = []
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            system_prompt_claude = msg["content"] # Override default if system prompt exists
            continue # Don't add system prompt to claude_messages list

        role = msg["role"]
        content = msg["content"]

        # Claude API expects content to be a string or a list of blocks (text, image)
        if isinstance(content, list): # Likely OpenAI multimodal format
            claude_content_blocks = []
            for item in content:
                if item["type"] == "text":
                    claude_content_blocks.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url" and "base64," in item["image_url"]["url"]:
                    try:
                        base64_data = item["image_url"]["url"].split("base64,")[1]
                        media_type = item["image_url"]["url"].split(":")[1].split(";")[0]
                        claude_content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            }
                        })
                    except Exception as e:
                        debug_error(f"Claude: Error processing image data for message: {e}")
            if claude_content_blocks:
                temp_messages_for_claude.append({"role": role, "content": claude_content_blocks})
        elif isinstance(content, str):
            temp_messages_for_claude.append({"role": role, "content": content})
    
    # Ensure alternating user/assistant messages if possible, Claude is strict
    # This is a simplified approach; true robust conversation history conversion can be complex.
    # For now, we'll pass them as they are. If there are consecutive 'user' or 'assistant' roles, Claude might error.
    # A common pattern is to merge consecutive messages of the same role.
    if not temp_messages_for_claude:
        debug_log("Claude: No messages to send after filtering.")
        return "沒有內容可以發送給 Claude。"
    
    # Filter out any messages that might have ended up empty after conversion
    claude_messages = [m for m in temp_messages_for_claude if m.get("content")]

    if not claude_messages:
        debug_log("Claude: All messages were empty after conversion.")
        return "轉換後所有訊息為空。"


    debug_log(f"Claude request system_prompt: {system_prompt_claude}")
    debug_log(f"Claude request messages: {json.dumps(claude_messages, default=lambda o: '<non-serializable>', indent=2)}")
    
    retries = 0
    wait_time = 5
    while retries < max_retries:
        try:
            debug_log(f"Calling Claude with model={model_name} (attempt {retries + 1})")
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt_claude, # System prompt here
                messages=claude_messages
            )
            completion = response.content[0].text.strip()
            debug_log(f"Claude 回應：{completion}")
            return completion
        except anthropic.APIStatusError as e: # More specific error handling
            debug_error(f"Claude API Status Error (attempt {retries+1}/{max_retries}): {e.status_code} - {e.message}")
            if e.status_code == 429: # Rate limit
                st.warning(f"Claude 請求過於頻繁，{wait_time}秒後重試...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
            elif e.status_code == 401: # Authentication error
                st.error("Claude API 金鑰無效或權限不足。")
                return "Claude API 金鑰無效。"
            elif e.status_code == 400 and "Invalid Request" in str(e.message) and "messages: roles must alternate between \"user\" and \"assistant\"" in str(e.message):
                 st.error("Claude 請求錯誤：訊息角色必須在 user 和 assistant 之間交替。請嘗試清除記憶或調整提示。")
                 debug_error(f"Claude message role order error: {e.message}")
                 return "Claude 訊息角色順序錯誤。"
            else:
                st.error(f"Claude API 狀態錯誤：{e.message}")
                return f"Claude API 狀態錯誤: {e.message}"
        except Exception as e:
            debug_error(f"Claude API 請求異常（嘗試 {retries+1}/{max_retries}）：{e}")
            st.warning(f"Claude 生成錯誤，{wait_time}秒後重試...") # Generic fallback
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2
            # Fallback for other errors not caught by APIStatusError
            if retries == max_retries:
                st.error(f"Claude API 請求異常，已達最大重試次數: {e}")
                return f"Claude API 請求異常: {e}"


    st.error("Claude 請求失敗次數過多，請稍後再試")
    return "Claude 請求失敗次數過多。"


def get_llm_response(client, model_params, max_retries=3):
    """獲取LLM模型回覆（支持OpenAI, Gemini, Claude）"""
    model_name = model_params.get("model", "gpt-4o") # Default if not specified
    debug_log(f"Starting to get llm response for model: {model_name}")
    
    if not st.session_state.messages:
        debug_log("No messages in session_state to send to LLM.")
        return "沒有訊息可以發送。"

    if "gpt" in model_name.lower():
        debug_log("Dispatching to OpenAI")
        return get_openai_response(client, model_params, max_retries)
    elif "gemini" in model_name.lower():
        debug_log("Dispatching to Gemini")
        # Gemini client is initialized within its function using its specific API key
        return get_gemini_response(model_params, max_retries)
    elif "claude" in model_name.lower():
        debug_log("Dispatching to Claude")
        # Claude client is initialized within its function
        return get_claude_response(model_params, max_retries)
    else:
        st.error(f"不支援的模型類型: {model_name}")
        return f"不支援的模型類型: {model_name}"

# ------------------------------
# 更新後的多模型交叉驗證函數
# ------------------------------

def get_cross_validated_response(client, model_params_validator, max_retries=3):
    """
    多模型交叉驗證：
    1. 在記憶流中添加一則系統提示，要求所選驗證模型使用全部對話記憶進行交叉驗證，
       清楚說明其任務：檢查先前回答的正確性、指出潛在錯誤並提供數據或具體理由支持。
    2. 呼叫所選的驗證模型 (可以是 GPT, Gemini, Claude) 獲取回答。
    3. 移除該系統提示後返回驗證模型的回應結果。
    """
    cross_validation_prompt_content = (
        "請仔細閱讀以下全部對話記憶，對先前模型的回答進行交叉驗證。"
        "你的任務是檢查回答的正確性，指出其中可能存在的錯誤或不足，"
        "並提供具體的數據、理由或例子來支持你的分析。"
        "請務必使用繁體中文回答。"
        "在回答時請回答的詳細，內容需要你盡可能的多。"
        "並且越漂亮越好。"
    )
    cross_validation_prompt = {
        "role": "system",
        "content": cross_validation_prompt_content
    }
    
    st.session_state.messages.insert(0, cross_validation_prompt)
    debug_log(f"Cross-validation prompt added. Validating with model: {model_params_validator.get('model')}")

    validator_response_text = get_llm_response(client, model_params_validator, max_retries)

    # 移除剛剛添加的系統提示，以免影響後續對話
    if st.session_state.messages and st.session_state.messages[0]["role"] == "system" and \
       st.session_state.messages[0]["content"] == cross_validation_prompt_content:
        st.session_state.messages.pop(0)
        debug_log("Cross-validation prompt removed.")
    else:
        debug_log("Cross-validation prompt not found at expected position or content changed, not removed.")

    final_response = {
        "validator_response": validator_response_text
    }
    return final_response


# ------------------------------
# 主應用入口
# ------------------------------

def main():
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory + 🖋️ Canvas (Debug & Deep & Cross-Validation)")

    # Initialize session state variables
    defaults = {
        "messages": [], "ace_code": "", "editor_location": "Sidebar",
        "uploaded_file_path": None, "uploaded_image_path": None, 
        "uploaded_image_path_for_openai": None, # For specific use with OpenAI if needed
        "image_base64_for_display": None, "image_mime_type_for_display": None,
        "debug_mode": False, "deep_analysis_mode": True,
        "second_response": "", "third_response": "", "deep_analysis_image": None,
        "debug_logs": [], "debug_errors": [], "thinking_protocol": None,
        "conversation_initialized": False,
        "openai_api_key_input": os.getenv("OPENAI_API_KEY", ""),
        "gemini_api_key_input": os.getenv("GEMINI_API_KEY", ""),
        "claude_api_key_input": os.getenv("ANTHROPIC_API_KEY", ""),
        "last_uploaded_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.subheader("🔑 API Key Settings")
        st.session_state.openai_api_key_input = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key_input, 
            type="password", 
            key="openai_key_widget" # Unique key for widget
        )
        st.session_state.gemini_api_key_input = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_api_key_input, 
            type="password", 
            key="gemini_key_widget"
        )
        st.session_state.claude_api_key_input = st.text_input(
            "Claude API Key", 
            value=st.session_state.claude_api_key_input, 
            type="password", 
            key="claude_key_widget"
        )

        # Update os.environ immediately if keys are provided, useful for some libraries
        if st.session_state.openai_api_key_input:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key_input
        if st.session_state.gemini_api_key_input:
            os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key_input # genai uses this
        if st.session_state.claude_api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = st.session_state.claude_api_key_input


        selected_model = st.selectbox(
            "選擇模型", 
            LLM_MODELS, 
            index=0, 
            key="selected_model"
        )
        
        # API Key check for selected model (visual cue)
        current_model_lower = selected_model.lower()
        show_key_warning = False
        if "gpt" in current_model_lower and not st.session_state.openai_api_key_input:
            show_key_warning = True
        elif "gemini" in current_model_lower and not st.session_state.gemini_api_key_input:
            show_key_warning = True
        elif "claude" in current_model_lower and not st.session_state.claude_api_key_input:
            show_key_warning = True
        
        if show_key_warning:
             st.warning(f"使用 {selected_model} 模型需要在上方輸入對應的 API 金鑰 🔑")


        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode", value=st.session_state.deep_analysis_mode)

        if not st.session_state.conversation_initialized:
            if st.session_state.openai_api_key_input or st.session_state.gemini_api_key_input or st.session_state.claude_api_key_input:
                st.session_state.conversation_initialized = True
                debug_log("Conversation marked as initialized due to API key presence.")
            else:
                st.info("⬅️ 請在側邊欄輸入至少一個平台的 API 金鑰以開始使用。")


        if st.button("🗑️ Clear All Memory & State"):
            keys_to_clear = list(defaults.keys()) # Get all keys we initialized
            for key in keys_to_clear:
                if key in st.session_state: # Make sure key exists before popping
                    # Reset to initial default values
                    st.session_state[key] = defaults[key]
            
            # Clear uploaded files directory content as well
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    file_path_to_delete = os.path.join(UPLOAD_DIR, filename)
                    try:
                        os.remove(file_path_to_delete)
                        debug_log(f"Deleted {file_path_to_delete} from cache.")
                    except Exception as e:
                        debug_error(f"Error deleting {file_path_to_delete}: {e}")

            st.success("所有記憶和狀態已清除!")
            debug_log("All memory and state have been cleared.")
            st.rerun()


        st.subheader("🧠 Memory State")
        if st.session_state.messages:
            memory_content_display = []
            for msg in st.session_state.messages:
                role = msg['role']
                content = msg['content']
                if isinstance(content, list): # Multimodal content
                    text_parts = [item['text'] for item in content if item['type'] == 'text']
                    text_summary = " ".join(text_parts)
                    num_images = sum(1 for item in content if item['type'] == 'image_url')
                    display_content = f"{text_summary} (+{num_images} image(s))" if num_images > 0 else text_summary
                else:
                    display_content = str(content)
                memory_content_display.append(f"{role}: {display_content[:100]}...") # Truncate long content
            st.text_area("Current Memory (Summary)", value="\n".join(memory_content_display), height=200, disabled=True)
        else:
            st.text_area("Current Memory", value="No messages yet.", height=200, disabled=True)

        st.subheader("📂 Upload a CSV File")
        uploaded_csv_file = st.file_uploader("Choose a CSV file:", type=["csv"], key="csv_uploader")
        if uploaded_csv_file:
            st.session_state.uploaded_file_path = save_uploaded_file(uploaded_csv_file)
            debug_log(f"Uploaded CSV file path: {st.session_state.uploaded_file_path}")
            try:
                csv_data = pd.read_csv(st.session_state.uploaded_file_path)
                st.write("### Data Preview (CSV)")
                st.dataframe(csv_data.head())
            except Exception as e:
                st.error(f"讀取 CSV 錯誤: {e}")
                debug_error(f"Error reading CSV: {e}")

        st.subheader("🖼️ Upload an Image")
        uploaded_image_file = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"], key="image_uploader_widget")
        if uploaded_image_file:
            # Process the image (save, base64 encode) but don't add to messages yet.
            # The main input logic will combine it with text if provided.
            add_user_image(uploaded_image_file) # This now sets session_state variables
            if st.session_state.get("image_base64_for_display"):
                st.image(
                    f"data:{st.session_state.image_mime_type_for_display};base64,{st.session_state.image_base64_for_display}",
                    caption=f"已上傳: {st.session_state.last_uploaded_filename}",
                    use_container_width=True
                )
                debug_log(f"Uploaded image {st.session_state.last_uploaded_filename} processed and previewed.")
                # Important: This image is now "staged". It will be sent with the *next* text prompt.


        st.subheader("🧠 Upload Thinking Protocol")
        uploaded_thinking_protocol = st.file_uploader("Choose a thinking_protocol.md file:", type=["md"], key="thinking_protocol_uploader")
        if uploaded_thinking_protocol:
            try:
                thinking_protocol_content = uploaded_thinking_protocol.read().decode("utf-8")
                st.session_state.thinking_protocol = thinking_protocol_content
                # Add to messages immediately as a user prompt or system instruction
                append_message("user", f"已載入思考協議:\n{thinking_protocol_content}")
                st.success("Thinking Protocol uploaded and added to messages!")
                debug_log("Thinking Protocol uploaded and added to messages.")
                st.rerun() # Rerun to display the new message
            except Exception as e:
                st.error(f"讀取 Thinking Protocol 錯誤: {e}")
                debug_error(f"Error reading Thinking Protocol: {e}")

        st.subheader("Editor Location")
        st.session_state.editor_location = st.radio(
            "Choose where to display the editor:",
            ["Main", "Sidebar"],
            index=1 if st.session_state.editor_location == "Sidebar" else 0,
            key="editor_loc_radio"
        )

        with st.expander("🛠️ 調試與會話資訊", expanded=False):
            if st.session_state.debug_mode:
                st.subheader("調試日誌")
                st.text_area("Debug Logs", value="\n".join(st.session_state.debug_logs), height=200,disabled=True)
                st.subheader("調試錯誤")
                st.text_area("Debug Errors", value="\n".join(st.session_state.debug_errors), height=200,disabled=True)
            
            st.subheader("會話資訊 (messages.json)")
            if st.session_state.messages:
                try:
                    messages_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4, default=lambda o: "<non-serializable>")
                    st.text_area("messages.json (raw)", value=messages_json, height=300, disabled=True)
                    st.download_button(
                        label="📥 下載 messages.json",
                        data=messages_json,
                        file_name="messages.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"無法序列化 messages: {e}")
            else:
                st.write("沒有找到 messages。")


    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, list): # Multimodal content (OpenAI format)
                for item in content:
                    if item["type"] == "text":
                        st.write(item["text"])
                    elif item["type"] == "image_url":
                        # Display image from base64 URL
                        st.image(item["image_url"]["url"], caption="🖼️ (圖片)", use_container_width=True)
                        debug_log(f"Displaying image from {message['role']} via base64 URL.")
            elif isinstance(content, str):
                if "```python" in content:
                    # This is a simple check. For robustness, use re. DOTALL for multiline code blocks.
                    code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
                    if code_match:
                        code_to_display = code_match.group(1).strip()
                        st.code(code_to_display, language="python")
                        # Display text before/after code block if any
                        before_code = content.split("```python")[0]
                        after_code = content.split("```",2)[-1] if content.count("```") >=2 else ""
                        if before_code.strip(): st.write(before_code)
                        # if after_code.strip() and not after_code.startswith("python"): st.write(after_code) # avoid printing python itself
                    else:
                        st.write(content) # Fallback if regex fails
                else:
                    st.write(content)
            else: # Should not happen if content is standardized
                 st.write(str(content))


    user_input = st.chat_input("哈囉！請問有什麼可以幫您的嗎？(可上傳圖片後在此提問)")
    if user_input:
        user_input_content = []
        user_input_content.append({"type": "text", "text": user_input})

        # Check if an image was staged (uploaded via sidebar) and ready to be sent
        if st.session_state.get("uploaded_image_path") and st.session_state.get("image_base64_for_display"):
            image_base64, mime_type = load_image_base64(st.session_state.uploaded_image_path) # Get fresh base64
            if image_base64:
                image_url_for_openai = f"data:{mime_type};base64,{image_base64}"
                user_input_content.insert(0, { # Image first, then text usually
                    "type": "image_url",
                    "image_url": {"url": image_url_for_openai, "detail": "auto"}
                })
                debug_log(f"Adding staged image {st.session_state.last_uploaded_filename} to user prompt.")
                # Mark that this specific image is now being sent with OpenAI format
                st.session_state.uploaded_image_path_for_openai = st.session_state.uploaded_image_path
                # For Gemini, uploaded_image_path will be used directly in get_gemini_response
                # It will be cleared there after use.
            else:
                debug_log("Staged image found but failed to load base64 for OpenAI.")
        
        append_message("user", user_input_content if len(user_input_content) > 1 else user_input_content[0]["text"])
        # Displaying user message is handled by the loop above after rerun or directly:
        with st.chat_message("user"):
            for item_content in user_input_content:
                if item_content["type"] == "text":
                    st.write(item_content["text"])
                elif item_content["type"] == "image_url":
                    st.image(item_content["image_url"]["url"], caption="🖼️ 您上傳的圖片", use_container_width=True)


        with st.spinner("思考中..."):
            try:
                # Ensure API key for selected model is available
                active_openai_key = st.session_state.openai_api_key_input
                active_gemini_key = st.session_state.gemini_api_key_input
                active_claude_key = st.session_state.claude_api_key_input
                
                model_ok = False
                current_model_lower = selected_model.lower()
                if "gpt" in current_model_lower and active_openai_key: model_ok = True
                elif "gemini" in current_model_lower and active_gemini_key: model_ok = True
                elif "claude" in current_model_lower and active_claude_key: model_ok = True
                
                if not model_ok:
                    st.error(f"請先在側邊欄輸入 {selected_model} 對應的 API 金鑰。")
                    st.stop()

                client = initialize_client(active_openai_key) # For OpenAI

                # Add system prompt if it's the first real message exchange (after initial empty or setup)
                is_first_meaningful_exchange = not any(msg["role"] == "assistant" for msg in st.session_state.messages[:-1]) # Exclude current user msg
                if not any(msg["role"] == "system" for msg in st.session_state.messages) and is_first_meaningful_exchange:
                    system_prompt = "You are a helpful assistant. Please respond in Traditional Chinese (繁體中文)."
                    # Prepend system prompt
                    st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
                    debug_log("Default system prompt added to messages.")

                # Construct prompt for CSV analysis if file is uploaded
                prompt_to_llm = user_input # Default
                if st.session_state.uploaded_file_path and ( "分析" in user_input or "資料" in user_input or "圖" in user_input or "csv" in user_input.lower()):
                    try:
                        df_temp = pd.read_csv(st.session_state.uploaded_file_path)
                        csv_columns = ", ".join(df_temp.columns)
                        debug_log(f"CSV columns for prompt: {csv_columns}")
                        
                        # Update the prompt for the LLM to request JSON and code specifically
                        prompt_for_csv_task = f"""
任務：根據使用者需求分析提供的 CSV 檔案。CSV 檔案路徑為 st.session_state.uploaded_file_path。
使用者需求: "{user_input}"
可用的欄位: {csv_columns}

請回覆一個 JSON 物件，包含以下兩個鍵：
1. "content": (字串) 對於使用者需求的文字分析、觀察與總結。
2. "code": (字串) 用於生成相關圖表或進行數據操作的 Python 程式碼。

重要指示：
- 程式碼必須使用 `st.session_state.uploaded_file_path` 讀取 CSV。
- 若要繪圖，必須使用 `matplotlib`，並確保圖表中的所有文字（標題、軸標籤、圖例等）都使用字型檔案 `{font_path}`（Microsoft JhengHei - 微軟正黑體）顯示。程式碼中應包含設定字型的邏輯。
- 使用 `st.pyplot(fig)` 來顯示 Matplotlib 圖表。
- 圖表顏色請使用預設以外的色彩搭配，讓視覺效果更佳。
- 程式碼應考慮繪圖的美觀性。
- 所有文字回覆（包含 `content` 欄位）都必須使用繁體中文。
- 如果使用者要求特定資訊，盡可能用圖表呈現。

範例程式碼片段設定字型：
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = "{font_path}" # 確保此路徑在執行環境中有效
if os.path.exists(font_path):
    my_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = my_font.get_name()
else:
    print(f"Font file not found at {font_path}, using default.")
plt.rcParams['axes.unicode_minus'] = False
# ... 你的繪圖代碼 ...
# fig, ax = plt.subplots()
# ...
# st.pyplot(fig)
