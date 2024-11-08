import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
import json
from datetime import datetime
from langdetect import detect  # Import language detection library

# --- Initialize and Settings ---
dotenv.load_dotenv()

# Define global variables
OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]

# --- Helper Functions ---
def initialize_client(api_key):
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key) if api_key else None

def load_image_base64(image):
    """Convert an image to Base64 encoding."""
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def add_user_image(image):
    """Add an image message to the session state."""
    img_base64 = load_image_base64(image)
    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]
    })

def reset_session_messages():
    """Clear conversation history from the session."""
    if "messages" in st.session_state:
        st.session_state.pop("messages")

def save_chat_to_json(messages):
    """Save chat history as a JSON file and provide a download button."""
    # Convert chat history to JSON string
    chat_json = json.dumps(messages, ensure_ascii=False, indent=4)
    
    # Use BytesIO to create a downloadable file
    json_bytes = BytesIO(chat_json.encode('utf-8'))
    json_bytes.seek(0)
    
    # Provide a download button for the JSON file
    file_name = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button(
        label="下載對話紀錄",
        data=json_bytes,
        file_name=file_name,
        mime="application/json"
    )

# Function to detect the language of the input and adjust the model's response
def detect_language_and_set_response(prompt):
    """Detect the language of the input and adjust the assistant's response."""
    lang = detect(prompt)
    if lang == 'zh-cn':  # Simplified Chinese detected
        return "zh-tw"  # Force the response to Traditional Chinese
    return lang  # Return the detected language (for English, Japanese, etc.)

# --- Chatbot Main Functionality ---
def stream_llm_response(client, model_params, prompt):
    """Stream responses from the LLM model and store them in session state."""
    assistant_response = ""
    # Detect language and set response language accordingly
    response_language = detect_language_and_set_response(prompt)

    # Set the appropriate language for the assistant's response
    for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-4o"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=4096,
            stream=True,
            language=response_language):  # Pass the language to the model
        chunk_text = chunk.choices[0].delta.content or ""
        assistant_response += chunk_text
    
    # Add the assistant's full response to the session state after completion
    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_response}]
    })

    # Display the complete response in the chat window
    with st.chat_message("assistant"):
        st.write(assistant_response)

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>OpenAI Chatbot</i> </h1>""")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        client = initialize_client(api_key)
        if not api_key or not client:
            st.warning("⬅️ Please enter the API key to continue...")
            return

        # Model Settings
        model = st.selectbox("Select Model:", OPENAI_MODELS)
        model_params = {"model": model, "temperature": st.slider("Temperature", 0.0, 2.0, 0.3)}

        # Image Upload and Camera
        st.write("### Upload Image or Take Photo")
        
        # Image Upload
        uploaded_img = st.file_uploader("Select an image:", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img)
            add_user_image(img)
            st.success("Image uploaded!")

        # Take Photo
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            img = Image.open(camera_img)
            add_user_image(img)
            st.success("Photo taken successfully!")

        # Reset Conversation
        st.button("🗑️ Clear Conversation", on_click=reset_session_messages)

    # --- Chat Window & Messages ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages in session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content.get("text", ""))
                elif content["type"] == "image_url":
                    st.image(content["image_url"].get("url", ""))

    # --- User Input ---
    prompt = st.chat_input("Hi! Ask me anything...")
    if prompt:
        # Append user's message to session state
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant's response and display it
        stream_llm_response(client, model_params, prompt)

    # --- Export Chat History ---
    st.sidebar.write("### Export Chat History")
    save_chat_to_json(st.session_state.messages)  # Add download button for chat history

# Entry point
if __name__ == "__main__":
    main()
