import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
from google.cloud import storage  # Import Google Cloud Storage library
import json
from datetime import datetime

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

def save_to_cloud(bucket_name, file_name, data):
    """Save conversation history to Google Cloud Storage."""
    client = storage.Client()  # Initialize Google Cloud client
    bucket = client.get_bucket(bucket_name)  # Retrieve the bucket
    blob = bucket.blob(file_name)  # Create a new blob for the file
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    return blob.public_url  # Return the public URL of the uploaded file

# --- Chatbot Main Functionality ---

def stream_llm_response(client, model_params):
    """Stream responses from the LLM model."""
    for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-4o"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=4096,
            stream=True):
        chunk_text = chunk.choices[0].delta.content or ""
        yield chunk_text

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
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

    # --- Export Chat History ---
    st.sidebar.write("### Export Chat History")
    bucket_name = st.text_input("Google Cloud Bucket Name")
    if st.sidebar.button("Save Chat History") and bucket_name:
        file_name = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        url = save_to_cloud(bucket_name, file_name, st.session_state.messages)
        st.sidebar.success(f"Chat history saved! [View File]({url})")

# Entry point
if __name__ == "__main__":
    main()
