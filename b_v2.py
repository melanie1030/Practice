# --- 新增：Base64 與 PIL 處理 ---
from PIL import Image
import base64
import io

def load_image_base64(image_path):
    """Convert an image to Base64 encoding using BytesIO + PIL."""
    img = Image.open(image_path)
    buffer = io.BytesIO()
    fmt = img.format if img.format else "PNG"
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_uploaded_image(image_file):
    """Save uploaded image to the specified directory and return its path."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())
    return file_path

# --- 主程式內圖片相關功能 ---
    st.subheader("🖼️ Upload or Capture an Image")

    # --- 圖像上傳 ---
    uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        file_path = save_uploaded_image(uploaded_image)
        st.session_state.uploaded_image_path = file_path

        # 使用 PIL 和 Base64 進行處理
        st.session_state.image_base64 = load_image_base64(file_path)
        st.image(file_path, caption="Uploaded Image Preview", use_column_width=True)
        st.success("Image uploaded and processed successfully!")
        debug_log(f"Image uploaded and saved to: {file_path}")

    # --- 拍照 ---
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        file_path = save_uploaded_image(camera_image)
        st.session_state.uploaded_image_path = file_path

        # 使用 PIL 和 Base64 進行處理
        st.session_state.image_base64 = load_image_base64(file_path)
        st.image(file_path, caption="Captured Image Preview", use_column_width=True)
        st.success("Photo captured and processed successfully!")
        debug_log(f"Photo captured and saved to: {file_path}")
