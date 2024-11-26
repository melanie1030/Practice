import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai  # 必须导入 openai 模块
import dotenv
import os
from io import BytesIO
import json
from PIL import Image
from datetime import datetime

# --- Initialize and Settings ---
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image_from_gpt_response(response, csv_data):
    """Generate a chart based on GPT's response."""
    try:
        chart_type = response.get("chart_type", "line")  # Default to line chart
        x_column = response.get("x_column", csv_data.columns[0])
        y_column = response.get("y_column", csv_data.columns[1])

        if chart_type == "line":
            plt.figure(figsize=(10, 6))
            plt.plot(csv_data[x_column], csv_data[y_column], marker='o')
            plt.title(f"{y_column} vs {x_column}", fontsize=16)
            plt.xlabel(x_column, fontsize=14)
            plt.ylabel(y_column, fontsize=14)
            plt.grid(True)
        elif chart_type == "bar":
            plt.figure(figsize=(10, 6))
            plt.bar(csv_data[x_column], csv_data[y_column], color='skyblue')
            plt.title(f"{y_column} vs {x_column}", fontsize=16)
            plt.xlabel(x_column, fontsize=14)
            plt.ylabel(y_column, fontsize=14)

        plt.tight_layout()

        # Save chart to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to generate the chart: {e}")
        return None


def save_chat_to_json(messages):
    """Save chat history as a JSON file."""
    try:
        chat_json = json.dumps(messages, ensure_ascii=False, indent=4)
        json_bytes = BytesIO(chat_json.encode("utf-8"))
        json_bytes.seek(0)
        file_name = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            label="Download Chat History",
            data=json_bytes,
            file_name=file_name,
            mime="application/json",
        )
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")


def analyze_csv_with_gpt(df, user_question):
    """
    使用 GPT 分析 CSV 数据。
    将数据的摘要和用户问题发送给 GPT 模型。
    """
    # 将数据的前几行转为 JSON 格式
    data_preview = df.head(5).to_json()
    # 创建 GPT 提示词
    prompt = (
        f"You are a data analyst. Below is a preview of the dataset:\n{data_preview}\n\n"
        f"User question: {user_question}\n"
        f"Provide detailed insights or suggestions based on the data."
    )

    # 调用 GPT 模型
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt},
        ]
    )
    # 返回 GPT 的回答
    return response["choices"][0]["message"]["content"]


def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot with Data & Images", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🖼️ Image Upload")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔐 Enter Your API Key")
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")

        if not api_key:
            st.warning("⬅️ Please enter the API key to continue...")
            return

        # Upload CSV
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

        # Upload Image
        st.subheader("🖼️ Upload an Image")
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Download Chat History
        st.subheader("💾 Export Chat History")
        if st.button("Save Chat History"):
            save_chat_to_json(st.session_state.messages)

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        # Add user input to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Immediately display user input
        with st.chat_message("user"):
            st.write(user_input)

        # Call GPT and display its response
        with st.spinner("Thinking..."):
            try:
                # Modify prompt if CSV is uploaded
                if csv_data is not None:
                    gpt_response = analyze_csv_with_gpt(csv_data, user_input)
                else:
                    # Simple GPT Chat
                    gpt_response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": user_input}]
                    )["choices"][0]["message"]["content"]

                # Add GPT response to conversation history
                st.session_state.messages.append({"role": "assistant", "content": gpt_response})

                # Immediately display GPT response
                with st.chat_message("assistant"):
                    st.write(gpt_response)

                # Generate chart if CSV is uploaded
                if csv_data is not None:
                    with st.spinner("Generating chart from GPT response..."):
                        try:
                            parsed_response = json.loads(gpt_response)
                            chart_buf = generate_image_from_gpt_response(parsed_response, csv_data)
                            if chart_buf:
                                st.image(chart_buf, caption="Generated Chart", use_column_width=True)
                                st.download_button(
                                    label="Download Chart",
                                    data=chart_buf,
                                    file_name="generated_chart.png",
                                    mime="image/png"
                                )
                        except json.JSONDecodeError:
                            st.error("Failed to parse GPT response for chart generation. Please check the response format.")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
