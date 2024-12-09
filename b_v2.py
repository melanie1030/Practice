def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot + Data Analysis", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot + 📊 Data Analysis + 🧠 Memory")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        if "conversation" not in st.session_state:
            if api_key:
                st.session_state.chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=api_key)
                st.session_state.memory = ConversationBufferMemory()
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            else:
                st.warning("⬅️ Please enter the API key to initialize the chatbot.")
                return

        # Memory management buttons
        if st.button("🗑️ Clear Memory"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("Memory cleared!")

        # Show current memory state
        st.subheader("🧠 Memory State")
        if "memory" in st.session_state:
            memory_content = st.session_state.memory.load_memory_variables({})
            st.text_area("Current Memory", value=str(memory_content), height=200)

        # Upload CSV
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
        csv_data = None
        if uploaded_file:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(csv_data)

        # Upload Image
        uploaded_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            img_bytes = BytesIO(uploaded_image.read())
            st.session_state.messages.append({"role": "user", "image": img_bytes.getvalue()})
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            if "image" in message:
                img = Image.open(BytesIO(message["image"]))
                st.image(img, caption="Uploaded Image", use_column_width=True)

    # User input
    user_input = st.chat_input("Hi! Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                if csv_data is not None:
                    csv_columns = ", ".join(csv_data.columns)
                    prompt = f"""
                    請根據以下用戶請求，用繁體中文回覆：
                    {{
                        "chart_type": "line", 
                        "x_column": "{csv_data.columns[0]}", 
                        "y_column": "{csv_data.columns[1]}",
                        "response": "請以繁體中文回答此問題。"
                    }}
                    基於以下內容：{user_input}。
                    可用欄位：{csv_columns}。
                    """
                else:
                    prompt = f"請以繁體中文回答此問題：{user_input}"

                response = st.session_state.conversation.run(prompt)

                # 後處理，檢查語言並修正不必要的標記
                if "#zh-tw" in response or not response.startswith("喜歡"):
                    response = response.replace("#zh-tw", "").strip()

                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)

                if csv_data is not None:
                    parsed_response = json.loads(response)
                    chart_buf = generate_image_from_gpt_response(parsed_response, csv_data)
                    if chart_buf:
                        st.image(chart_buf, caption="Generated Chart", use_column_width=True)
                        st.download_button(
                            label="Download Chart",
                            data=chart_buf,
                            file_name="generated_chart.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.sidebar.button("💾 Save Conversation"):
        save_conversation_to_file()
    if st.sidebar.button("📂 Load Conversation"):
        load_conversation_from_file()

if __name__ == "__main__":
    main()
