import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import json

# --- Main Function ---
def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖", layout="centered")
    st.title("🤖 Chatbot with Memory 🧠")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.subheader("🔒 Enter Your API Key")
        api_key = st.text_input("OpenAI API Key", type="password")

        # Initialize LangChain chat model and memory
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
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call LangChain conversation
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.run(user_input)

                # Add assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Save memory and messages to file
    if st.sidebar.button("💾 Save Conversation"):
        save_conversation_to_file()

    # Load memory from file
    if st.sidebar.button("📂 Load Conversation"):
        load_conversation_from_file()


# --- Helper Functions ---
def save_conversation_to_file():
    """Save conversation and memory to a JSON file."""
    try:
        messages = st.session_state.messages
        memory = st.session_state.memory.load_memory_variables({})
        data = {"messages": messages, "memory": memory}

        file_name = "conversation_history.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        st.success(f"Conversation saved to {file_name}")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")


def load_conversation_from_file():
    """Load conversation and memory from a JSON file."""
    try:
        file_name = st.file_uploader("Upload a conversation file", type="json")
        if file_name:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)

            st.session_state.messages = data.get("messages", [])
            memory_data = data.get("memory", {})
            if memory_data:
                st.session_state.memory = ConversationBufferMemory.from_memory_variables(memory_data)
                st.session_state.conversation = ConversationChain(
                    llm=st.session_state.chat_model,
                    memory=st.session_state.memory
                )
            st.success("Conversation loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")


# --- Run Main ---
if __name__ == "__main__":
    main()
