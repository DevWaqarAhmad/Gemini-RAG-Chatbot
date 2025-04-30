import streamlit as st
from backend import rag_response

# Page configuration
st.set_page_config(
    page_title="Housess Real Estate AI Agent",
    layout="wide"
)

# --- Custom Supported Languages ---
SUPPORTED_LANGUAGES = {
    "Auto-Detect": None,
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "Chinese": "zh-cn",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Turkish": "tr",
    "Filipino (Tagalog)": "tl"
}

# Sidebar setup
st.sidebar.title("Settings")
selected_lang_name = st.sidebar.selectbox(
    "Select Language",
    options=SUPPORTED_LANGUAGES.keys(),
    index=0  # Default to Auto-Detect
)
selected_lang_code = SUPPORTED_LANGUAGES[selected_lang_name]

# Refresh chat button
if st.sidebar.button("Refresh Chat"):
    st.session_state.messages = []
    # Clear memory if the chat is refreshed
    st.session_state.memory = []

# Main content
st.title("Housess Real Estate AI Agent")

# Initialize chat history and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []  # This memory will store the conversation summary

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    target_lang = selected_lang_code

    # Check if user wants to summarize the chat
    if "summarize" in prompt.lower():
        summary = "\n".join(st.session_state.memory)  # Show conversation summary
        with st.chat_message("assistant"):
            st.markdown(f"**Chat Summary**:\n{summary}")
        st.session_state.messages.append({"role": "assistant", "content": f"**Chat Summary**:\n{summary}"})
    else:
        # Generate a response based on the prompt
        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                answer = rag_response(prompt, target_lang=target_lang)
            st.markdown(answer)

        # Store assistant response to memory
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.memory.append(answer)  # Add the answer to the memory

# Default greeting if there are no previous messages
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hello! How can I help you today?")
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})
