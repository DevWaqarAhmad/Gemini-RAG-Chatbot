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
    st.session_state.chat_history = []

# Title
st.title("Housess Real Estate AI Agent")

# Initialize chat and memory history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("Ask your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Target language
    target_lang = selected_lang_code

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            answer = rag_response(prompt, chat_history=st.session_state.chat_history, target_lang=target_lang)
        st.markdown(answer)

    # Add both prompt and response to memory and visible messages
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(f"User: {prompt}")
    st.session_state.chat_history.append(f"Bot: {answer}")

# Initial greeting
if len(st.session_state.messages) == 0:
    greeting = "Hello! How can I help you today?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.messages.append({"role": "assistant", "content": greeting})
