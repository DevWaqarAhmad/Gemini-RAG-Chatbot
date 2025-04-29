import streamlit as st
from backend import rag_response  


st.set_page_config(page_title="Housess AI Assistant", layout="centered")

st.title("Housess Real Estate AI Assistant")
st.markdown("Housess Real Estate: Where Excellence Meets Your Expectations.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your question about UAE real estate..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using your RAG backend
    with st.chat_message("assistant"):
        with st.spinner("Response Creating..."):
            answer = rag_response(prompt)
        st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})