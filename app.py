import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from chatbot import configure_retrieval_chain, MEMORY

st.set_page_config(page_title="Chatbot", page_icon="🦜")
st.title("My Personal Chatbot")

uploaded_files = st.sidebar.file_uploader(
    label="Upload File",
    type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please Upload Document Here")
    st.stop()

CONV_CHAIN = configure_retrieval_chain(uploaded_files)

avatars = {"human": "user", "ai": "assistant"}

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("Ask me anything!")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assistant = st.chat_message("assistant")

if user_query := st.chat_input(placeholder="Type your query here..."):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)

    with st.chat_message("assistant"):
        print("Here we are Processing the Conversational Chain!")

        response = CONV_CHAIN.run({
            "question": user_query,
            "chat_history": MEMORY.chat_memory.messages
        },
        callbacks=[stream_handler]
        )
