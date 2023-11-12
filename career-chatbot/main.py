import os
from typing import Set

import streamlit as st
from backend import Store, run_llm, save_vector_store
from langchain.schema import AIMessage, HumanMessage
from streamlit_chat import message

st.title("🤖 Ask me anything about Minjoo")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")


st.markdown(
    """ 
    ####  🗨️ Chat with Minjoo's portfolio 📜  
    """
)


@st.cache_resource
def get_database_session():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    store = Store(sessionId="test1")
    if not os.path.exists("./faiss_index_career"):
        save_vector_store()
    return store


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        
        return ""

    sources_list = list(source_urls)
    sources_list.sort()

    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source} \n"
    return sources_string


store = get_database_session()

if prompt:
    with st.spinner("Generate response"):
        print(prompt)
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        store.update_history(prompt, generated_response["answer"])
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if store.get_history():
    for index, msg in enumerate(store.get_history()):
        if isinstance(msg, HumanMessage):
            message(
                msg.content,
                is_user=True,
                key=f"{index}_user",
                logo="https://api.dicebear.com/7.x/thumbs/svg?seed=Sammy",
            )
        if isinstance(msg, AIMessage):
            message(
                msg.content,
                is_user=False,
                key=str(index),
                logo="https://api.dicebear.com/7.x/thumbs/svg?seed=Kiki",
            )
