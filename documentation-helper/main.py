from typing import Set

import streamlit as st
from backend import run_llm
from streamlit_chat import message

# session_state is to store session state between reruns

st.header("Langchain Udemy Course - Document Helper Bot")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""

    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source} \n"
    return sources_string


if prompt:
    with st.spinner("Generate response"):
        print(prompt)
        generate_response = run_llm(query=prompt)
        sources = set(
            [doc.metadata["source"] for doc in generate_response["source_documents"]]
        )  # collection sources
        formatted_response = (
            f"{generate_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
