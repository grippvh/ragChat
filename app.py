#!/bin/env python3
import os
import streamlit as st
from streamlit_chat import message
from rag import RagChat
from config import Config

st.set_page_config(page_title="RagChat")
DATA_FOLDER = Config.DATA_PATH

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text, st.session_state["messages"][-30:])

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""

def update_sources():
    """Update the UI to display the current set of sources in the vector store."""
    sources = st.session_state["assistant"].vector_store_manager.get_sources()
    st.session_state["sources"] = sources

def read_and_save_file():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for file in st.session_state["file_uploader"]:
        file_path = os.path.join(DATA_FOLDER, file.name)

        if os.path.exists(file_path):
            st.warning(f"The file {file.name} already exists in the data folder and will not be uploaded again.")
            continue

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingestor.ingest_file(file_path)

    update_sources()

def ingest_url():
    url = st.session_state["url_input"].strip()
    if url:
        try:
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting content from {url}"):
                st.session_state["assistant"].ingestor.ingest_url(url)

                update_sources()

        except Exception as e:
            st.error(f"Could not open the link: {e}")

def scan_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    existing_files = os.listdir(DATA_FOLDER)
    if existing_files:
        for file_name in existing_files:
            file_path = os.path.join(DATA_FOLDER, file_name)
            if not st.session_state["assistant"].vector_store_manager.vector_store.similarity_search(file_name, k=1):
                st.session_state["assistant"].ingestor.ingest_file(file_path)

    update_sources()

def clear_database_and_move_files():
    st.session_state["assistant"].clear()
    st.success("Database cleared and files moved to 'unused_data'.")
    update_sources()

def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = RagChat()
        st.session_state["ingestion_spinner"] = st.empty()
        st.session_state["sources"] = []
        scan_data_folder()

    st.header("RagChat")

    st.subheader("Upload new document or provide a link")

    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.text_input(
        "Or enter a URL",
        key="url_input",
        on_change=ingest_url,
        label_visibility="collapsed",
        placeholder="Provide your link here",
    )

    st.session_state["ingestion_spinner"] = st.empty()


    if st.button("Clear Database"):
        clear_database_and_move_files()


    st.subheader("Sources in Vector Store")
    if st.session_state["sources"]:
        for source in st.session_state["sources"]:
            st.write(source)
    else:
        st.write("No sources found in the vector store.")

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
