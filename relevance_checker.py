import os

from llama_index.llms.ollama import Ollama
from config import Config
from llama_index.core.base.llms.types import MessageRole

def is_query_relevant(domain, query, chat_history_messages):
    try:
        files = os.listdir(Config.DATA_PATH)
    except Exception:
        files = []
    # format file names
    file_names = [f"{fname}" for fname in files] if files else []

    # format chat history into readable strings
    formatted_history = []
    if chat_history_messages:
        for msg in chat_history_messages:
            message_text = " ".join(block.text for block in msg.blocks)
            if msg.role == MessageRole.USER:
                role_label = "User"
            elif msg.role == MessageRole.ASSISTANT:
                role_label = "Assistant"
            else:
                role_label = str(msg.role)
            formatted_history.append(f"{role_label}: {message_text}")

    prompt = build_relevance_prompt(domain, query, chat_history=formatted_history, file_names=file_names)

    llm = Ollama(model=Config.MODEL_NAME)

    response = llm.complete(prompt).text.lower().strip()
    print(response)
    return response

def build_relevance_prompt(domain, query, chat_history=None, file_names=None):
    """
    Constructs a prompt that asks if the user query is related to the given domain.
    The prompt includes:
      - The domain information.
      - A list of uploaded file names (if any).
      - The formatted chat history (if any).
      - The user query.

    The prompt instructs the LLM to answer strictly with "yes" or "no".

    Parameters:
        domain (str): The domain context.
        query (str): The user query.
        chat_history (list[str], optional): List of strings representing past conversation lines.
        file_names (list[str], optional): List of file names.

    Returns:
        str: The complete prompt.
    """
    parts = []

    parts.append(
        "You are a relevance checker for a chatbot. A user set domain and may have uploaded the files. Based on the on the information below, your task is to verify whether the user query is related to the domain or not? "
        "you should prioritise domain, but you should allow follow-up questions as well. If the query is related to one of the files, but its clear that is has no connection to the domain, answer 'no'."
        "Answer strictly with 'yes' or 'no' only. ")

    parts.append(f"Domain: {domain}\n")

    if file_names:
        parts.append("Uploaded Files:")
        for fname in file_names:
            parts.append(f"- {fname}")
        parts.append("")

    if chat_history:
        parts.append("Chat History:")
        parts.extend(chat_history)
        parts.append("")

    parts.append(f"User Query: {query}\n")


    return "\n".join(parts)
