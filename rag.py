from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from config import Config
from vector_store import VectorStoreManager
from ingestion import Ingestor
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate


def _build_context(results):
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])


class RagChat:
    def __init__(self, db_path = Config.DB_PATH):
        self.model = ChatOllama(model=Config.MODEL_NAME)
        self.vector_store_manager = VectorStoreManager(db_path, Config.JINA_API_KEY)
        self.ingestor = Ingestor(self.vector_store_manager)
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        self.web_search_tool = DuckDuckGoSearchRun(api_wrapper=self.wrapper)
        self.ragPrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks. Use the following pieces 
                of retrieved context to answer the question. If you don't know the answer, strictly answer 
                with "I don't know" only. Provide a concise answer in four sentences maximum.[/INST] </s> 
                [INST] Question: {question} [/INST]
                [INST] Context: {context} [/INST]
                [INST] Answer: 
                """
        )
        self.basePrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks.
                Provide a concise answer in four sentences maximum. Answer based on your knowledge [/INST] </s> 
                [INST] Question: {question} [/INST]
                [INST] Answer: 
                """
        )
        self.web_search_prompt = PromptTemplate.from_template(
            """
                <s> [INST]  You are a highly knowledgeable assistant for question-answering tasks. Use the following pieces 
                of retrieved context to answer the question. If you don't know the answer, strictly answer 
                with "I don't know" only.
                keep the answer concise, but provide all of the details you can based on context only in the form of a research report. 
                Only make direct references to material if and only if provided in the context. [/INST] </s> 
                [INST] Question: {question} [/INST]
                [INST] Context: {context} [/INST]
                [INST] Answer: 
                """
        )

    def ask(self, query: str, history=None):
        results = self.vector_store_manager.similarity_search(query, k=5)
        context = _build_context(results)
        prompt = self.ragPrompt.format(history=history or "No previous conversation.", context=context, question=query)
        response_text = self.model.invoke(prompt).content
        if "i don't know" in response_text.lower():
            return self._ask_with_general_knowledge(query, history)
        return response_text

    def ask_using_web_search(self, query, history=None):
        context = self.web_search_tool.invoke("current men chelsea squad")
        print('context', context)
        prompt = self.web_search_prompt.format(history=history or "No previous conversation.", context=context, question=query)
        note = "Note: this answer is based on the web search results. \n"
        return note + self.model.invoke(prompt).content

    def _ask_with_general_knowledge(self, query, history):
        prompt = self.basePrompt.format(history=history or "No previous conversation.", question=query)
        note = "Note: this answer is based on the model's general knowledge. \n"
        return note + self.model.invoke(prompt).content

    def clear(self):
        self.vector_store_manager.clear()