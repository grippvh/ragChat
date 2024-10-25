from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from config import Config
from vector_store import VectorStoreManager
from ingestion import Ingestor
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from transformers import pipeline

def _build_context(results):
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])


class RagChat:
    def __init__(self, db_path = Config.DB_PATH):
        self.model = ChatOllama(model=Config.MODEL_NAME)
        self.vector_store_manager = VectorStoreManager(db_path, Config.JINA_API_KEY)
        self.ingestor = Ingestor(self.vector_store_manager)
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        self.web_search_tool = DuckDuckGoSearchRun(api_wrapper=self.wrapper)
        self.domain = None
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.ragPrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks. Use the following pieces 
                of retrieved context to answer the question. If you don't know the answer, strictly answer 
                with "I don't know" only. Provide a concise answer in four sentences maximum.[/INST] </s> 
                [INST] Question: {question} [/INST]
                [INST] Context: {context} [/INST]
                [INST] Conversation history: {history} [/INST]
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
                [INST] Conversation history: {history} [/INST]
                [INST] Answer: 
                """
        )

    def ask(self, query: str, history=None):
        results = self.vector_store_manager.similarity_search(query, k=5)
        context = _build_context(results)
        is_history_relevant = self.is_history_relevant(query, context, history)
        history = history if is_history_relevant else "No previous conversation."
        prompt = self.ragPrompt.format(history=history, context=context, question=query)
        try:
            if self._is_question_related_to_domain(query) is False:
                return "I am afraid you query is not related to the domain you specified. Please change either the domain or the question."
            response_text = self.model.invoke(prompt).content
            if "i don't know" in response_text.lower():
                return self.ask_using_web_search(query, history)
            return response_text
        except Exception as e:
            return "There was an issue connecting to the model service. Please make sure ollama is running and try again later."


    def ask_using_web_search(self, query, history=None):
        context = self.web_search_tool.invoke(query)
        print('context', context)
        prompt = self.web_search_prompt.format(history=history, context=context, question=query)
        note = "Note: this answer is based on the web search results. \n"
        return note + self.model.invoke(prompt).content

    def _ask_with_general_knowledge(self, query, history):
        prompt = self.basePrompt.format(history=history or "No previous conversation.", question=query)
        note = "Note: this answer is based on the model's general knowledge. \n"
        return note + self.model.invoke(prompt).content

    def set_domain(self, domain):
        if self.domain == domain:
            return
        self.domain = domain

    def _is_question_related_to_domain(self, query):
        if not self.domain:
            return True  # in case user considers it's not needed

        sequence_to_classify = query
        candidate_labels = [self.domain]
        similarity = self.classifier(sequence_to_classify, candidate_labels)['scores'][0]

        return similarity > 0.1 # classifier is kinda strict so taking lower threshold

    def is_history_relevant(self, query: str, context: str, history: str) -> bool:
        if not history:
            return False  # No history, so assume it's relevant by default
        history_relevance_prompt = PromptTemplate.from_template(
            """<s> [INST] Determine if the chat history is useful for answering the following question based on the context provided. Answer only "yes" or "no".
            [/INST] Question: {question}
            [INST] Context: {context} [/INST]
            [INST] Chat History: {history} [/INST]
            [INST] Relevance: """
        )
        relevance_prompt = history_relevance_prompt.format(
            question=query,
            context=context,
            history=history or "No previous conversation."
        )
        relevance_response = self.model.invoke(relevance_prompt).content.strip().lower()
        print(relevance_response)
        return "yes" in relevance_response

    def clear(self):
        self.vector_store_manager.clear()