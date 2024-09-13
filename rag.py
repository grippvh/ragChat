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
        self.ragPrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks. Use the following pieces 
                of retrieved context to answer the question. If you don't know the answer, strictly answer 
                with "I don't know" only. Provide a concise answer in four sentences maximum.[/INST] </s> 
                [INST] Chat History: {history} [/INST]
                [INST] Question: {question} [/INST]
                [INST] Context: {context} [/INST]
                [INST] Answer: 
                """
        )
        self.basePrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks.
                Provide a concise answer in four sentences maximum. If you cannot get the answer from context, find it in your own data[/INST] </s> 
                [INST] Chat History: {history} [/INST]
                [INST] Question: {question} [/INST]
                [INST] Answer: 
                """
        )

    def ask(self, query: str, history=None):
        results = self.vector_store_manager.similarity_search(query, k=5)
        context = _build_context(results)
        prompt = self.ragPrompt.format(history=history or "No previous conversation.", context=context, question=query)
        print(prompt)
        response_text = self.model.invoke(prompt).content
        if "i don't know" in response_text.lower():
            return self._ask_with_general_knowledge(query, history)
        return response_text

    def _ask_with_general_knowledge(self, query, history):
        prompt = self.basePrompt.format(history=history or "No previous conversation.", question=query)
        note = "Note: this answer is based on the model's general knowledge."
        return note + self.model.invoke(prompt).content

    def clear(self):
        self.vector_store_manager.clear()