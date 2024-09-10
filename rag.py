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
                with "I don't know" only. 
                Provide a concise answer in four sentences maximum.[/INST] </s> 
                [INST] Question: {question}
                Context: {context}
                Answer: 
                """
        )
        self.basePrompt = PromptTemplate.from_template(
            """
                <s> [INST] You are a highly knowledgeable assistant for question-answering tasks.
                Provide a concise answer in four sentences maximum. If you cannot get the answer from context, find it in your own data[/INST] </s> 
                [INST] Question: {question}
                Answer: 
                """
        )

    def ask(self, query: str):
        results = self.vector_store_manager.similarity_search(query, k=5)
        context = _build_context(results)
        prompt = self.ragPrompt.format(context=context, question=query)
        response_text = self.model.invoke(prompt).content
        if "i don't know" in response_text.lower():
            return self._ask_with_general_knowledge(query)
        return response_text

    def _ask_with_general_knowledge(self, query):
        prompt = self.basePrompt.format(question=query)
        note = "Note: this answer is based on the model's general knowledge."
        return note + self.model.invoke(prompt).content

    def clear(self):
        self.vector_store_manager.clear()



# -------------------------------------------------- CURRENTLY UNUSED --------------------------------------------------

# def rerank(self, documents):
#     start_time = time.time()
#     # query_embedding = self.embedding_model.embed_query(self.last_query)
#     docs = [doc.page_content for doc, _score in documents]
#     print(documents)
#     print('\n\n\nCohere pre reranked Documents:')
#     for i, (doc, score) in enumerate(documents, 1):
#         print(f"{i}. Relevance score: {score:.4f}. Document: {doc.page_content}...")
#
#     co = cohere.Client(COHERE_API_KEY)
#     cohere_response = co.rerank(query=self.last_query, documents=docs, top_n=25, model="rerank-english-v2.0")
#
#     cohere_reranked_docs = [
#         (documents[result.index], result.relevance_score)
#         for result in cohere_response.results
#         if result.relevance_score >= 0.5  # filter out irrelevant stuff
#     ]
#
#     print('\n\n\nCohere Reranked Documents:')
#     for i, ((doc, score1), score2) in enumerate(cohere_reranked_docs, 1):
#         print(f"{i}. Document: {doc}...")
#         print(f"{i}. First score: {score1:.4f}. Reranked score: {score2:.4f}. Document: {doc.page_content}...")
#
#     end_time = time.time()
#     logging.info(f"Reranking took {end_time - start_time:.2f} seconds")
#
#     # return without metadata
#     return [doc.page_content for (doc, score1), score2 in cohere_reranked_docs]
#
# def set_chain(self, prompt: PromptTemplate):
#     self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
#                   | prompt
#                   | self.model
#                   | StrOutputParser())
#
# def summarize(self, documents):
#     start_time = time.time()
#     summarized_docs = []
#     for doc in documents:
#         # summary = self.summarizer(doc.page_content, max_length=100, min_length=30, do_sample=False)[0][
#         #     'summary_text']
#         summary = self.summarizer(doc, min_lenght=30, max_length=100, do_sample=False)[0][
#             'summary_text']
#         summarized_docs.append(summary)
#         print('\nSummarized from: ', doc)
#         print('\nSummarized to: ', summary)
#
#     end_time = time.time()
#     logging.info(f"Summarization took {end_time - start_time:.2f} seconds")
#     return summarized_docs
# # Post-Retrieval Functions
#
# def fuse(self, documents):
#     start_time = time.time()
#     fused_content = " ".join(documents)
#     end_time = time.time()
#     print('\n\n\nfused content: ', fused_content)
#     logging.info(f"Fusing documents took {end_time - start_time:.2f} seconds")
#     return fused_content
#
# # Pre-Retrieval Functions?
# def query_rewriting(self, query):
#     start_time = time.time()
#     original_query = query
#     messages = [
#         SystemMessage(content="Rewrite the following query for better retrieval accuracy. "
#                               "Ensure the rewritten query remains as close to the original meaning as possible,"
#                               "Reply only with the rewritten query, nothing else"),
#         HumanMessage(content=query)
#     ]
#     rewritten_query = self.model.invoke(messages).content
#     end_time = time.time()
#     logging.info(
#         f"Query rewriting: '{original_query}' -> '{rewritten_query}', took {end_time - start_time:.2f} seconds")
#     return rewritten_query