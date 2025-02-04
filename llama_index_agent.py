import numexpr
from langchain_community.tools import DuckDuckGoSearchResults
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from transformers import pipeline

from config import Config
from ingestion import Ingestor
from prompt import llama_index_prompt
from vector_store import VectorStoreManager

def _build_context(results):
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])

class RagChat:
    def __init__(self, db_path=Config.DB_PATH):
        web_search_tool = FunctionTool.from_defaults(fn=self.web_search)
        rag_search_tool = FunctionTool.from_defaults(fn=self.rag_search)
        math_solver_tool = FunctionTool.from_defaults(fn=self.solve_math)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self.agent = ReActAgent.from_tools([rag_search_tool, web_search_tool, math_solver_tool], llm=Ollama(model=Config.MODEL_NAME), verbose=True,
                                      max_iterations=20,timeout=None,chat_history=None,memory=memory)

        self.agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(llama_index_prompt)})
        self.agent.reset()
        self.vector_store_manager = VectorStoreManager(db_path)
        self.ingestor = Ingestor(self.vector_store_manager)
        self.duck_duck_go_search = DuckDuckGoSearchResults()
        self.domain = None
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # todo: use this tool only after rag
    def web_search(self, request: str):
        """Runs web search and gets information about the query.
         Use data fetched from this tool to answer the question.

         PRIORITY: MEDIUM, SHOULD BE CALLED ONLY AFTER rag_search_tool
         """
        print("\n----RUNNING WEB SEARCH for " + request + "----")
        res = self.duck_duck_go_search.invoke(request)
        print(res)
        return res

    # todo: give priority
    def rag_search(self, query: str):
        """Runs local database search and gets the context for the query to help to answer the question.
        Use data fetched from this tool to answer the question.

        PRIORITY: HIGHEST, SHOULD BE PRIORITIZED OVER web_search_tool
        """
        print("\n----RUNNING RAG SEARCH for " + query + "----")
        results = self.vector_store_manager.similarity_search(query, k=3)
        context = _build_context(results)
        return context


    def solve_math(self, expression: str) -> str:
        """
        Safely evaluates a mathematical expression using numexpr.
        Returns the result as a string.
        """
        try:
            result = numexpr.evaluate(expression)
            return str(result)
        except Exception as e:
            # You can log or handle errors here
            return f"Error evaluating expression: {str(e)}"

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

    def ask(self, query: str):
        #todo: check number transformer (5000=5.000)
        # print("\n MEMORY---------\n")
        # print(self.agent.memory.to_string())
        try:
            if self._is_question_related_to_domain(query) is False:
                return "I am afraid you query is not related to the domain you specified. Please change either the domain or the question."
            return self.agent.chat(query).response
        except Exception as e:
            return "There was an issue connecting to the model service. Please make sure ollama is running and try again later." + str(e)


    def clear(self):
        self.vector_store_manager.clear()
