import numexpr
import re
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
        try:
            if self._is_question_related_to_domain(query) is False:
                return "I am afraid you query is not related to the domain you specified. Please change either the domain or the question."
            return self.agent.chat(normalize_numbers(query)).response
        except Exception as e:
            return "There was an issue connecting to the model service. Please make sure ollama is running and try again later." + str(e)


    def clear(self):
        self.vector_store_manager.clear()


def normalize_numbers(query: str) -> str:
    """
    Normalizes numeric formats in the query.

    1. Converts commas used as decimal separators (e.g., "3,14") into periods ("3.14").
    2. Removes thousand separators (e.g., converts "5.000" to "5000") by finding numbers
       that match the pattern for thousand-separated digits.
    """
    # replace commas between digits with a period (decimal separator normalization)
    query = re.sub(r'(?<=\d),(?=\d)', '.', query)

    # remove thousand separators: look for numbers like "1.234" or "12.345.678"
    def remove_thousands(match):
        # Remove all dots (thousand separators) from the matched string.
        return match.group(0).replace('.', '')

    # pattern matches numbers with one to three digits, then at least one group of a dot
    # followed by exactly three digits, and word boundaries to avoid matching parts of a larger string.
    query = re.sub(r'\b\d{1,3}(?:\.\d{3})+\b', remove_thousands, query)

    return query