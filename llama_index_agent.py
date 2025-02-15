import numexpr
from langchain_community.tools import DuckDuckGoSearchResults
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from transformers import pipeline

from config import Config
from ingestion import Ingestor
from utils import normalize_numbers, _build_context
from prompt import llama_index_prompt
from relevance_checker import is_query_relevant
from vector_store import VectorStoreManager

class RagChat:
    def __init__(self, db_path=Config.DB_PATH):
        web_search_tool = FunctionTool.from_defaults(fn=self.web_search)
        rag_search_tool = FunctionTool.from_defaults(fn=self.rag_search)
        math_solver_tool = FunctionTool.from_defaults(fn=self.solve_math)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        llm = Ollama(model=Config.MODEL_NAME, request_timeout=120.0)
        self.agent = ReActAgent.from_tools([rag_search_tool, web_search_tool, math_solver_tool], llm=llm, verbose=True,
                                      max_iterations=20,timeout=None,chat_history=None,memory=memory)

        self.agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(llama_index_prompt)})
        self.agent.reset()
        self.vector_store_manager = VectorStoreManager(db_path)
        self.ingestor = Ingestor(self.vector_store_manager)
        self.duck_duck_go_search = DuckDuckGoSearchResults()
        self.domain = None
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def web_search(self, request: str):
        """Runs web search and gets information about the query.
         Use data fetched from this tool to answer the question.
         """
        print("\n----RUNNING WEB SEARCH for " + request + "----")
        res = self.duck_duck_go_search.invoke(request)
        print(res)
        return res

    def rag_search(self, query: str):
        """Runs local database search and gets the context for the query to help to answer the question.
        Use data fetched from this tool to answer the question.
        """
        print("\n----RUNNING RAG SEARCH for " + query + "----")
        results = self.vector_store_manager.similarity_search(query, k=3)
        context = _build_context(results)
        return context


    def solve_math(self, expression: str) -> str:
        """
        Safely evaluates a mathematical expression using numexpr.
        Returns the result as a string.
        STRICTLY PROVIDE AN EXPRESSION IN A FORMAT SUITED FOR numexpr.evaluate(expression) from numexpr library
        do not add "math." to expression, e.g: "math.sqrt(2.0485e+006) * 2000" should be "sqrt(2.0485e+006) * 2000"
        to get the power of a number, use "**" and not "^", e.g. "two to the power of three" is "2**3" and not "2^3"
        Always provide and expect full numbers in every part of the expression. This rule applies both to inputs and to the interpretation of tool responses.

        Do not use whitespaces in numbers.

        - Always Use Observed Values:
        When performing follow-up calculations, refer strictly to the numerical value returned by the math_solver_tool’s observation. Do not substitute or modify this number in your chain-of-thought.

        - No Number Fabrication:
        Under no circumstances should you generate or “hallucinate” a new value for a previously computed result. For example, if the math tool returns 2828427.1247461904 for an expression, that exact value must be used in any subsequent calculations.

        - Pass Exact Observations for Further Calculations:
        When the query instructs further operations (e.g., “divide the final answer by 3 and take a square root”), use the math tool’s output from the previous step directly in the new expression.

        - Maintain Full Precision:
        Always treat every number as exact (i.e., full numbers without rounding or abbreviations) unless the query explicitly requests rounding.
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

    def ask(self, query: str):
        try:
            if "yes" not in is_query_relevant(self.domain, query, self.agent.memory.get_all()):
                return "I am afraid your query is not related to the domain you specified. Please change either the domain or the question."
            return self.agent.chat(normalize_numbers(query)).response
        except Exception as e:
            return "There was an issue connecting to the model service. Please make sure ollama is running and try again later."


    def clear(self):
        self.vector_store_manager.clear()
