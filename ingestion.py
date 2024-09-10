import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import Config


class Ingestor:
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

    def ingest_file(self, pdf_file_path):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        self.vector_store_manager.add_documents(chunks)

    def ingest_url(self, url):
        response = requests.get(url, headers=Config.REQUEST_HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.get_text(separator="\n")
        chunks = self.text_splitter.create_documents([texts])
        self.vector_store_manager.add_documents(chunks)
