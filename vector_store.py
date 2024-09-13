import os
import shutil

from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from config import Config


class VectorStoreManager:
    def __init__(self, db_path, jina_api_key):
        self.db_path = db_path
        self.embedding_model = JinaEmbeddings(model_name='jina-embeddings-v2-base-en', trust_remote_code=True, jina_api_key=jina_api_key)
        self.vector_store = Chroma(
            collection_name= (db_path + "_chroma"),
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents=documents)

    def similarity_search(self, query, k=5):
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception:
            return []

    def clear(self):
        if self.db_path == Config.DB_PATH:
            # Move files from DATA_PATH to UNUSED_DATA_PATH
            if not os.path.exists(Config.UNUSED_DATA_PATH):
                os.makedirs(Config.UNUSED_DATA_PATH)

            for file_name in os.listdir(Config.DATA_PATH):
                src_file = os.path.join(Config.DATA_PATH, file_name)
                dst_file = os.path.join(Config.UNUSED_DATA_PATH, file_name)
                shutil.move(src_file, dst_file)
        self.vector_store.reset_collection()