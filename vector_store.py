import os
import shutil
import stat

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma

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
        return self.vector_store.similarity_search_with_score(query, k=k)

    def clear(self):
        shutil.rmtree(self.db_path)
        os.makedirs(self.db_path)
        os.chmod(self.db_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
