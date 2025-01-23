import os
import shutil
from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from config import Config


class VectorStoreManager:
    def __init__(self, db_path, jina_api_key):
        self.db_path = db_path
        # todo: change embeddings to BAAI/bge-base-en-v1.5
        # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.embedding_model = JinaEmbeddings(model_name='jina-embeddings-v2-base-en', trust_remote_code=True, jina_api_key=jina_api_key)
        self.vector_store = Chroma(
            collection_name=(db_path + "_chroma"),
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    # a solution that would check unique files was inspired by https://github.com/pixegami/rag-tutorial-v2/blob/main/populate_database.py
    def add_documents(self, documents):
        # Calculate chunk IDs for the documents
        chunks_with_ids = self.calculate_chunk_ids(documents)

        # Load the existing database and check for already existing IDs
        existing_items = self.vector_store.get(include=[])  # Get all existing IDs
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add new chunks (documents) that are not already in the vector store
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅  No new documents to add")

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

    def calculate_chunk_ids(self, chunks):
        """
        Generate unique chunk IDs based on the source, page number, and chunk index.
        """

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add the chunk ID to the chunk's metadata.
            chunk.metadata["id"] = chunk_id

        return chunks


    def get_sources(self):
        all_data = self.vector_store.get(include=["metadatas"])

        # Extract sources from metadata
        sources = set()
        for metadata in all_data["metadatas"]:
            source = metadata.get("source")
            if source:
                sources.add(source)
            else:
                sources.add("unknown source")

        return sources if sources else ["vector base is empty"]
