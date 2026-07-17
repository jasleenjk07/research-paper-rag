from src.embeddings import get_embedding_model
from src.retriever import retrieve_documents
from src.vector_store import load_vector_store
from src.llm import get_llm
from src.generator import generate_answer
from src.config import (TOP_K, FAISS_INDEX_PATH, CHUNKS_PATH)

from pathlib import Path

class RAGPipeline:
    def __init__(self):
        if(not Path(FAISS_INDEX_PATH).exists() or not Path(CHUNKS_PATH).exists()):
            raise FileNotFoundError("Vector store not found. Run 'python -m src.ingest' first.")

        print("Loading embedding model...")
        self.embedding_model = get_embedding_model()

        print("Loading vector store...")
        self.index, self.chunks = load_vector_store(FAISS_INDEX_PATH, CHUNKS_PATH)

        print("Loading local LLM...")
        self.llm = get_llm()

        print("RAG pipeline ready!")

    def answer(self, question):
        retrieved_documents = retrieve_documents(
            question, 
            self.index,
            self.chunks,
            self.embedding_model,
            TOP_K
        )
        answer = generate_answer(
            question, 
            retrieved_documents, 
            self.llm
        )
        return answer