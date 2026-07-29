from src.embeddings import get_embedding_model
from src.hybrid_retriever import retrieve_hybrid
from src.bm25_retriever import build_bm25_index
from src.vector_store import load_vector_store
from src.llm import get_llm
from src.generator import generate_answer
from src.config import (
    FINAL_TOP_K,
    HYBRID_RETRIEVE_K,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
)
from src.reranker import (
    load_reranker,
    rerank_documents,
)
from pathlib import Path

class RAGPipeline:
    def __init__(self):
        if(not Path(FAISS_INDEX_PATH).exists() or not Path(CHUNKS_PATH).exists()):
            raise FileNotFoundError("Vector store not found. Run 'python -m src.ingest' first.")

        print("Loading embedding model...")
        self.embedding_model = get_embedding_model()

        print("Loading vector store...")
        self.index, self.chunks = load_vector_store(FAISS_INDEX_PATH, CHUNKS_PATH)

        print("Building BM25 index...")
        self.bm25 = build_bm25_index(self.chunks)

        print("Loading local LLM...")
        self.llm = get_llm()

        print("Loading reranker...")
        self.reranker = load_reranker()

        print("RAG pipeline ready!")

    def answer(self, question):
        retrieved_documents = retrieve_hybrid(
            question,
            self.index,
            self.bm25,
            self.chunks,
            self.embedding_model,
            retrieve_k=HYBRID_RETRIEVE_K,
            top_k=HYBRID_RETRIEVE_K,
        )

        retrieved_documents = rerank_documents(
            question,
            retrieved_documents,
            self.reranker,
            top_k=FINAL_TOP_K,
        )
        self.last_retrieved_documents = retrieved_documents
        answer = generate_answer(
            question,
            retrieved_documents,
            self.llm
        )
        return answer