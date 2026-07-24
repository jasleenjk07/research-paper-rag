from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import retrieve_documents
from src.config import (FAISS_INDEX_PATH, CHUNKS_PATH, TOP_K)

def print_retrieved_chunks(query):
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    embedding_model = get_embedding_model()

    index, chunks = load_vector_store(FAISS_INDEX_PATH, CHUNKS_PATH)

    retrieved_documents = retrieve_documents(query, index, chunks, embedding_model, TOP_K)

    for rank, (document, score) in enumerate(retrieved_documents, start=1,):
        metadata = document.metadata

        print(f"\nRank: {rank}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Paper: {metadata.get('paper_name')}")
        print(f"Page: {metadata.get('page', 0) + 1}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")

        preview = document.page_content[:400].replace("\n", " ")

        print("\nPreview:")
        print(preview)
        print("-" * 80)

if __name__ == "__main__":
    query = input("Enter a question: ")
    print_retrieved_chunks(query)