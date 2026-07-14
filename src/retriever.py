import faiss
import numpy as np

def retrieve_documents(question, index, chunks, model, top_k):
    question_embedding = model.encode([question])
    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )
    faiss.normalize_L2(question_embedding)
    scores, indices = index.search(
        question_embedding,
        top_k
    )
    retrieved_documents = []
    for score, index_id in zip(scores[0], indices[0]):
        document = chunks[index_id]
        retrieved_documents.append(
            (document, float(score))
        )
    return retrieved_documents

if __name__ == "__main__":
    from src.loader import load_documents
    from src.chunker import chunk_documents
    from src.vector_store import build_vector_store
    from src.embeddings import get_embedding_model
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

    documents = load_documents("data/papers")

    chunks = chunk_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    model = get_embedding_model()
    index, chunks = build_vector_store(chunks, model)

    question = "Why can Transformers be trained in parallel?"

    results = retrieve_documents(
        question,
        index,
        chunks,
        model,
        TOP_K
    )

    print("\nQuestion:")
    print(question)

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):
        print(f"\n--- Rank {rank} ---")

        print("Score:", score)

        print(
            "Paper:",
            document.metadata.get("title")
            or document.metadata.get("paper_name")
        )

        print(
            "Page:",
            document.metadata.get("page", 0) + 1
        )

        print("\nContent:")
        print(document.page_content)