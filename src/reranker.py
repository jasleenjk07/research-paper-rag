from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL

def load_reranker(model_name=RERANKER_MODEL):
    """
    Load the Cross-Encoder reranker model.
    Returns:
        CrossEncoder
    """
    return CrossEncoder(model_name)

def rerank_documents(
    question,
    retrieved_documents,
    reranker,
    top_k=5,
):
    """
    Rerank retrieved documents using a Cross-Encoder.
    Args:
        question: User query.
        retrieved_documents: List of (Document, retrieval_score).
        reranker: Loaded CrossEncoder model.
        top_k: Number of documents to return.
    Returns:
        List of (Document, reranker_score)
    """

    if not retrieved_documents:
        return []

    sentence_pairs = [
        (question, document.page_content)
        for document, _ in retrieved_documents
    ]

    scores = reranker.predict(sentence_pairs)

    reranked_results = []

    for (document, _), score in zip(retrieved_documents, scores):
        reranked_results.append((document, float(score)))

    reranked_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return reranked_results[:top_k]