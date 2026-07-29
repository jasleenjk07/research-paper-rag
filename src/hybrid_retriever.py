from src.retriever import retrieve_documents
from src.bm25_retriever import retrieve_bm25
from src.config import (
    RRF_K,
    HYBRID_RETRIEVE_K,
    FINAL_TOP_K,
)

def rrf_fusion(
    dense_results,
    bm25_results,
    top_k,
):
    """
    Combine Dense Retrieval and BM25 results using
    Reciprocal Rank Fusion (RRF).

    Returns:
        List of (Document, RRF Score)
    """

    fusion_scores = {}

    # Dense Retrieval contribution
    for rank, (document, _) in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = document.metadata["chunk_id"]

        if chunk_id not in fusion_scores:
            fusion_scores[chunk_id] = {
                "document": document,
                "score": 0.0,
            }

        fusion_scores[chunk_id]["score"] += (
            1 / (RRF_K + rank)
        )

    # BM25 contribution
    for rank, (document, _) in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = document.metadata["chunk_id"]

        if chunk_id not in fusion_scores:
            fusion_scores[chunk_id] = {
                "document": document,
                "score": 0.0,
            }

        fusion_scores[chunk_id]["score"] += (
            1 / (RRF_K + rank)
        )

    fused_results = []

    for item in fusion_scores.values():
        fused_results.append(
            (
                item["document"],
                item["score"],
            )
        )

    fused_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return fused_results[:top_k]


def retrieve_hybrid(
    question,
    index,
    bm25,
    chunks,
    embedding_model,
    retrieve_k=HYBRID_RETRIEVE_K,
    top_k=FINAL_TOP_K,
):
    """
    Retrieve documents using both Dense Retrieval
    and BM25, then combine them using
    Reciprocal Rank Fusion (RRF).
    """

    dense_results = retrieve_documents(
        question,
        index,
        chunks,
        embedding_model,
        retrieve_k,
    )

    bm25_results = retrieve_bm25(
        question,
        bm25,
        chunks,
        retrieve_k,
    )

    hybrid_results = rrf_fusion(
        dense_results,
        bm25_results,
        top_k,
    )

    return hybrid_results