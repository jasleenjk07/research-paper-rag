import json
from collections import defaultdict

from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store

from src.bm25_retriever import build_bm25_index
from src.hybrid_retriever import retrieve_hybrid

from src.config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    HYBRID_RETRIEVE_K,
    FINAL_TOP_K,
)


def load_evaluation_questions(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def find_relevant_rank(
    retrieved_documents,
    expected_paper,
):
    for rank, (document, score) in enumerate(
        retrieved_documents,
        start=1,
    ):
        paper_name = document.metadata.get("paper_name")

        if paper_name == expected_paper:
            return rank

    return None


def evaluate_hybrid(
    evaluation_questions,
    index,
    bm25,
    chunks,
    embedding_model,
    top_k,
):
    hits = 0
    reciprocal_rank_sum = 0

    category_stats = defaultdict(
        lambda: {
            "questions": 0,
            "hits": 0,
            "rr_sum": 0,
        }
    )

    for item in evaluation_questions:
        question = item["question"]
        expected_paper = item["expected_paper"]
        category = item["category"]

        category_stats[category]["questions"] += 1

        retrieved_documents = retrieve_hybrid(
            question,
            index,
            bm25,
            chunks,
            embedding_model,
            retrieve_k=HYBRID_RETRIEVE_K,
            top_k=top_k,
        )

        relevant_rank = find_relevant_rank(
            retrieved_documents,
            expected_paper,
        )

        if relevant_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / relevant_rank

            category_stats[category]["hits"] += 1
            category_stats[category]["rr_sum"] += (
                1 / relevant_rank
            )

        print(f"\nQuestion: {question}")
        print(f"Category: {category}")
        print(f"Expected Paper: {expected_paper}")

        print("\nRetrieved Papers:")

        for rank, (document, _) in enumerate(
            retrieved_documents,
            start=1,
        ):
            print(
                f"{rank}. {document.metadata['paper_name']}"
            )

        if relevant_rank is None:
            print("Relevant Paper: NOT RETRIEVED")
        else:
            print(f"Relevant Paper Rank: {relevant_rank}")

    total_questions = len(evaluation_questions)

    recall_at_k = hits / total_questions
    mrr = reciprocal_rank_sum / total_questions

    return (
        recall_at_k,
        mrr,
        category_stats,
    )


if __name__ == "__main__":
    print("Loading evaluation questions...")

    evaluation_questions = load_evaluation_questions(
        "evaluation/questions.json"
    )

    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Loading vector store...")

    index, chunks = load_vector_store(
        FAISS_INDEX_PATH,
        CHUNKS_PATH,
    )

    print("Building BM25 index...")

    bm25 = build_bm25_index(chunks)

    top_k = FINAL_TOP_K

    (
        recall_at_k,
        mrr,
        category_stats,
    ) = evaluate_hybrid(
        evaluation_questions,
        index,
        bm25,
        chunks,
        embedding_model,
        top_k,
    )

    print("\n========== Overall Results ==========")
    print(f"Questions Evaluated : {len(evaluation_questions)}")
    print(f"Recall@{top_k}       : {recall_at_k:.4f}")
    print(f"MRR                 : {mrr:.4f}")

    print("\n======= Category-wise Results =======")

    for category, stats in category_stats.items():
        category_recall = (
            stats["hits"] / stats["questions"]
        )

        category_mrr = (
            stats["rr_sum"] / stats["questions"]
        )

        print(f"\n{category.title()}")
        print("-" * len(category))
        print(f"Questions : {stats['questions']}")
        print(f"Recall@{top_k}: {category_recall:.4f}")
        print(f"MRR: {category_mrr:.4f}")