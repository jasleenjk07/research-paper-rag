from pathlib import Path

import numpy as np

from rank_bm25 import BM25Okapi

from src.config import FINAL_TOP_K


def tokenize(text):
    """
    Convert text into simple lowercase whitespace tokens.
    """

    return text.lower().split()


def build_bm25_index(chunks):
    """
    Build a BM25 index over all chunks.
    """

    corpus = []

    for document in chunks:

        tokens = tokenize(
            document.page_content
        )

        corpus.append(
            tokens
        )

    bm25 = BM25Okapi(
        corpus
    )

    return bm25


def _matches_document(document, document_name):
    """
    Check whether a chunk belongs to the selected document.
    """

    if not document_name:
        return True

    selected_name = Path(
        document_name
    ).stem.strip().lower()

    paper_name = document.metadata.get(
        "paper_name",
        ""
    )

    paper_name = Path(
        str(paper_name)
    ).stem.strip().lower()

    return paper_name == selected_name


def retrieve_bm25(
    question,
    bm25,
    chunks,
    top_k=FINAL_TOP_K,
    document_name=None,
):
    """
    Retrieve chunks using BM25.

    If document_name is provided, only chunks belonging
    to that document are returned.

    All BM25 scores are calculated first. This is important
    because filtering only the global top-k could cause the
    selected document to disappear from the results.
    """

    query_tokens = tokenize(
        question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    # --------------------------------------------------------
    # No document selected:
    # use the normal global BM25 ranking.
    # --------------------------------------------------------

    if not document_name:

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append(
                (
                    chunks[idx],
                    float(scores[idx]),
                )
            )

        return results

    # --------------------------------------------------------
    # Specific document selected:
    # rank only chunks belonging to that document.
    # --------------------------------------------------------

    matching_indices = []

    for idx, document in enumerate(
        chunks
    ):

        if _matches_document(
            document,
            document_name,
        ):

            matching_indices.append(
                idx
            )

    if not matching_indices:
        return []

    matching_indices = np.array(
        matching_indices,
        dtype=np.int64
    )

    matching_scores = scores[
        matching_indices
    ]

    order = np.argsort(
        matching_scores
    )[::-1]

    selected_indices = matching_indices[
        order[:top_k]
    ]

    results = []

    for idx in selected_indices:

        results.append(
            (
                chunks[idx],
                float(scores[idx]),
            )
        )

    return results