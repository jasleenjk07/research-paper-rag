import faiss
import numpy as np

from pathlib import Path


def _matches_document(document, document_name):
    """
    Check whether a chunk belongs to the selected document.

    document_name is normally the uploaded filename, while
    document metadata stores the filename stem in paper_name.
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


def retrieve_documents(
    question,
    index,
    chunks,
    model,
    top_k,
    document_name=None,
):
    """
    Retrieve documents using dense vector similarity.

    If document_name is provided, retrieval is restricted
    to chunks belonging to that document.

    When a document filter is used, all indexed chunks are
    searched first so that relevant chunks from the selected
    document are not missed just because unrelated documents
    occupy the global top-k positions.
    """

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(
        question_embedding
    )

    # --------------------------------------------------------
    # Search strategy
    # --------------------------------------------------------
    #
    # Without a document filter:
    #     normal top-k search is sufficient.
    #
    # With a document filter:
    #     search the complete index, then keep only chunks
    #     belonging to the selected document.
    #
    # This is important because a selected document might
    # otherwise be ranked below unrelated documents.
    # --------------------------------------------------------

    if document_name:

        search_k = index.ntotal

    else:

        search_k = min(
            top_k,
            index.ntotal
        )

    scores, indices = index.search(
        question_embedding,
        search_k
    )

    retrieved_documents = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        if index_id < 0:
            continue

        document = chunks[index_id]

        if not _matches_document(
            document,
            document_name,
        ):
            continue

        retrieved_documents.append(
            (
                document,
                float(score),
            )
        )

        if len(retrieved_documents) >= top_k:
            break

    return retrieved_documents


if __name__ == "__main__":

    from src.loader import load_documents
    from src.chunker import chunk_documents
    from src.vector_store import build_vector_store
    from src.embeddings import get_embedding_model
    from src.config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        FINAL_TOP_K,
    )

    documents = load_documents(
        "data/papers"
    )

    chunks = chunk_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    model = get_embedding_model()

    index, chunks = build_vector_store(
        chunks,
        model
    )

    question = (
        "Why can Transformers be trained in parallel?"
    )

    results = retrieve_documents(
        question,
        index,
        chunks,
        model,
        FINAL_TOP_K
    )

    print("\nQuestion:")
    print(question)

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Rank {rank} ---"
        )

        print(
            "Score:",
            score
        )

        print(
            "Paper:",
            document.metadata.get("title")
            or document.metadata.get(
                "paper_name"
            )
        )

        print(
            "Page:",
            document.metadata.get(
                "page",
                0
            ) + 1
        )

        print("\nContent:")
        print(
            document.page_content
        )