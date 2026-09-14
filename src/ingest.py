import json
from pathlib import Path

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    build_vector_store,
    add_to_vector_store,
    save_vector_store,
    load_vector_store,
)

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    PROCESSED_PAPERS_PATH,
)


def load_processed_papers():
    """
    Load the names of documents that have already been processed.
    """

    path = Path(PROCESSED_PAPERS_PATH)

    if not path.exists():
        return set()

    with open(path, "r") as file:
        processed_papers = json.load(file)

    return set(processed_papers)


def save_processed_papers(processed_papers):
    """
    Save the names of successfully processed documents.
    """

    Path(PROCESSED_PAPERS_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(PROCESSED_PAPERS_PATH, "w") as file:
        json.dump(
            sorted(processed_papers),
            file,
            indent=4,
        )


def ingest_pdf_files(pdf_files):
    """
    Ingest PDF files into the shared knowledge base.

    This function is used for both:
    - Built-in research papers
    - User-uploaded PDFs

    Returns:
        list[str]: Names of successfully processed PDF files.
    """

    pdf_files = [
        Path(pdf_path)
        for pdf_path in pdf_files
    ]

    if not pdf_files:
        print("No PDF files provided.")
        return []

    processed_papers = load_processed_papers()

    new_papers = [
        pdf_path
        for pdf_path in pdf_files
        if pdf_path.name not in processed_papers
    ]

    if not new_papers:
        print("No new papers found.")
        return []

    print(
        f"New papers found: {len(new_papers)}"
    )

    # Load embedding model once
    embedding_model = get_embedding_model()

    # Check whether an existing vector store is available
    vector_store_exists = (
        Path(FAISS_INDEX_PATH).exists()
        and Path(CHUNKS_PATH).exists()
    )

    if vector_store_exists:
        print("Loading existing vector store...")

        index, chunks = load_vector_store(
            FAISS_INDEX_PATH,
            CHUNKS_PATH,
        )

    else:
        print("Creating a new vector store...")

        index = None
        chunks = []

    successfully_processed = []

    for pdf_path in new_papers:

        print(
            f"\nProcessing: {pdf_path.name}"
        )

        try:
            # -----------------------------
            # 1. Load PDF
            # -----------------------------

            documents = load_pdf(pdf_path)

            print(
                f"Pages loaded: {len(documents)}"
            )

            if not documents:
                print(
                    f"No pages found in {pdf_path.name}."
                )
                continue

            # -----------------------------
            # 2. Create chunks
            # -----------------------------

            new_chunks = chunk_documents(
                documents,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            print(
                f"Chunks created: {len(new_chunks)}"
            )

            if not new_chunks:
                print(
                    f"No chunks created for "
                    f"{pdf_path.name}."
                )
                continue

            # -----------------------------
            # 3. Add embeddings to FAISS
            # -----------------------------

            if index is None:

                index, chunks = build_vector_store(
                    new_chunks,
                    embedding_model,
                )

            else:

                index, chunks = add_to_vector_store(
                    index,
                    chunks,
                    new_chunks,
                    embedding_model,
                )

            # -----------------------------
            # 4. Mark document as processed
            # -----------------------------

            processed_papers.add(
                pdf_path.name
            )

            successfully_processed.append(
                pdf_path.name
            )

            print(
                f"Successfully processed: "
                f"{pdf_path.name}"
            )

        except Exception as e:

            print(
                f"Failed to process "
                f"{pdf_path.name}: {e}"
            )

    # -----------------------------
    # Save only if something succeeded
    # -----------------------------

    if not successfully_processed:

        print(
            "\nNo documents were successfully "
            "processed."
        )

        return []

    print(
        "\nSaving vector store..."
    )

    save_vector_store(
        index,
        chunks,
        FAISS_INDEX_PATH,
        CHUNKS_PATH,
    )

    save_processed_papers(
        processed_papers
    )

    print(
        "\nIngestion completed successfully."
    )

    print(
        f"Documents processed: "
        f"{len(successfully_processed)}"
    )

    return successfully_processed


def ingest_documents(papers_folder):
    """
    Ingest all PDFs from the built-in papers folder.
    """

    papers_folder = Path(papers_folder)

    if not papers_folder.exists():

        print(
            f"Folder does not exist: "
            f"{papers_folder}"
        )

        return []

    pdf_files = list(
        papers_folder.glob("*.pdf")
    )

    return ingest_pdf_files(
        pdf_files
    )


def ingest_uploaded_files(pdf_paths):
    """
    Ingest user-uploaded PDF files.

    Args:
        pdf_paths: Paths of uploaded PDF files.

    Returns:
        list[str]: Names of successfully
        processed files.
    """

    pdf_files = [
        Path(pdf_path)
        for pdf_path in pdf_paths
    ]

    return ingest_pdf_files(
        pdf_files
    )


if __name__ == "__main__":

    ingest_documents(
        "data/papers"
    )