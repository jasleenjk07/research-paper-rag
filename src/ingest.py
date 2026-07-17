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
    path = Path(PROCESSED_PAPERS_PATH)

    if not path.exists():
        return set()

    with open(path, "r") as file:
        processed_papers = json.load(file)

    return set(processed_papers)


def save_processed_papers(processed_papers):
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


def ingest_documents(papers_folder):
    papers_folder = Path(papers_folder)

    pdf_files = list(
        papers_folder.glob("*.pdf")
    )

    processed_papers = load_processed_papers()

    new_papers = [
        pdf_path
        for pdf_path in pdf_files
        if pdf_path.name not in processed_papers
    ]

    if not new_papers:
        print("No new papers found.")
        return

    print(f"New papers found: {len(new_papers)}")

    embedding_model = get_embedding_model()

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
        index = None
        chunks = []

    for pdf_path in new_papers:
        print(f"\nProcessing: {pdf_path.name}")

        documents = load_pdf(pdf_path)

        print(f"Pages loaded: {len(documents)}")

        new_chunks = chunk_documents(
            documents,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        print(f"Chunks created: {len(new_chunks)}")

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

        processed_papers.add(pdf_path.name)

    print("\nSaving vector store...")

    save_vector_store(
        index,
        chunks,
        FAISS_INDEX_PATH,
        CHUNKS_PATH,
    )

    save_processed_papers(
        processed_papers
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest_documents("data/papers")