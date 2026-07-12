from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_documents(folder_path):
    documents = []
    folder = Path(folder_path)

    for pdf_path in folder.glob('*.pdf'):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["paper_name"] = pdf_path.stem

        documents.extend(pages)

    return documents