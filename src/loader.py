from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["paper_name"] = pdf_path.stem
    return pages

def load_documents(folder_path):
    documents = []
    folder = Path(folder_path)
    for pdf_path in folder.glob('*.pdf'):
        pages = load_pdf(pdf_path)
        documents.extend(pages)
    return documents