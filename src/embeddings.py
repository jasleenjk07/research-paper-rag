from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME

def get_embedding_model():
    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        model = SentenceTransformer(MODEL_NAME)
    return model