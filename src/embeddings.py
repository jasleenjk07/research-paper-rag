from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME

def get_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model