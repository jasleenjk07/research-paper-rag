from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)