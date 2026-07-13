import faiss
import numpy as np

def build_vector_store(chunks, model):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)
    embeddings = np.array(
        embeddings,
        dtype="float32"
    )
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks