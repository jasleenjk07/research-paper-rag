import pickle
from pathlib import Path

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

def save_vector_store(index, chunks, index_path, chunks_path):
    Path(index_path).parent.mkdir(
        parents = True,
        exist_ok = True
    )
    faiss.write_index(
        index, 
        index_path
    )
    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

def load_vector_store(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)
    return index, chunks

def add_to_vector_store(index, chunks, new_chunks, model):
    texts = [chunk.page_content for chunk in new_chunks]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings, dtype = "float32")
    faiss.normalize_L2(embeddings)
    start_chunk_id = len(chunks)
    for index_offset, chunk in enumerate(new_chunks):
        chunk.metadata["chunk_id"] = (start_chunk_id + index_offset)
    index.add(embeddings)
    chunks.extend(new_chunks)
    return index, chunks