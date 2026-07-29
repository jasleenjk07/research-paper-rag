from rank_bm25 import BM25Okapi
import numpy as np

from src.config import FINAL_TOP_K

def tokenize(text):
    return text.lower().split()

def build_bm25_index(chunks):
    corpus = []
    for document in chunks:
        tokens = tokenize(document.page_content)
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)
    return bm25

def retrieve_bm25(question, bm25, chunks, top_k=FINAL_TOP_K):
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((chunks[idx], scores[idx]))
    return results