# backend/faiss_manager.py

import numpy as np
from ai_engine.vectorstore.faiss_store import FaissStore
from backend.config import VECTOR_DIM

LEAF_DIM = 512
BARK_DIM = 512
TREE_DIM = 512
COMBINED_DIM = VECTOR_DIM  # e.g., 1536

# 3 separate indexes
faiss_leaf = FaissStore(vector_dim=LEAF_DIM)
faiss_bark = FaissStore(vector_dim=BARK_DIM)
faiss_combined = FaissStore(vector_dim=COMBINED_DIM)

def add_leaf_vector(vec, meta):
    faiss_leaf.add(np.array(vec, dtype='float32'), meta)

def add_bark_vector(vec, meta):
    faiss_bark.add(np.array(vec, dtype='float32'), meta)

def add_combined_vector(vec, meta):
    faiss_combined.add(np.array(vec, dtype='float32'), meta)

def search_leaf(vec, k=1):
    return faiss_leaf.search(vec, k)

def search_bark(vec, k=1):
    return faiss_bark.search(vec, k)

def search_combined(vec, k=1):
    return faiss_combined.search(vec, k)
