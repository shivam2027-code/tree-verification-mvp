# backend/faiss_adapter.py
import numpy as np
from ai_engine.vectorstore.faiss_store import FaissStore  # assumes implementation exists
from backend.config import VECTOR_DIM

# initialize global faiss store; expected vector dim = VECTOR_DIM
faiss_store = FaissStore(vector_dim=VECTOR_DIM)

def add_vector(vector, meta):
    faiss_store.add(vector, meta)

def search_vector(vector, k=1):
    dist, idx, meta = faiss_store.search(vector, k=k)
    return dist, idx, meta
