import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from ai_engine.vectorstore.faiss_store import FaissStore

vec1 = np.random.rand(1536)
vec2 = vec1 + np.random.normal(0, 0.01, 1536)  # very similar

store = FaissStore(1536)

store.add(vec1, {"tree_id": 1})
dist, idx, meta = store.search(vec2)

print("Distance:", dist)
print("Match meta:", meta)
