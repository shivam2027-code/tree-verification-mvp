# import faiss
# import numpy as np

# class FaissStore:
#     def __init__(self, vector_dim):
#         self.vector_dim = vector_dim
#         self.index = faiss.IndexFlatL2(vector_dim)  # L2 distance
#         self.metadata = []  # store tree_id + location + images

#     def add(self, vector, meta):
#         vector = np.array(vector).astype('float32')
#         self.index.add(np.expand_dims(vector, axis=0))
#         self.metadata.append(meta)

#     def search(self, vector, k=1):
#         vector = np.array(vector).astype('float32')
#         distances, indices = self.index.search(np.expand_dims(vector, axis=0), k)
#         return distances[0][0], indices[0][0], self.metadata[indices[0][0]]



import faiss
import numpy as np

class FaissStore:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)
        self.metadata = []  # parallel list to match Faiss IDs

    def add(self, vector: np.ndarray, meta: dict):
        vector = vector.astype('float32')
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, vector: np.ndarray, k=1):
        vector = vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        idx = indices[0][0]

        if idx < 0 or idx >= len(self.metadata):
            return None, None, None

        return float(distances[0][0]), int(idx), self.metadata[idx]
