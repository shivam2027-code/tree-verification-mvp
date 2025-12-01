# backend/config.py

# Vector size used in FAISS (set to your final_vector length)
VECTOR_DIM = 1536

# Duplicate thresholds (L2 distance)
DUPLICATE_DISTANCE_THRESHOLD = 0.65   # registration: final_vector distance
VERIFICATION_DISTANCE_THRESHOLD_LEAF = 0.75  # when verifying with leaf (tighter/looser tune)
VERIFICATION_DISTANCE_THRESHOLD_BARK = 0.72

# GPS distance threshold (meters)
GPS_DISTANCE_THRESHOLD_M = 12.0  # <=12 m considered same tree

# Data persistence path (for in-memory dump)
MEMORY_STORE_FILE = "backend/db/memory_store.json"
