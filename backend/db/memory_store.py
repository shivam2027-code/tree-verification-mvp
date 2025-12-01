# backend/db/memory_store.py
import json
import os
from threading import Lock
from typing import Dict, Any, Optional
from backend.config import MEMORY_STORE_FILE

_lock = Lock()

class MemoryStore:
    def __init__(self):
        self.trees = []  # list of dicts with keys: tree_id, owner_id, lat, lng, leaf_vec, bark_vec, tree_vec, combined_vec, images, created_at
        self._load()

    def _load(self):
        if os.path.exists(MEMORY_STORE_FILE):
            try:
                with open(MEMORY_STORE_FILE, "r") as f:
                    self.trees = json.load(f)
            except Exception:
                self.trees = []

    def _save(self):
        os.makedirs(os.path.dirname(MEMORY_STORE_FILE) or ".", exist_ok=True)
        with _lock:
            with open(MEMORY_STORE_FILE, "w") as f:
                json.dump(self.trees, f)

    def add_tree(self, tree_record: Dict[str, Any]):
        with _lock:
            self.trees.append(tree_record)
            self._save()

    def get_all(self):
        return self.trees

    def get_tree_by_id(self, tree_id: str) -> Optional[Dict[str, Any]]:
        for t in self.trees:
            if t.get("tree_id") == tree_id:
                return t
        return None

    def update_tree(self, tree_id: str, patch: Dict[str, Any]):
        for i, t in enumerate(self.trees):
            if t.get("tree_id") == tree_id:
                t.update(patch)
                self._save()
                return t
        return None

# single instance
memory_store = MemoryStore()
