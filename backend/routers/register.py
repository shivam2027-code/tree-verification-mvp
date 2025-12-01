# backend/routers/register.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import uuid, time
import numpy as np

from backend.db.memory_store import memory_store
from backend.faiss_adapter import add_vector, search_vector
from backend.utils.gps_utils import haversine_meters
from backend.config import DUPLICATE_DISTANCE_THRESHOLD

# import your ai pipeline pieces
from ai_engine.preprocess.image_preprocessor import ImagePreprocessor
from ai_engine.segmentation.segmenter import SegmentationEngine
from ai_engine.encoders.leaf_encoder import LeafEncoder
from ai_engine.encoders.bark_encoder import BarkEncoder
from ai_engine.encoders.tree_encoder import TreeEncoder
from ai_engine.encoders.combine_encoder import CombinedEncoder

router = APIRouter()

pre = ImagePreprocessor()
seg = SegmentationEngine()
leaf_enc = LeafEncoder()
bark_enc = BarkEncoder()
tree_enc = TreeEncoder()
combiner = CombinedEncoder()

@router.post("/register-tree")
async def register_tree(
    owner_id: str = Form(...),
    gps_lat: float = Form(...),
    gps_lng: float = Form(...),
    leaf_image: UploadFile = File(...),
    bark_image: UploadFile = File(...),
    tree_image: UploadFile = File(...),
):
    # read bytes
    leaf_bytes = await leaf_image.read()
    bark_bytes = await bark_image.read()
    tree_bytes = await tree_image.read()

    # preprocess -> returns numpy array normalized (0-1) e.g. ImagePreprocessor.load_and_preprocess
    leaf_np = pre.load_and_preprocess_bytes(leaf_bytes)
    bark_np = pre.load_and_preprocess_bytes(bark_bytes)
    tree_np = pre.load_and_preprocess_bytes(tree_bytes)

    # segmentation
    leaf_seg = seg.segment(leaf_np)
    bark_seg = seg.segment(bark_np)
    tree_seg = seg.segment(tree_np)

    # encode (vectors are numpy arrays)
    leaf_vec = leaf_enc.encode(leaf_seg)
    bark_vec = bark_enc.encode(bark_seg)
    tree_vec = tree_enc.encode(tree_seg)

    # combine
    combined_vec = combiner.combine(leaf_vec, bark_vec, tree_vec)

    # check duplicate via faiss (returns distance and meta)
    dist, idx, meta = search_vector(combined_vec, k=1)
    if meta is not None and dist < DUPLICATE_DISTANCE_THRESHOLD:
        return {"status": "DUPLICATE", "existing_tree_id": meta.get("tree_id"), "distance": float(dist)}

    # not duplicate -> register new tree
    new_tree_id = "TID-" + uuid.uuid4().hex[:10]
    record = {
        "tree_id": new_tree_id,
        "owner_id": owner_id,
        "gps_lat": gps_lat,
        "gps_lng": gps_lng,
        "leaf_vec": leaf_vec.tolist(),
        "bark_vec": bark_vec.tolist(),
        "tree_vec": tree_vec.tolist(),
        "combined_vec": combined_vec.tolist(),
        "images": {
            "leaf_filename": leaf_image.filename,
            "bark_filename": bark_image.filename,
            "tree_filename": tree_image.filename
        },
        "created_at": int(time.time())
    }

    # save metadata and vector in FAISS
    memory_store.add_tree(record)
    add_vector(np.array(combined_vec, dtype='float32'), {"tree_id": new_tree_id, "gps_lat": gps_lat, "gps_lng": gps_lng})

    return {"status": "REGISTERED", "tree_id": new_tree_id}
