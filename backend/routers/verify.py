# backend/routers/verify.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
from backend.faiss_adapter import search_vector
from backend.db.memory_store import memory_store
from backend.utils.gps_utils import haversine_meters
from backend.config import VERIFICATION_DISTANCE_THRESHOLD_LEAF, VERIFICATION_DISTANCE_THRESHOLD_BARK, GPS_DISTANCE_THRESHOLD_M

from ai_engine.preprocess.image_preprocessor import ImagePreprocessor
from ai_engine.segmentation.segmenter import SegmentationEngine
from ai_engine.encoders.leaf_encoder import LeafEncoder
from ai_engine.encoders.bark_encoder import BarkEncoder

router = APIRouter()
pre = ImagePreprocessor()
seg = SegmentationEngine()
leaf_enc = LeafEncoder()
bark_enc = BarkEncoder()

@router.post("/verify-tree")
async def verify_tree(
    user_id: str = Form(...),
    gps_lat: float = Form(...),
    gps_lng: float = Form(...),
    leaf_image: UploadFile = File(None),
    bark_image: UploadFile = File(None),
):
    if (leaf_image is None and bark_image is None):
        raise HTTPException(status_code=400, detail="Please upload leaf_image or bark_image for verification")

    # choose mode
    if leaf_image:
        img_bytes = await leaf_image.read()
        np_img = pre.load_and_preprocess_bytes(img_bytes)
        seg_img = seg.segment(np_img)
        vec = leaf_enc.encode(seg_img)
        threshold = VERIFICATION_DISTANCE_THRESHOLD_LEAF
        vector_field = "leaf_vec"
    else:
        img_bytes = await bark_image.read()
        np_img = pre.load_and_preprocess_bytes(img_bytes)
        seg_img = seg.segment(np_img)
        vec = bark_enc.encode(seg_img)
        threshold = VERIFICATION_DISTANCE_THRESHOLD_BARK
        vector_field = "bark_vec"

    # search (we search against combined vectors or leaf/bark vectors depending on implementation)
    # For speed we can search combined vectors by padding/embedding scheme or maintain separate indexes.
    # Here we search combined vector index (we encoded leaf/bark to same size? adjust accordingly).
    dist, idx, meta = search_vector(vec, k=1)

    if meta is None:
        return {"verified": False, "reason": "no_trees_registered"}

    # GPS check
    db_lat = meta.get("gps_lat")
    db_lng = meta.get("gps_lng")
    gps_distance = haversine_meters(gps_lat, gps_lng, db_lat, db_lng)

    # decide verification
    if dist < threshold and gps_distance <= GPS_DISTANCE_THRESHOLD_M:
        # fetch tree metadata
        tree_id = meta.get("tree_id")
        tree_record = memory_store.get_tree_by_id(tree_id)
        # optionally update health/carbon here
        return {"verified": True, "tree_id": tree_id, "similarity": float(dist), "gps_distance_m": gps_distance}
    else:
        # ambiguous or not verified
        return {"verified": False, "similarity": float(dist), "gps_distance_m": gps_distance}
