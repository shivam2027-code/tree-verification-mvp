# Tree Identity MVP

MVP for tree registration and verification using image embeddings + GPS + segmentation.

## Features
- Leaf / Bark / Tree Embeddings
- Duplicate Tree Checking
- Simple Health Score
- Carbon Estimation
- FastAPI backend
- FAISS vector search

## Run
pip install -r requirements.txt
uvicorn backend.main:app --reload



#for dowloading sam follow this its for cpu 
run these 
pip install opencv-python pillow matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
 
 & place it    ai_engine/weights/sam_vit_b_01ec64.pth
