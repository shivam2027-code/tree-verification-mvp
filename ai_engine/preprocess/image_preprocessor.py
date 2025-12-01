# from PIL import Image
# import numpy as np

# class ImagePreprocessor:
#     def __init__(self, size=(224, 224)):
#         self.size = size

#     def load_and_preprocess(self, path):
#         img = Image.open(path).convert("RGB")
#         img = img.resize(self.size)
#         arr = np.array(img) / 255.0  # normalize
#         return arr


def load_and_preprocess_bytes(self, file_bytes: bytes):
    import cv2
    import numpy as np

    # convert bytes → numpy array → image (BGR)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image from bytes")

    # your existing preprocessing pipeline, e.g.:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype("float32") / 255.0

    return img
