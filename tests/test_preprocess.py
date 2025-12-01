import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ai_engine.preprocess.image_preprocessor import ImagePreprocessor

pre = ImagePreprocessor()

img = pre.load_and_preprocess("sample_images/leafs/leaf1.jpeg")

print(img.shape)
print(img[:2])  # print first 2 rows
