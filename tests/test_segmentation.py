import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ai_engine.preprocess.image_preprocessor import ImagePreprocessor
from ai_engine.segmentation.segmenter import SegmentationEngine
import matplotlib.pyplot as plt

pre = ImagePreprocessor()
img = pre.load_and_preprocess("sample_images/leafs/leaf4.jpeg")

segmenter = SegmentationEngine()
seg_img = segmenter.segment(img)


#generally it not work in linux 


# plt.imshow(seg_img)
# plt.title("Segmented Leaf")
# plt.show()

#print("Segmentation test completed.")


#for see image 
import cv2

cv2.imwrite("segmented_output.jpg", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
print("Saved segmented image as segmented_output.jpg")

