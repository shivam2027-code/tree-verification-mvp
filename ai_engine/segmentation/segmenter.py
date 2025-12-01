import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SegmentationEngine:
    def __init__(self, checkpoint_path="ai_engine/weights/sam_vit_b_01ec64.pth"):
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam_model.to("cpu")
        self.mask_generator = SamAutomaticMaskGenerator(sam_model)

    def segment(self, image_np):
        """
        image_np = normalized numpy array from preprocessor (224x224x3)
        returns: masked image (background removed)
        """

        # Convert back to uint8 image
        img_uint8 = (image_np * 255).astype(np.uint8)

        # Generate all masks
        masks = self.mask_generator.generate(img_uint8)

        if len(masks) == 0:
            print("⚠ No masks found — returning original image.")
            return img_uint8

        # Pick the largest mask (most of the tree/leaf)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        best_mask = masks[0]['segmentation']

        # Apply mask (remove background)
        output = img_uint8.copy()
        output[~best_mask] = 0

        return output
