import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ai_engine.encoders.leaf_encoder import LeafEncoder
from ai_engine.encoders.bark_encoder import BarkEncoder
from ai_engine.encoders.tree_encoder import TreeEncoder
from ai_engine.encoders.combine_encoder import CombinedEncoder
from PIL import Image

leaf = Image.open("sample_images/leafs/leaf1.jpeg")
bark = Image.open("sample_images/bark/bark1.jpg")
tree = Image.open("sample_images/tree/mangotree.jpeg")

leaf_enc = LeafEncoder().encode(leaf)
bark_enc = BarkEncoder().encode(bark)
tree_enc = TreeEncoder().encode(tree)

final_vec = CombinedEncoder.combine(leaf_enc, bark_enc, tree_enc)

print("Leaf:", leaf_enc.shape)
print("Bark:", bark_enc.shape)
print("Tree:", tree_enc.shape)
print("Final:", final_vec.shape)
