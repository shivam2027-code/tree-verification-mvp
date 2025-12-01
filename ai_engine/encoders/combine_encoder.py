import numpy as np

class CombinedEncoder:
    @staticmethod
    def combine(leaf_vec, bark_vec, tree_vec):
        return np.concatenate([leaf_vec, bark_vec, tree_vec])
