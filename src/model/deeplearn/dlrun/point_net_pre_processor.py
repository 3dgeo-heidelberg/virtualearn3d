import numpy as np


class PointNetPreProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into the PointNet neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, num_points):
        self.num_points = num_points

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        # TODO Rethink : Use support points to build the input ?
        X, y = inputs['X'], inputs['y']
        return np.expand_dims(X[:self.num_points], 0), \
            np.expand_dims(y[:self.num_points], 0)
