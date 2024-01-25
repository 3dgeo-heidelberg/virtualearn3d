# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.features_downsampling_layer import \
    FeaturesDownsamplingLayer
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesDownsamplingLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Features downsampling layer test that checks the operations of a
    features downsampling layer yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Features downsampling layer test')
        self.eps = 1e-5  # Decimal tolerance for error checks

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run features downsampling layer test.

        :return: True if :class:`.FeaturesDownsamplingLayer` works as expected
            for the test cases, False otherwise.
        :rtype: bool
        """
        # Generate test data
        points_per_axis = 10
        num_points = points_per_axis**3
        num_encoding_neighbors = 3
        num_features = 5
        t = np.linspace(0, 2, points_per_axis)
        o = np.ones(3)
        X1 = np.array([
            [x, y, z]
            for x in t for y in t for z in t
        ])
        F1 = np.random.normal(0, 1, (num_points, num_features))
        X2 = X1 / np.array([1.0, 1.0, 2.0])
        F2 = np.random.normal(0, 1, (num_points, num_features))
        rf1 = ReceptiveFieldFPS(
            num_points=num_points//3,
            num_encoding_neighbors=num_encoding_neighbors,
            fast=False
        )
        rf1.fit(X1, o)
        rf2 = ReceptiveFieldFPS(
            num_points=num_points//3,
            num_encoding_neighbors=num_encoding_neighbors,
            fast=False
        )
        rf2.fit(X2, o)
        X1b = rf1.centroids_from_points(X1)  # Next depth structure space
        X2b = rf2.centroids_from_points(X2)
        ND1 = rf1.N  # Indexing matrix governing the downsampling
        ND2 = rf2.N
        inputs = [
            np.array([X1, X2]),
            np.array([X1b, X2b]),
            np.array([F1, F2]),
            np.array([ND1, ND2])
        ]
        # Instantiate features downsampling layer
        # TODO Rethink : Implement
        fdl = FeaturesDownsamplingLayer()
        with tf.device("cpu:0"):
            fdl_out = fdl.call(inputs)
        # Return
        return True
