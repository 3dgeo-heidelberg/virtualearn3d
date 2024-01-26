# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.grouping_point_net_layer import \
    GroupingPointNetLayer
from scipy.spatial import KDTree as KDT
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class GroupingPointNetLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Grouping PointNet layer test that checks the operations of a
    grouping PointNet layer yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Grouping PointNet layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run grouping PointNet layer test.

        :return: True if :class:`.GroupingPointNetLayer` works as expected
            for the test cases, False otherwise.
        :rtype: bool
        """
        # Generate test data
        points_per_axis = 10
        num_points = points_per_axis**3
        num_near_neighs = 8
        num_features = 6
        dim_out = 7
        t = np.linspace(0, 2, points_per_axis)
        o = np.ones(3)
        X1 = np.array([
            [x, y, z]
            for x in t for y in t for z in t
        ])
        F1 = np.random.normal(0, 1, (num_points, num_features))
        N1 = KDT(X1).query(X1, k=num_near_neighs)[1]
        X2 = X1 / np.array([1.0, 1.0, 2.0])
        F2 = np.random.normal(0, 1, (num_points, num_features))
        N2 = KDT(X2).query(X2, k=num_near_neighs)[1]
        inputs = [
            np.array([X1, X2], dtype='float32'),
            np.array([F1, F2], dtype='float32'),
            np.array([N1, N2], dtype='int')
        ]
        # Instantiate grouping PointNet layer
        gpnl = GroupingPointNetLayer(dim_out)
        gpnl.build([inputs[i].shape for i in range(len(inputs))])
        # Compute grouping PointNet layer
        with tf.device("cpu:0"):
            gpnl_out = gpnl.call(inputs)
        # TODO Rethink : Implement
        return True
