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
        # Validate
        valid = True
        valid = valid and self.validate_no_activation(
            inputs, num_near_neighs, dim_out, gpnl, gpnl_out
        )
        # TODO Rethink : Implement
        return valid

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def validate_no_activation(self, inputs, nneighs, Dout, gpnl, gpnl_out):
        # TODO Rethink : Doc
        X_batch, F_batch, N_batch = inputs
        num_elems_in_batch = X_batch.shape[0]
        gpnl_out = np.array(gpnl_out)
        H, gamma, gamma_bias = list(map(np.array, [
            gpnl.H, gpnl.gamma, gpnl.gamma_bias
        ]))
        for k in range(num_elems_in_batch):
            X, F, N = X_batch[k], F_batch[k], N_batch[k]
            P = np.hstack([X, F])
            P_groups = np.array([P[Ni] for Ni in N])
            PHT = np.tensordot(P_groups, H, axes=[[2], [1]])
            PHT_max = np.max(PHT, axis=1)
            gamma_max = PHT_max @ gamma + gamma_bias
            if not np.allclose(gpnl_out[k], gamma_max, atol=self.eps, rtol=0):
                return False
            # TODO Rethink : Implement
        return True

