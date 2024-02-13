# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.kpconv_layer import KPConvLayer
from scipy.spatial import KDTree as KDT
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class KPConvLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Kernel point convolution (KPConv) layer test that checks the operations of
    a KPConv layer yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Kernel point convolution layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run KPConv layer test.

        :return: True if :class:`.KPConvLayer` works as epxected for the test
            cases, False otherwise.
        :rtype bool:
        """
        # Generate test data
        points_per_axis = 10
        num_points = points_per_axis**3
        num_near_neighs = 8
        num_features = 6
        dim_out = 7
        t = np.linspace(0, 2, points_per_axis)
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
        # Instantiate KPConv layer
        kpcl = KPConvLayer(
            sigma=0.5,
            kernel_radius=1.5,
            num_kernel_points=13,
            deformable=False,
            Dout=dim_out
        )
        kpcl.build([inputs[i].shape for i in range(len(inputs))])
        # Compute grouping PointNet layer
        with tf.device("cpu:0"):
            kpcl_out = kpcl.call(inputs)
        # Validate
        valid = True
        valid = valid and self.validate_no_activation(
            inputs, num_near_neighs, dim_out, kpcl, kpcl_out
        )
        return valid

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def validate_no_activation(self, inputs, nneighs, Dout, kpcl, kpcl_out):
        """
        Check whether the :class:`.KPConvLayer` yielded
        the expected output (True) or not (False).

        :param inputs: The inputs to the layer.
        :param nneighs: The number of neighbors per group.
        :param Dout: The output dimensionality.
        :param kpcl: The layer.
        :type kpcl: :class:`.KPConvLayer`
        :param kpcl_out: The output of the layer.
        :return: True if the output is okay, False otherwise.
        """
        X_batch, F_batch, N_batch = inputs
        num_elems_in_batch = X_batch.shape[0]
        kpcl_out = np.array(kpcl_out)
        Q, W = list(map(np.array, [kpcl.Q, kpcl.W]))
        sigma = kpcl.sigma
        mq = Q.shape[0]  # Number of points defining the kernel
        for batch in range(num_elems_in_batch):
            X, F, N = X_batch[batch], F_batch[batch], N_batch[batch]
            Fout = np.zeros((F.shape[0], Dout), dtype='float32')
            for i, xi in enumerate(X):  # For each point in the receptive field
                for j in N[i]:  # For each point in its neighborhood
                    inner_sum = np.zeros(W[0].shape)
                    for k in range(mq):
                        dist_weight = max(
                            0,
                            1 - np.linalg.norm(X[j]-X[i]-Q[k])/sigma
                        )
                        inner_sum += dist_weight * W[k]
                    Fout[i] += inner_sum.T @ F[j].reshape((-1, 1)).flatten()
            if not np.allclose(kpcl_out[batch], Fout, atol=self.eps, rtol=0):
                return False
        return True
