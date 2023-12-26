# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.rbf_feat_processing_layer import \
    RBFFeatProcessingLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class RBFFeatProcessingLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Radial basis function feature processing layer test checks that the
    operations of a radial basis function feature processing layer yield the
    expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('RBF feature processing layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run RBF feature processing test.

        :return: True if RBFFeatProcessingLayer works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        # Instantiate test layer
        num_batches, num_points, num_feats = 8, 256, 10
        F = np.random.normal(
            0,
            1,
            (num_batches, num_points, num_feats),
        ).astype(dtype='float32')
        Fvstack = np.vstack(F)
        means = np.mean(Fvstack, axis=0)
        stdevs = np.std(Fvstack, axis=0)
        num_kernels = 16
        a, b = 0.01, 1
        rfpl = RBFFeatProcessingLayer(
            num_kernels=num_kernels,
            means=means,
            stdevs=stdevs,
            a=a,
            b=b,
            kernel_function_type='Gaussian',
            trainable_M=False,
            trainable_Omega=False
        )
        # Build layer
        rfpl.build(F.shape)
        # Call layer
        with tf.device("cpu:0"):
            rfpl_out = np.array(rfpl.call(F), dtype='float32')
        # Compute expected output
        expected_out = self.compute_expected_output(
            F,
            np.array(rfpl.M, dtype='float32'),
            np.array(rfpl.Omega, dtype='float32')
        )
        # Compare RFPL with expected
        return self.validate_output(rfpl_out, expected_out)

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def compute_expected_output(self, F, M, Omega):
        """
        Compute the expected output to compare it against what is generated
        by the RBF feature processing layer.

        :param F: The input.
        :param M: The kernel's centers.
        :param Omega: The kernel's sizes.
        :return: The expected output.
        :rtype: :class:`np.ndarray`
        """
        num_batches, num_points, num_in_feats = F.shape
        num_kernels = M.shape[0]
        num_out_feats = num_kernels * num_in_feats
        # Generate output batches
        tensor_Y = []
        for batch_idx in range(num_batches):
            # Generate output batch
            Y = np.ndarray((num_points, num_out_feats), dtype='float32')
            Fbatch = F[batch_idx]
            for p in range(num_out_feats):  # For each output feature
                k = p % num_kernels  # Kernel index
                j = p // num_kernels  # Input feature index
                for i in range(num_points):  # For each point
                    Y[i, p] = (Fbatch[i, j]-M[k, j])/Omega[k, j]
            Ysquared = Y*Y
            zero_mask = Ysquared > 50
            Ysquared[zero_mask] = 0
            Y = np.exp(-Ysquared)
            Y[zero_mask] = 0
            # Component-wise exponentiation
            # Append output batch
            tensor_Y.append(Y)
        # Return
        return np.array(tensor_Y)

    def validate_output(self, rfpl, expected):
        """
        Check whether the RBF feature processing layer yields the expected
        results or not.

        :return: TRUE if the RBFPL output is not different in more than a given
            decimal tolerance (eps) wrt the expected output.
        """
        return np.max(np.abs(rfpl-expected)) < self.eps