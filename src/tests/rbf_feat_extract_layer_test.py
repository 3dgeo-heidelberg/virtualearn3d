# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.rbf_feat_extract_layer import \
    RBFFeatExtractLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class RBFFeatExtractLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Radial basis function feature extraction layer test checks that the
    operations of a radial basis function feature extraction layer yield the
    expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('RBF feature extraction layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run RBF feature extraction layer test.

        :return: True if RBFFeatExtractLayer works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        # Instantiate test layer
        max_radii = (1.5, 1.5, 1.5)
        radii_resolution = 4
        angular_resolutions = (1, 2, 4, 8)
        kernel_function_type = 'Gaussian'
        structure_initialization_type = 'concentric_ellipsoids'
        trainable_Q, trainable_omega = [True]*2
        rfel = RBFFeatExtractLayer(
            max_radii=max_radii,
            radii_resolution=radii_resolution,
            angular_resolutions=angular_resolutions,
            kernel_function_type=kernel_function_type,
            structure_initialization_type=structure_initialization_type,
            trainable_Q=trainable_Q,
            trainable_omega=trainable_omega
        )
        # Generate test data
        bs = 5  # Batch size
        m = 128  # Num points
        nx = 3  # Structure dim
        K = rfel.num_kernel_points
        X = np.random.normal(0, 1, (bs, m, nx)).astype("float32")
        # Build layer
        rfel.build([bs, m, nx])
        # Call layer
        with tf.device("cpu:0"):
            rfel_out = rfel.call(X)
        # Compute expected output
        expected_out = self.compute_expected_output(
            X,
            np.array(rfel.Q, dtype='float32'),
            np.array(rfel.omega, dtype='float32'),
        )
        # Compare RFEL with expected
        return self.validate_output(rfel_out, expected_out)

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def compute_expected_output(self, X, Q, omega):
        """
        Compute the expected output to compare it against what is generated
        by the RBF feature extraction layer.

        :param X: The input.
        :param Q: The kernel's structure.
        :param omega: The kernel's sizes.
        :return: The expected output.
        :rtype: :class:`np.ndarray`
        """
        # Compute distances
        D = np.empty(
            (X.shape[0], X.shape[1], Q.shape[0]),
            dtype='float32'
        )
        for slice_idx in range(X.shape[0]):
            for i in range(X.shape[1]):
                for j in range(Q.shape[0]):
                    try:
                        # Compute Gaussian kernel function
                        D[slice_idx, i, j] = np.exp(
                            -np.sum(np.square(X[slice_idx, i, :]-Q[j, :])) / (
                                omega[j]*omega[j]
                            )
                        )
                    except FloatingPointError as fperr:
                        D[slice_idx, i, j] = 0
        # Return expected values
        return D

    def validate_output(self, rfel, expected):
        """
        Check whether the RBF feature extraction layer yields the expected
        results or not.

        :return: True if the RBFEL output is not different in more than a
            given decimal tolerance (eps) wrt the expected output.
        """
        return np.max(np.abs(rfel-expected)) < self.eps

