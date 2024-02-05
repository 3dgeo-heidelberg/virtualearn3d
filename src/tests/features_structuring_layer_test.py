# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.features_structuring_layer import \
    FeaturesStructuringLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesStructuringLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Features structuring layer test that checks the operations of a
    features structuring layer yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Features structuring layer test')
        self.eps = 1e-5  # Decimal tolerance for error checks

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run features structuring layer test.

        :return: True if :class:`.FeaturesStructuringLayer` works as expected
            for the test cases, False otherwise.
        :rtype: bool
        """
        # Instantiate test layer
        max_radii = (1, 1, 1)
        radii_resolution = 4
        angular_resolutions = (1, 2, 4, 8)
        dim_out = 4
        fsl = FeaturesStructuringLayer(
            max_radii=max_radii,
            radii_resolution=radii_resolution,
            angular_resolutions=angular_resolutions,
            dim_out=dim_out,
            concatenation_strategy='OPAQUE',
            name='FSL'
        )
        # Generate test data
        bs = 5  # Batch size
        m = 128  # Num points
        nx = 3  # Structure dim
        nf = 8  # Features dim
        K = fsl.num_kernel_points
        X = np.random.normal(0, 1, (bs, m, nx)).astype("float32")
        F = np.random.normal(0, 1, (bs, m, nf)).astype("float32")
        # Build layer
        fsl.build([[bs, m, nx], [bs, m, nf]])
        # Call layer
        with tf.device("cpu:0"):
            fsl_out = fsl.call([X, F])
        # Compute tensorflow output
        with tf.device("cpu:0"):
            tf_out = self.tf_compute_output(
                X, F,
                np.array(fsl.omegaF, dtype="float32"),
                np.array(fsl.omegaD, dtype="float32"),
                np.array(fsl.QX, dtype="float32"),
                np.array(fsl.QW, dtype="float32")
            )
        # Compute expected output
        expected_out = self.compute_expected_output(
            X, F,
            np.array(fsl.omegaF, dtype="float32"),
            np.array(fsl.omegaD, dtype="float32"),
            np.array(fsl.QX, dtype="float32"),
            np.array(fsl.QW, dtype="float32"),
            m, nf, K
        )
        # Compare tf with expected
        valid = self.validate_output(tf_out, expected_out)
        # Compare FSL with expected
        valid = valid and self.validate_output(fsl_out, expected_out)
        # Return
        return valid

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def compute_expected_output(self, X, F, omegaF, omegaD, QX, QW, m, nf, K):
        """
        Compute the expected output to compare it against what is generated
        by the features structuring layer.

        :return: The expected output.
        :rtype: :class:`np.ndarray`
        """
        # Distance function
        def dQ(i, qi, x):
            return np.exp(-np.sum(np.power(x-qi, 2))/(omegaD[i]*omegaD[i]))
        # For each element in the batch
        batch_size = X.shape[0]
        out = []
        for batch_idx in range(batch_size):
            # Extract elements from batch
            Xt, Ft = X[batch_idx], F[batch_idx]
            # Compute kernel's distance matrix
            QD = np.zeros((K, m))
            for i in range(K):
                for j in range(m):
                    QD[i, j] = dQ(i, QX[i], Xt[j])
            # Compute kernel's feature matrix (QF)
            """QF = np.zeros((K, nf))
            for i in range(K):
                QF[i, :] = np.sum([
                    dQ(i, QX[i], Xt[j]) * omegaF*Ft[j]
                    for j in range(m)
                ], axis=0)"""
            QF = omegaF/m * (QD @ Ft)
            # Generate output
            out.append(QD.T@QF@QW)
        out = np.array(out)
        return out

    def tf_compute_output(self, X, F, omegaF, omegaD, QX, QW):
        """
        The tensorflow operations defining the layer but outside the layer.

        :return: The structured features.
        """
        X, F = tf.Variable(X, dtype="float32"), tf.Variable(F, dtype="float32")
        m = tf.cast(tf.shape(X)[-2], dtype="float32")  # Num input points
        omegaF, omegaD = tf.Variable(omegaF), tf.Variable(omegaD)
        QX, QW = tf.Variable(QX), tf.Variable(QW)
        SUBTRAHEND = tf.tile(
            tf.expand_dims(QX, 1),
            [1, tf.shape(X)[1], 1]
        )
        SUB = tf.subtract(tf.expand_dims(X, 1), SUBTRAHEND)
        omegaD_squared = omegaD * omegaD
        QDunexp = tf.reduce_sum(SUB*SUB, axis=-1)
        QD = tf.exp(
            -tf.transpose(
                tf.transpose(QDunexp, [0, 2, 1]) / omegaD_squared, [0, 2, 1]
            )
        )
        QF = omegaF/m * tf.matmul(QD, F)
        QY = tf.matmul(QF, QW)
        QDT = tf.transpose(QD, [0, 2, 1])
        QY = tf.matmul(QDT, QY)
        return QY

    def validate_output(self, fsl, expected):
        """
        Check whether the feature structuring layer yields the expected
        results or not.

        :return: True if the FSL output is not different in more than a
            given decimal tolerance (eps) wrt the expected output.
        """
        return np.max(np.abs(fsl-expected)) < self.eps
