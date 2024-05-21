# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.ptransf.fps_decorator_transformer import FPSDecoratorTransformer
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratorTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    FPS decorator test that checks the FPS representation of point clouds is
    correctly computed.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('FPS decorator test')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run FPS decorator test.

        :return: True if the FPS decorator works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        # Build test data
        X = np.array([
            [-10, 10, 0],
            [-9.9, 10.1, 0],
            [11, 11, 0],
            [11.1, 10.9, 0],
            [0, 0, 0],
            [0.1, 0, 0]
        ])
        F1 = np.array([
            [0, 1], [0, 1], [10, 11], [10, 11], [100, 111], [100, 111]
        ])
        ErepF1a = np.array([[0, 1], [10, 11], [100, 111]])
        EproF1a = np.array(F1)
        ErepF1b = np.array([
            [33.333333333, 37.666666666],
            [40, 44.333333333],
            [66.666666666, 74.333333333]
        ])
        EproF1b = np.array([[46.666666666, 52.111111111]]*6)
        F2 = F1[:, 0]
        ErepF2a = ErepF1a[:, 0]
        EproF2a = EproF1a[:, 0]
        ErepF2b = ErepF1b[:, 0]
        EproF2b = EproF1b[:, 0]
        # Build transformers
        fpsdt1a = FPSDecoratorTransformer(
            num_points=3,
            num_encoding_neighbors=1,
            num_decoding_neighbors=1,
            release_encoding_neighborhoods=False,
            fast=False,
            threads=1,
            representation_report_path=None
        )
        fpsdt1b = FPSDecoratorTransformer(
            num_points=3,
            num_encoding_neighbors=3,
            num_decoding_neighbors=3,
            release_encoding_neighborhoods=False,
            fast=False,
            threads=1,
            representation_report_path=None
        )
        fpsdt2a = FPSDecoratorTransformer(
            num_points=3,
            num_encoding_neighbors=1,
            num_decoding_neighbors=1,
            release_encoding_neighborhoods=False,
            fast=False,
            threads=1,
            representation_report_path=None
        )
        fpsdt2b = FPSDecoratorTransformer(
            num_points=3,
            num_encoding_neighbors=3,
            num_decoding_neighbors=3,
            release_encoding_neighborhoods=False,
            fast=False,
            threads=1,
            representation_report_path=None
        )
        # Validate transformers
        if not self.validate_transformer(fpsdt1a, X, F1, ErepF1a, EproF1a):
            return False
        if not self.validate_transformer(fpsdt1b, X, F1, ErepF1b, EproF1b):
            return False
        if not self.validate_transformer(fpsdt2a, X, F2, ErepF2a, EproF2a):
            return False
        if not self.validate_transformer(fpsdt2b, X, F2, ErepF2b, EproF2b):
            return False
        # Return true if all transformers were successfully validated
        return True

    @staticmethod
    def validate_transformer(fpsdt, X, F, ErepF, EproF):
        """
        Check whether the given :class:`.FPSDecoratorTransformer` yields the
        expected output.

        :param fpsdt: The :class:`.FPSDecoratorTransformer` to be validated.
        :type fpsdt: :class:`.FPSDecoratorTransformer`
        :param X: The input structure space matrix (i.e., matrix of point-wise
            coordinates).
        :type X: :class:`np.ndarray`
        :param F: The input feature space matrix (i.e., matrix of point-wise
            features).
        :type F: :class:`np.ndarray`
        :param ErepF: The expected features in the FPS representation space.
        :type ErepF: :class:`np.ndarray`
        :param EproF: The expected features after propagating the features
            in the representation space back to the original space.
        :type EproF: :class:`np.ndarray`
        :return: True if the :class:`.FPSDecoratorTransformer` yielded the
            expected output, False otherwise.
        :rtype: bool
        """
        repF = fpsdt.transform(X, F=F)[1]
        if not np.allclose(repF, ErepF):
            return False
        proF = fpsdt.propagate(repF)
        if not np.allclose(proF, EproF):
            return False
        return True
