from src.tests.vl3d_test import VL3DTest, VL3DTestException
from src.vl3dpp import vl3dpp_loader


class VL3DPPBackendTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    C++ backend test.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ backend test')

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run C++ backend test.

        :return: True if C++ backend is working, False otherwise.
        :rtype: bool
        """
        # Load and import
        vl3dpp_loader.vl3dpp_load(logging=False, warning=True)
        import pyvl3dpp as vl3dpp
        # Get example matrix
        import numpy as np
        X = np.random.normal(0, 1, (25, 3))
        F = np.random.normal(0, 1, (25, 16))
        Fhat = vl3dpp.mine_smooth_feats(X, F)
        print(Fhat)
        return True  # Test : success
