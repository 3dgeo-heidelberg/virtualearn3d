from src.tests.vl3d_test import VL3DTest, VL3DTestException
from src.vl3dpp import vl3dpp_loader


class VL3DPPBindingTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    C++ link test.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ binding test')

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run C++ link test.

        :return: True if C++ binding is working, False otherwise.
        :rtype: bool
        """
        # Load and import
        vl3dpp_loader.vl3dpp_load(logging=False, warning=True)
        import pyvl3dpp as vl3dpp
        # Get HELLO WORLD string
        s = vl3dpp.get_hello_world()
        # Validate string
        if s != 'HELLO WORLD':  # Test : failed
            return False
        return True  # Test : success
