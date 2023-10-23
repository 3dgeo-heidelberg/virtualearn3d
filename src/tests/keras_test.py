# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest, VL3DTestException


# ---   CLASS   --- #
# ----------------- #
class KerasTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Keras test that checks TensorFlow can be used and there is an accessible
    GPU.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Keras, TensorFlow, GPU test')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run Keras test.

        :return: True if Keras can be used in GPU mode with TensorFlow
            background, False otherwise.
        :rtype: bool
        """
        # Check imports
        import tensorflow as tf
        from tensorflow.python.client import device_lib

        # Check some basic operations
        k = tf.constant([[5, 2], [1, 3]])
        no_negative_constraint = tf.keras.constraints.NonNeg

        # Check GPU support
        devs = device_lib.list_local_devices()
        has_gpu = False
        for dev in devs:
            if str.lower(dev.device_type) == "gpu":
                has_gpu = True
                break
        if not has_gpu:
            return False

        # Test successfully passed
        return True
