# ---   IMPORTS   --- #
# ------------------- #
from src.tests.hsv_from_rgb_test import HSVFromRGBTest
from src.tests.keras_test import KerasTest
from src.tests.receptive_field_test import ReceptiveFieldTest
from src.tests.features_structuring_layer_test import \
    FeaturesStructuringLayerTest
from src.tests.rbf_feat_extract_layer_test import \
    RBFFeatExtractLayerTest
from src.tests.rbf_feat_processing_layer_test import \
    RBFFeatProcessingLayerTest
from src.tests.model_serialization_test import ModelSerializationTest
from src.tests.vl3dpp_binding_test import VL3DPPBindingTest
from src.tests.vl3dpp_backend_test import VL3DPPBackendTest
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class MainTest:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for tests.
    """
    # ---  MAIN METHODS  --- #
    # ---------------------- #
    @staticmethod
    def main():
        """
        Entry point logic for tests.

        :return: Boolean flag indicating whether all tests passed successfully
            (True) or not (False).
        :rtype: bool
        """
        # Prepare tests
        np.seterr(all='raise')  # Make numpy raise warnings
        # Initialize flag
        success = True
        # ---------------------------------------------------------------------
        # Run tests
        success = success and HSVFromRGBTest().start()
        success = success and KerasTest().start()
        success = success and ReceptiveFieldTest().start()
        success = success and FeaturesStructuringLayerTest().start()
        success = success and RBFFeatExtractLayerTest().start()
        success = success and RBFFeatProcessingLayerTest().start()
        success = success and ModelSerializationTest().start()
        success = success and VL3DPPBindingTest().start()
        success = success and VL3DPPBackendTest().start()
        # ---------------------------------------------------------------------
        # Return
        return success
