# ---   IMPORTS   --- #
# ------------------- #
from src.tests.keras_test import KerasTest
from src.tests.receptive_field_test import ReceptiveFieldTest
from src.tests.features_structuring_layer_test import \
    FeaturesStructuringLayerTest
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
        success = success and KerasTest().start()
        success = success and ReceptiveFieldTest().start()
        success = success and FeaturesStructuringLayerTest().start()
        # ---------------------------------------------------------------------
        # Return
        return success
