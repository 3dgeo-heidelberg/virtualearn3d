# ---   IMPORTS   --- #
# ------------------- #
from src.tests.hsv_from_rgb_test import HSVFromRGBTest
from src.tests.keras_test import KerasTest
from src.tests.receptive_field_test import ReceptiveFieldTest
from src.tests.hierarchical_receptive_field_test import \
    HierarchicalReceptiveFieldTest
from src.tests.features_structuring_layer_test import \
    FeaturesStructuringLayerTest
from src.tests.rbf_feat_extract_layer_test import \
    RBFFeatExtractLayerTest
from src.tests.rbf_feat_processing_layer_test import \
    RBFFeatProcessingLayerTest
from src.tests.features_downsampling_layer_test import \
    FeaturesDownsamplingLayerTest
from src.tests.features_upsampling_layer_test import \
    FeaturesUpsamplingLayerTest
from src.tests.grouping_point_net_layer_test import \
    GroupingPointNetLayerTest
from src.tests.kpconv_layer_test import KPConvLayerTest
from src.tests.strided_kpconv_layer_test import StridedKPConvLayerTest
from src.tests.model_serialization_test import ModelSerializationTest
from src.tests.las_inout_test import LASInoutTest
from src.tests.vl3dpp_binding_test import VL3DPPBindingTest
from src.tests.vl3dpp_backend_test import VL3DPPBackendTest
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
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
        TEST = VL3DCFG.get('TEST', None)
        if TEST is None:
            raise ValueError('No TEST YAML specification was found.')
        pass_count = 0
        # ---------------------------------------------------------------------
        # Run error-detection tests (like asserts)
        error_count = 0
        error_suite = TEST.get('ErrorTestSuites')
        if error_suite is not None:
            for suite_name, suite_tests in error_suite.items():
                LOGGING.LOGGER.info(
                    '\033[1m\033[31m'
                    f'ERROR-TEST SUITE: {suite_name}'
                    '\033[0m'
                )
                for test in suite_tests:
                    if not globals()[test]().start():
                        error_count += 1
                    else:
                        pass_count += 1
        # ---------------------------------------------------------------------
        # Run warning-emitting tests (just reports)
        exception_count = 0
        warning_count = 0
        warning_suite = TEST.get('WarningTestSuites')
        if warning_suite is not None:
            for suite_name, suite_tests in warning_suite.items():
                LOGGING.LOGGER.info(
                    '\033[1m\033[33m'
                    f'WARNING-TEST SUITE: {suite_name}'
                    '\033[0m'
                )
                for test in suite_tests:
                    try:
                        if not globals()[test]().start():
                            warning_count += 1
                        else:
                            pass_count += 1
                    except Exception as ex:
                        LOGGING.LOGGER.warning(
                            '\033[38;5;208m'
                            f'Exception raised by {test}'
                            '\033[0m'
                        )
                        exception_count += 1
        """success = success and HSVFromRGBTest().start()
        success = success and KerasTest().start()
        success = success and ReceptiveFieldTest().start()
        success = success and HierarchicalReceptiveFieldTest().start()
        success = success and FeaturesDownsamplingLayerTest().start()
        success = success and FeaturesUpsamplingLayerTest().start()
        success = success and GroupingPointNetLayerTest().start()
        success = success and KPConvLayerTest().start()
        success = success and StridedKPConvLayerTest().start()
        success = success and FeaturesStructuringLayerTest().start()
        success = success and RBFFeatExtractLayerTest().start()
        success = success and RBFFeatProcessingLayerTest().start()
        success = success and ModelSerializationTest().start()
        success = success and LASInoutTest().start()
        #success = success and VL3DPPBindingTest().start()
        #success = success and VL3DPPBackendTest().start()"""
        # ---------------------------------------------------------------------
        # Report results
        total_count = error_count + exception_count + warning_count + pass_count
        LOGGING.LOGGER.info(
            '\n'
            '\033[91m'
            f'Number of errors: {error_count}\n'
            '\033[0m'
            '\033[38;5;208m'
            f'Number of exceptions: {exception_count}\n'
            '\033[0m'
            '\033[38;5;226m'
            f'Number of warnings: {warning_count}\n'
            '\033[0m'
            '\033[92m'
            f'Number of passed tests: {pass_count}\n'
            '\033[0m'
            f'Number of total tests: {total_count}'
        )
        # Return
        return error_count < 1
