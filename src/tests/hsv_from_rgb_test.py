# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.mining.hsv_from_rgb_miner import HSVFromRGBMiner
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class HSVFromRGBTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    HSV from RGB test that checks the HSV color representation is correctly
    derived from given RGB color components.
    """
    # ---   INIT   --- #
    # ⨪--------------- #
    def __init__(self):
        super().__init__('HSV from RGB test')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run HSV from RGB test.

        :return: True if HSVFromRGB works as expected for the test cases,
            False otherwise.
        :rtype: bool
        """
        # Build test case
        pcloud1 = PointCloudFactoryFacade.make_from_arrays(
            np.random.normal(0, 1, (5, 3)),  # Point coordinates
            np.array([  # Point-wise RGB color components
                [200/255, 150/255, 100/255],
                [222/255, 50/255, 10/255],
                [20/255, 20/255, 151/255],
                [0, 0, 1],
                [85/255, 85/255, 85/255]
            ]),
            fnames=['red', 'green', 'blue']
        )
        pcloud2 = PointCloudFactoryFacade.make_from_arrays(
            np.random.normal(0, 1, (3, 3)),  # Point coordinates
            np.array([  # Point-wise RGB color components
                [0, 255, 0],
                [92, 128, 11],
                [111, 172, 200]
            ]),
            fnames=['red', 'green', 'blue']
        )
        EXPECTED_HSV1 = np.array([  # The expected HSV values (case 1)
            [30, 0.5, 0.784],
            [11.32, 0.955, 0.871],
            [240, 0.868, 0.592],
            [240, 1, 1],
            [0, 0, 0.333]
        ])
        EXPECTED_HSV2 = np.array([  # The expected HSV values (case 2)
            [120, 1, 1],
            [78.46, 0.914, 0.502],
            [198.88, 0.445, 0.784],
        ])
        # Compute HSV
        pcloud1 = HSVFromRGBMiner(
            frenames=['H', 'S', 'V'],
            hue_unit='degrees'
        ).mine(pcloud1)
        HSV1 = pcloud1.get_features_matrix(['H', 'S', 'V'])
        pcloud2 = HSVFromRGBMiner(
            frenames=['H', 'S', 'V'],
            hue_unit='degrees'
        ).mine(pcloud2)
        HSV2 = pcloud2.get_features_matrix(['H', 'S', 'V'])
        # Validate results
        eps = 0.1
        return (
            (not np.any(np.abs(EXPECTED_HSV1-HSV1) > eps)) and
            (not np.any(np.abs(EXPECTED_HSV2-HSV2) > eps))
        )
