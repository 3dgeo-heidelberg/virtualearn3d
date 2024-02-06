# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.pcloud.point_cloud import PointCloud
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import numpy as np
import tempfile
import os


# ---   CLASS   --- #
# ----------------- #
class LASInoutTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    LAS input/output test that checks that a LAS (or LAZ) file can be written
    and read such that the read information matches the one before writing.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__("LAS/LAZ input/output test")

    # ---  TEST INTERFACE  --- #
    # ⨪----------------------- #
    def run(self):
        """
        Run LAS/LAZ input/output test.

        :return: True if LASInoutTest works as expected for the test cases,
            False otherwise.
        :rtype: bool
        """
        # Build test case
        pcloud1 = PointCloudFactoryFacade.make_from_arrays(
            np.random.normal(0, 1, (128, 3)),
            np.random.normal(0, 1, (128, 5)),
            fnames=['intensity', 'red', 'green', 'blue', 'planarity']
        )
        # Temporary directory as workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            pcloud1_las_path = os.path.join(tmpdir, 'pcloud1.las')
            PointCloudIO.write(pcloud1, pcloud1_las_path)
            pcloud1_las = PointCloudIO.read(pcloud1_las_path)
            if not pcloud1.equals(pcloud1_las, compare_header=False):
                return False
            pcloud1_laz_path = os.path.join(tmpdir, 'pcloud1.laz')
            PointCloudIO.write(pcloud1, pcloud1_laz_path)
            pcloud1_laz = PointCloudIO.read(pcloud1_laz_path)
            if not pcloud1.equals(pcloud1_laz, compare_header=False):
                return False
        # All checks were passed
        return True
