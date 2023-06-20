# ---   IMPORTS   --- #
# ------------------- #
from src.io.io_utils import IOUtils
from src.pcloud.point_cloud import PointCloud
import laspy
import os


# ---   CLASS   --- #
# ----------------- #
class PointCloudIO:
    """
    :author: Alberto M. Esmoris Pena
    Class with util static methods for input/output operations related
        to point clouds.
    """
    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read(path):
        """
        Read a LAS/LAZ point cloud file.
        :param path: Path pointing to a LAS/LAZ point cloud file.
        :return: :class:`PointCloud`
        """
        # Validate input path as file
        IOUtils.validate_path_to_file(
            path,
            'Cannot find point cloud file at given input path:'
        )
        # Read and return point cloud
        return PointCloud(laspy.read(path))

    # ---  WRITE / STORE  --- #
    # ----------------------- #
    @staticmethod
    def write(pcloud, path):
        """Write a LAS/LAZ point cloud file.
        :param pcloud: The point cloud to be written.
        :param path: Path where the LAS/LAZ file must be written.
        """
        # Validate output directory
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'The parent of the output path is not a directory:'
        )
        # Write output point cloud
        pcloud.las.write(path)
