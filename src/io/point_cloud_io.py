# ---   IMPORTS   --- #
# ------------------- #
from src.pcloud.point_cloud import PointCloud
import laspy
import os


# ---   CLASS   --- #
# ----------------- #
class PointCloudIO:

    # ---  READ / LOAD  --- #
    # --------------------- #
    @staticmethod
    def read(path):
        # Validate input path as file
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'Cannot find file at given input path:\n'
                f'"{path}"'
            )
        # Read and return point cloud
        return PointCloud(laspy.read(path))

    # ---  WRITE / STORE  --- #
    # ----------------------- #
    @staticmethod
    def write(pcloud, path):
        # Validate output directory
        outdir = os.path.dirname(path)
        if not os.path.isdir(outdir):
            raise NotADirectoryError(
                'The parent of the output path is not a directory:\n'
                f'"{outdir}"'
            )
        # Write output point cloud
        pcloud.las.write(path)
