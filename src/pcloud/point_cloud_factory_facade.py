# ---   IMPORTS   --- #
# ------------------- #
from src.pcloud.factory.point_cloud_file_factory import PointCloudFileFactory


# ---   CLASS   --- #
# ----------------- #
class PointCloudFactoryFacade:
    """
    :author: Alberto M. Esmoris Pena

    Facade-like class offering methods to make point clouds.
    """
    # ---  METHODS TO BUILD A POINT CLOUD  --- #
    # ---------------------------------------- #
    @staticmethod
    def make_from_file(path):
        """
        Build a PointCloud from a given file path.

        :param path: Path to a file representing a point cloud (must be stored
            in LAS format).
        :return: Built PointCloud
        :rtype: :class:`.PointCloud`
        """
        return PointCloudFileFactory(path).make()
