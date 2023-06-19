# ---   IMPORTS   --- #
# ------------------- #
from src.pcloud.factory.point_cloud_factory import PointCloudFactory
from src.io.point_cloud_io import PointCloudIO


# ---   CLASS   --- #
# ----------------- #
class PointCloudFileFactory(PointCloudFactory):
    """
    :author: Alberto M. Esmoris Pena
    :brief: Class to instantiate PointCloud objects from files.
    See :class:`PointCloud` and also :class:`PointCloudFactory`
    :ivar path: The path where the input point cloud file is located.
    :vartype path: str
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, path):
        """
        Initialize an instance of PointCloudFileFactory.
        :param path: The path to the file representing a point cloud (must be
            stored in LAS format).
        """
        # Call parent init
        super(PointCloudFactory).__init__()
        # Basic attributes of the PointCloudFileFactory
        self.path = path

    # ---  FACTORY METHODS  --- #
    # ------------------------- #
    def make(self):
        """
        Make a point cloud from a file.
        See :method:`PointCloudFactory.make()`
        """
        return PointCloudIO.read(self.path)
