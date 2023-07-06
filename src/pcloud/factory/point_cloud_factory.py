# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException


# ---   EXCEPTIONS   --- #
# ---------------------- #
class PointCloudFactoryException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to factories of point clouds.

    See :class:`.VL3DException`.
    """
    def __init__(self, message):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PointCloudFactory:
    """
    :author: Alberto M. Esmoris Pena

    Interface governing any point cloud factory.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        pass

    # ---  FACTORY METHODS  --- #
    # ------------------------- #
    @abstractmethod
    def make(self):
        """
        Make a point cloud

        :return: The built point cloud.
        :rtype: :class:`.PointCloud`
        """
        pass
