# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


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
