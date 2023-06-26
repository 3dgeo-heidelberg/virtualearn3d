# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException


# ---  EXCEPTIONS  --- #
# -------------------- #
class MinerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to data mining components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Miner:
    """
    :author: Alberto M. Esmoris Pena

    Interface governing any miner.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        pass

    # ---  MINER METHODS  --- #
    # ----------------------- #
    @abstractmethod
    def mine(self, pcloud):
        """
        Mine features from a given input point cloud.

        :param pcloud: The input point cloud for which features must be mined.
        :return: The point cloud extended with the mined features.
        :rtype: :class:`.PointCloud`
        """
        pass
