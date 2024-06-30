# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.main.main_config import VL3DCFG
import numpy as np


# ---  EXCEPTIONS  --- #
# -------------------- #
class MinerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to data mining components.
    See :class:`.VL3DException`.
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

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def get_structure_space_matrix(pcloud):
        """
        Obtain the structure space matrix (i.e., matrix of point-wise
        coordinates) considering the mining config.

        If the structure space must be represented with less than 64 bits,
        then it will be shifted before the conversion (the bounding box center
        defines the translation vector) to prevent coordinate corruption when
        the input coordinates are given in a CRS with high numbers.

        :param pcloud: The point cloud whose structure space matrix must be
            obtained.
        :type pcloud: :class:`.PointCloud`
        :return: The structure space matrix representing the point cloud.
        :rtype: :class:`np.ndarray`
        """
        X = pcloud.get_coordinates_matrix()
        structure_space_bits = VL3DCFG['MINING']['structure_space_bits']
        if structure_space_bits == 64:
            return X
        # Compute bounding box
        pmin, pmax = np.min(X, axis=0), np.max(X, axis=0)
        u = (pmin+pmax)/2.0
        # Shift structure space to bounding box center
        X = X-u
        # Change data type
        if structure_space_bits == 32:
            X = X.astype(np.float32)
        elif structure_space_bits == 16:
            X = X.astype(np.float16)
        else:
            raise MinerException(
                'Miner could not change the data type for the '
                'structure space.'
            )
        # Return
        return X

    @staticmethod
    def get_feature_type():
        """
        Determine the data type to be used to represent the feature space.

        :return: The type to be used to represent the features.
        :rtype: :class:`np.dtype`
        """
        ftype = VL3DCFG['MINING']['feature_space_bits']
        if ftype == 64:
            ftype = float
        elif ftype == 32:
            ftype = np.float32
        elif ftype == 16:
            ftype = np.float16
        else:
            raise MinerException(
                'Miner failed to determine the data type '
                'for the features.'
            )
        return ftype

    @staticmethod
    def get_feature_space_matrix(pcloud, fnames):
        """
        Obtain the feature space matrix (i.e., matrix of point-wise features)
        considering the mining config.

        :param pcloud: The point cloud whose feature space matrix must be
            obtained.
        :type pcloud: :class:`.PointCloud`
        :param fnames: The names of the features.
        :type fnames: list of str
        :return: The feature space matrix representing the point cloud.
        :rtype: :class:`np.ndarray`
        """
        F = pcloud.get_features_matrix(fnames)
        ftype = Miner.get_feature_type()
        if F.dtype != ftype:
            F = F.astype(ftype)
        return F





