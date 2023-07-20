# ---   IMPORTS   --- #
# ------------------- #
from src.pcloud.factory.point_cloud_file_factory import PointCloudFileFactory
from src.pcloud.factory.point_cloud_arrays_factory import \
    PointCloudArraysFactory


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
        Build a PointCloud from a given file path (either in the local file
        system or from a URL pointing to a LAS/LAZ file).

        :param path: Path to a file representing a point cloud (must be stored
            in LAS format). Alternatively, it can be a URL starting with
            "http://" or "https://".
        :type path: str
        :return: Built PointCloud
        :rtype: :class:`.PointCloud`
        """
        return PointCloudFileFactory(path).make()

    @staticmethod
    def make_from_arrays(X, F, y=None, header=None, fnames=None):
        """
        Build a PointCloud from given arrays, and (optionally) header.

        :param X: The matrix of coordinates.
        :param F: The matrix of features.
        :param y: The vector of classes.
        :param header: The LAS header. If None, default header is used.
        :param fnames: The name of each feature. If None, then features will be
            named f1,...,fn.
        :return: Built PointCloud
        :rtype: :class:`.PointCloud`
        """
        return PointCloudArraysFactory(
            X, F, y=y, header=header, fnames=fnames
        ).make()
