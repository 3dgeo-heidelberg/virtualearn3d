# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
import numpy as np
import laspy


# ---  EXCEPTIONS  --- #
# -------------------- #
class PointCloudException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to the PointCloud data structure and
    associated methods.
    See :class:`.VL3DException`
    """
    def __init__(self, message):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PointCloud:
    """
    :author: Alberto M. Esmoris Pena

    Base class representing the point cloud data structure and
        providing basic data handling methods.

    :ivar las: The LASPY data structure representing the point cloud.
    :vartype las: See :mod:`laspy`
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, las):
        """
        Initialize a PointCloud instance.
        See the :class:`.PointCloudFactoryFacade` for methods to successfully
        built PointCloud instances, e.g., by reading a LAS file.

        :param las: The LASPY datastructure representing the point cloud.
        """
        self.las = las

    # ---  ACCESS-ONLY METHODS  --- #
    # ----------------------------- #
    def get_num_points(self):
        """
        Obtain the number of points in the point cloud.

        :return: Number of points in the point cloud.
        :rtype: int
        """
        return len(self.las.X)

    def get_coordinates_matrix(self):
        """
        Obtain the matrix of coordinates representing the point cloud (supported
        for 3D point clouds only).

        :return: The matrix of coordinates representing the point cloud.
        :rtype: :class:`np.ndarray`
        """
        scales, offsets = self.las.header.scales, self.las.header.offsets
        return np.array([
            self.las.X * scales[0] + offsets[0],
            self.las.Y * scales[1] + offsets[1],
            self.las.Z * scales[2] + offsets[2]
        ]).T

    def get_features_matrix(self, fnames):
        """
        Obtain the matrix of features representing the point cloud.

        :param list fnames: The name of the point cloud's attributes
            corresponding to the features of interest.
        :return: The matrix of features representing the point cloud.
        :rtype: :class:`np.ndarray`
        """
        return np.array([self.las[fname] for fname in fnames]).T

    def get_classes_vector(self):
        """
        Obtain a vector which components represent the point-wise classes of
        the point cloud.

        :return: The vector of classes.
        :rtype: :class:`np.ndarray`
        """
        return self.las.classification

    # ---  UPDATE METHODS  --- #
    # ------------------------ #
    def add_features(self, fnames, feats, ftypes="f"):
        """
        Add new features to the point cloud.

        :param fnames: The name of the features in a list or tuple format.
        :param feats: The matrix of features such that each column represents
            a feature and each row a point (as a numpy array). See
            :class:`np.ndarray`.
        :param ftypes: The list or tuple of types representing the type for
            each new feature. If it is a single type, then all feature are
            assumed to have the same type.
        :return: The updated point cloud.
        :rtype: :class:`PointCloud`
        """
        # Extract useful information
        nfeats = feats.shape[1]
        # Check each feature has its own name
        if len(fnames) != nfeats:
            raise PointCloudException(
                "There is no one-to-one relationship between features and "
                "names."
            )
        # Handle single type specification
        if not isinstance(ftypes, (list, tuple)):
            ftypes = [ftypes for i in range(nfeats)]
        # Add extra dimensions to the output LAS
        extra_bytes = []
        for i, fname in enumerate(fnames):  # For each new feature (i)
            extra_bytes.append(
                laspy.ExtraBytesParams(name=fname, type=ftypes[i])
            )
        # Create the new point cloud
        las = laspy.LasData(self.las.header)
        las.points = self.las.points.copy()
        las.add_extra_dims(extra_bytes)
        for i in range(nfeats):
            las[fnames[i]] = feats[:, i]
        # Replace the old point cloud with the new one
        self.las = las
        # Return
        return self
