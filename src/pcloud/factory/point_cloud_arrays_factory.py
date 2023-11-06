# ---   IMPORTS   --- #
# ------------------- #
from src.pcloud.factory.point_cloud_factory import PointCloudFactory, \
    PointCloudFactoryException
from src.pcloud.point_cloud import PointCloud
import laspy
import numpy as np
import copy


# ---   CLASS   --- #
# ----------------- #
class PointCloudArraysFactory(PointCloudFactory):
    """
    :author: Alberto M. Esmoris Pena

    Class to instantiate PointCloud objects from arrays.
    See :class:`.PointCloud` and also :class:`.PointCloudFactory`.

    :ivar X: The matrix representing the coordinates of the points.
    :vartype X: :class:`np.ndarray`
    :ivar F: The matrix representing the point-wise features.
    :vartype F: :class:`np.ndarray`
    :ivar y: The vector representing the point-wise classes.
    :vartype y: :class:`np.ndarray`
    :ivar header: The LAS header.
    :ivar fnames: The name for each feature.
    :vartype fnames: list or tuple
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, X, F, y=None, header=None, fnames=None, **kwargs):
        """
        Initialize an instance of PointCloudArraysFactory.

        :param X: The matrix of point coordinates.
        :param F: The matrix of point-wise features.
        :param y: The vector of classes (OPTIONAL).
        :param header: The LAS header (OPTIONAL).
        :param fnames: The name for each feature (OPTIONAL).
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.X = X
        self.F = F
        self.y = y
        self.header = header
        self.fnames = fnames
        # Validate
        if self.F is not None and self.X.shape[0] != self.F.shape[0]:
            raise PointCloudFactoryException(
                'PointCloudArraysFactory cannot deal with coordinates for '
                f'{self.X.shape[0]} points and features for {self.F.shape[0]} '
                'points.'
            )
        # Load defaults when necessary
        if self.F is not None and self.fnames is None:
            self.fnames = [f'f{i}' for i in range(1, self.F.shape[1]+1)]

    # ---  FACTORY METHODS  --- #
    # ------------------------- #
    def make(self):
        """
        Make a point cloud from arrays.
        See :meth:`point_cloud_factory.PointCloudFactory.make`
        """
        # Initialize LAS
        if self.header is None:  # Initialize from scratch
            las = laspy.create(
                point_format=1,
                file_version="1.2"
            )
            las.header.offsets = np.min(self.X, axis=0)
            las.header.scales = [0.01]*3
            if self.F is not None:
                extra_bytes = [
                    laspy.ExtraBytesParams(name=fname, type='f')
                    for fname in self.fnames
                ]
                las.add_extra_dims(extra_bytes)
        else:  # Initialize from previous header
            header = copy.deepcopy(self.header)
            header.point_count = self.X.shape[0]
            las = laspy.LasData(header)
        # Assign coordinates
        las.x = self.X[:, 0]
        las.y = self.X[:, 1]
        las.z = self.X[:, 2]
        # Assign features
        if self.F is not None:
            for i, fname in enumerate(self.fnames):
                las[fname] = self.F[:, i]
        # Assign classification
        if self.y is not None:
            las.classification = self.y
        # Return point cloud
        return PointCloud(las)
