# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.pcloud.mem_to_file_proxy import MemToFileProxy
import src.main.main_logger as LOGGING
import numpy as np
import laspy
import time


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


# ---  CONSTANTS  --- #
# ------------------- #
"""
The names of the features that are not considered extra dimensions.
These features must not be removed when calling the remove_features method
as this is incompatible with the LASPY backend used to handle point clouds.
"""
NON_EXTRA_DIMS_FEATURES = [
    'intensity', 'return_number', 'number_of_returns',
    'scan_direction_flag', 'edge_of_flight_line', 'classification',
    'synthetic', 'key_point', 'withheld',
    'scan_angle_rank', 'user_data', 'point_source_id',
    'gps_time', 'red', 'green', 'blue'
]


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
        # Assign attributes
        self.las = las
        self.proxy = MemToFileProxy()

    # ---  ACCESS-ONLY METHODS  --- #
    # ----------------------------- #
    def get_num_points(self):
        """
        Obtain the number of points in the point cloud.

        :return: Number of points in the point cloud.
        :rtype: int
        """
        self.proxy_load()
        return len(self.las.X)

    def get_header(self):
        """
        Obtain the header representing the input point cloud.

        :return: Header representing the input point cloud.
        """
        self.proxy_load()
        return self.las.header

    def get_coordinates_matrix(self):
        """
        Obtain the matrix of coordinates representing the point cloud (supported
        for 3D point clouds only).

        :return: The matrix of coordinates representing the point cloud.
        :rtype: :class:`np.ndarray`
        """
        self.proxy_load()
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
        self.proxy_load()
        try:
            return np.array([self.las[fname] for fname in fnames]).T
        except Exception as ex:
            raise PointCloudException(
                'PointCloud get_features_matrix method received unexpected '
                'feature names. Supported feature names for the particular '
                f'point cloud are:\n{self.get_features_names()}'
            ) from ex

    def get_features_names(self):
        """
        Obtain a list of strings representing the name for each feature in
            the point cloud.

        :return: A list with the name for each feature.
        :rtype: list
        """
        self.proxy_load()
        return [
            x
            for x in self.las.point_format.dimension_names
            if x not in ['X', 'Y', 'Z']
        ]

    def has_given_features(self, fnames):
        """
        Check whether the point cloud contains the features represented by
        the given feature names.

        :param fnames: The names of the features to be checked.
        :return: True if all the features specified in fnames are available,
            False otherwise.
        """
        self.proxy_load()
        pcloud_fnames = self.get_features_names()
        return all(fname in pcloud_fnames for fname in fnames)

    def get_classes_vector(self):
        r"""
        Obtain a vector which components represent the point-wise classes of
        the point cloud.

        :return: The vector of classes (:math:`\pmb{y}`).
        :rtype: :class:`np.ndarray`
        """
        self.proxy_load()
        return np.array(self.las.classification)

    def has_classes(self):
        """
        Check whether there are available classes for the point cloud.

        :return: True if classes are available, false otherwise.
        :rtype: bool
        """
        self.proxy_load()
        return self.las.classification is not None and \
            len(self.las.classification) > 0

    def get_predictions_vector(self):
        r"""
        Obtain a vector which components represent the point-wise classes
        of a classification model computed on the point cloud.

        :return: The vector of predicted classes (:math:`\hat{\pmb{y}}`)
        :rtype: :class:`np.ndarray`
        """
        self.proxy_load()
        return np.array(self.las['prediction'])

    def has_predictions(self):
        """
        Check whether there are available predictions for the point cloud.

        :return: True if predictions are available, false otherwise.
        :rtype: bool
        """
        self.proxy_load()
        return 'prediction' in self.get_features_names() and \
            self.las['prediction'] is not None and \
            len(self.las.prediction) > 0

    def equals(self, pcloud, compare_header=True):
        """
        Check whether this (self) and given (pcloud) point clouds are equal
        or not. In this method, equality holds if and only if all the values
        of one point cloud match those of the other.

        :param pcloud: The point cloud to be compared against.
        :type pcloud: :class:`.PointCloud`
        :param compare_header: True to consider the LAS header when comparing
            the point clouds, False otherwise.
        :type compare_header: bool
        :return: True if point clouds are equal, False otherwise.
        :rtype: bool
        """
        self.proxy_load()
        # Equality checks
        if self.get_num_points() != pcloud.get_num_points():
            return False
        if compare_header and self.get_header() != pcloud.get_header():
            return False
        if np.count_nonzero(
            self.get_coordinates_matrix() != pcloud.get_coordinates_matrix()
        ):
            return False
        fnames = self.get_features_names()
        if fnames != pcloud.get_features_names():
            return False
        if np.count_nonzero(
            self.get_features_matrix(fnames) !=
            pcloud.get_features_matrix(fnames)
        ):
            return False
        if self.has_classes() != pcloud.has_classes():
            return False
        if self.has_classes():
            if np.count_nonzero(
                self.get_classes_vector() != pcloud.get_classes_vector()
            ):
                return False
        if self.has_predictions() != pcloud.has_predictions():
            return False
        if self.has_predictions():
            if np.count_nonzero(
                self.get_predictions_vector() !=
                pcloud.get_predictions_vector()
            ):
                return False
        # All checks were passed so both point clouds should be equal
        return True

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
        :rtype: :class:`.PointCloud`
        """
        self.proxy_load()
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
            if len(fname) > 32:
                LOGGING.LOGGER.warning(
                    'PointCloud.add_features truncated the feature name '
                    f'"{fname}" to {fname[:32]}  because no more than 32 '
                    'bytes are supported.'
                )
                fname = fname[:32]
                fnames[i] = fname
            extra_bytes.append(
                laspy.ExtraBytesParams(name=fname, type=ftypes[i])
            )
        # Create the new point cloud
        #las = laspy.LasData(self.las.header)  # TODO Restore : Legacy
        print('Copying LAS points ...')  # TODO Remove
        #las.points = self.las.points.copy()  # TODO Restore : Legacy
        print('LAS points copied!')  # TODO Remove
        # Replace the old point cloud with the new one
        #self.las, las = las, None  # TODO Restore : Legacy
        print('LAS replaced!')  # TODO Remove
        # Update the new point cloud
        self.las.add_extra_dims(extra_bytes)
        print('Added LAS extra bytes!')  # TODO Remove
        for i in range(nfeats):
            self.las[fnames[i]] = feats[:, i]
        print('Updated features!')  # TODO Remove
        # Return
        return self

    def remove_features(self, fnames):
        """
        Remove features from the point cloud.

        :param fnames: The name of the features to be removed.
        :type fnames: list or tuple (of str)
        :return: The updated point cloud.
        :rtype: :class:`.PointCloud`
        """
        fnames = [  # Filter out names of non-extra dims features
            fname for fname in fnames if fname not in NON_EXTRA_DIMS_FEATURES
        ]
        self.las.remove_extra_dims(fnames)  # Remove the features

    def preserve_mask(self, mask):
        """
        Preserve the points whose index in the mask corresponds to True.
        Otherwise, the point is discarded.

        :param mask: The binary mask to be applied (True means preserve,
            False means discard).
        :return: The point cloud after applying the mask.
        :rtype: :class:`.PointCloud`
        """
        self.proxy_load()
        self.las.points = self.las.points[mask]
        self.las.header.point_records_count = len(self.las.points)
        return self

    def remove_mask(self, mask):
        """
        Discard the points whose index in the mask corresponds to True.
        Otherwise, the point is preserved.

        :param mask: The binary mask to be applied (True means remove, False
            means preserve).
        :return: The point cloud after applying the mask.
        :rtype: :class:`.PointCloud`
        """
        return self.preserve_mask(~np.array(mask))

    def set_classes_vector(self, y):
        r"""
        Set the point-wise classes of the point cloud from the given vector
        of classes.

        :param y: The vector of classes (:math:`\pmb{y}`).
        :type y: :class:`np.ndarray`
        :return: The updated point cloud.
        :rtype: :class:`.PointCloud`
        """
        self.proxy_load()
        self.las.classification = y
        return self

    def clear_data(self, proxy_release=False):
        """
        Remove all the data from the point cloud. Note that this means removing
        the data from memory. In case the data is stored in a proxy file
        it can be restored (from file to memory) when needed. To also remove
        the proxy file, ``proxy_release=True`` must be passed explicitly.
        """
        self.las = None
        if proxy_release:
            self.proxy.release()

    # ---   PROXY METHODS   --- #
    # ------------------------- #
    def proxy_dump(self):
        """
        Dump the point cloud to a binary proxy file if and only if it is
        recommended and not already dumped.

        See :class:`.MemToFileProxy`, :meth:`MemToFileProxy.dump`, and
        :meth:`MemToFileProxy.is_dump_recommended`.
        """
        if not self.proxy.is_dumped() and self.proxy.is_dump_recommended(self):
            start = time.perf_counter()
            self.proxy.dump(self)
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'PointCloud data dumped through proxy in {end-start:.3f} '
                'seconds.'
            )

    def proxy_load(self):
        """
        Load the point cloud from a binary proxy file if necessary. Any
        method from the :class:`.PointCloud` class that needs to operate with
        the point cloud's date should call ``proxy_load`` to be sure that
        the data is available in main memory.

        See :class:`.MemToFileProxy`, :meth:`MemToFileProxy.dump`, and
        :meth:`MemToFileProxy.load`.
        """
        if self.proxy.is_dumped():
            start = time.perf_counter()
            self.proxy.load(self)
            end = time.perf_counter()
            LOGGING.LOGGER.debug(
                f'PointCloud data loaded from proxy in {end-start:.3f} '
                'seconds.'
            )
