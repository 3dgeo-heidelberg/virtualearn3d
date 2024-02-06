# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ImputerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to imputation components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Imputer:
    """
    :author: Alberto M. Esmoris Pena

    Class for imputation operations.

    :ivar target_val: The target value, i.e., features that match this value
        will be imputed.
    :vartype target_val: str or int or float
    :ivar fnames: The names of the features to be imputed (by default).
    :vartype fnames: list or tuple
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_imputer_args(spec):
        """
        Extract the arguments to initialize/instantiate an Imputer from a
        key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an Imputer.
        """
        # Initialize
        kwargs = {
            'target_val': spec.get('target_val', None),
            'fnames': spec.get('fnames', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an Imputer.

        :param kwargs: The attributes for the Imputer.
        """
        # Fundamental initialization of any imputer
        self.target_val = kwargs.get("target_val", np.NaN)
        self.fnames = kwargs.get('fnames', None)

    # ---   IMPUTER METHODS   --- #
    # --------------------------- #
    @abstractmethod
    def impute(self, F, y=None):
        """
        The fundamental imputation logic defining the imputer.

        :param F: The input matrix of features to be imputed.
        :param y: The input vector of classes as an optional argument. When
            given, the imputation in F will be propagated to y if necessary.
            This imputation is not often needed, but some strategies demand it
            to keep the consistency between features and expected classes.
            For example, when the imputation strategy consists of removing
            points with NaN values, the corresponding component from the
            vector of classes must also be removed.
        :return: The matrix of features and the vector of classes after
            imputation. If the vector of classes is not given (i.e., is None),
            then only imputed F will be returned.
        :rtype: :class:`np.ndarray` or tuple
        """
        pass

    def impute_pcloud(self, pcloud, fnames=None):
        """
        Apply the impute method to a point cloud.

        See :meth:`imputer.Imputer.impute`.

        :param pcloud: The point cloud to be imputed.
        :type pcloud: :class:`.PointCloud`
        :param fnames: The list of features to be imputed. If None, it will be
            taken from the internal fnames of the imputer. If those are None
            too, then an exception will raise.
        :type fnames: list or tuple
        :return: A new point cloud that is the imputed version of the input
            point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Check feature names
        if fnames is None:
            if self.fnames is None:
                raise ImputerException(
                    'The features of a point cloud cannot be imputed if '
                    'they are not specified.'
                )
            else:
                fnames = self.fnames
        # Obtain points and classes
        P = np.hstack([
            pcloud.get_coordinates_matrix(),
            pcloud.get_features_matrix(fnames)
        ])
        y = pcloud.get_classes_vector()
        pcloud_header = pcloud.las.header
        pcloud.proxy_dump()  # Save memory from point cloud data if necessary
        # Impute
        if y is None:
            P = self.impute(P)
        else:
            P, y = self.impute(P, y=y)
        # Return new point cloud
        return PointCloudFactoryFacade.make_from_arrays(
            P[:, :P.shape[1]-len(fnames)],
            P[:, P.shape[1]-len(fnames):],
            y=y,
            header=pcloud_header,
            fnames=fnames
        )
