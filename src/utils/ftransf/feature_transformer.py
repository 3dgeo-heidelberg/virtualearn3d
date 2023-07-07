# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class FeatureTransformerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to feature transformation components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class FeatureTransformer:
    """
    :author: Alberto M. Esmoris Pena

    Class for feature transformation operations.

    :ivar fnames: The names of the features to be transformed (by default).
    :vartype fnames: list or tuple
    :ivar report_path: The path to write the report file reporting the behavior
        of the transformer.
    :vartype report_path: str
    :ivar selected_features: Either boolean mask or list of indices
        corresponding to the selected features (columns of the feature matrix).
    :vartype selected_features: list
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate an FeatureTransformer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a FeatureTransformer.
        """
        # Initialize
        kwargs = {
            'fnames': spec.get('fnames', None),
            'report_path': spec.get('report_path', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a FeatureTransformer.

        :param kwargs: The attributes for the FeatureTransformer.
        """
        # Fundamental initialization of any feature transformer
        self.fnames = kwargs.get('fnames', None)
        self.report_path = kwargs.get('report_path', None)
        self.selected_features = None

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    @abstractmethod
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental transformation logic defining the feature transformer.

        :param F: The input matrix of features to be imputed.
        :type F: :class:`np.ndarray`
        :param y: The vector of point-wise classes.
        :type y: :class:`np.ndarray`
        :param fnames: The list of features to be transformed. If None, it will
            be taken from the internal fnames of the feature transformer. If
            those are None too, then an exception will raise.
        :type fnames: list or tuple
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: str
        :return: The transformed matrix of features.
        :rtype: :class:`np.ndarray`
        """
        pass

    def transform_pcloud(self, pcloud, fnames=None, out_prefix=None):
        """
        Apply the transform method to a point cloud.

        See :meth:`feature_transformer.FeatureTransformer.transform`

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param fnames: The list of features to be transformed. If None, it will
            be taken from the internal fnames of the feature transformer. If
            those are None too, then an exception will raise.
        :type fnames: list or tuple
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: str
        :return: A new point cloud that is the transformed version of the
            input point cloud.
        :rtype: :class:`PointCloud`
        """
        # Check feature names
        if fnames is None:
            if self.fnames is None:
                raise FeatureTransformerException(
                    'The features of a point cloud cannot be transformed if '
                    'they are not specified.'
                )
            else:
                fnames = self.fnames
        # Transform features
        F = self.transform(
            pcloud.get_features_matrix(fnames),
            y=pcloud.get_classes_vector(),
            fnames=fnames,
            out_prefix=out_prefix
        )
        fnames = np.array(fnames)[self.selected_features].tolist()
        # Return new point cloud
        return PointCloudFactoryFacade.make_from_arrays(
            pcloud.get_coordinates_matrix(),
            F,
            y=pcloud.get_classes_vector(),
            header=pcloud.las.header,
            fnames=fnames
        )

    def report(self, report, out_prefix=None):
        """
        Handle the way a report is reported. First, it will be reported using
        the logging system. Then, it will be written to a file if the
        transformer has a not None report_path.

        :param report: The report to be reported.
        :param out_prefix: The output prefix in case the output path must be
            expanded.
        :return: Nothing.
        """
        LOGGING.LOGGER.info(report)
        if self.report_path is not None:
            report.to_file(self.report_path, out_prefix=out_prefix)
