# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.report.minmax_normalization_report import MinmaxNormalizationReport
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class MinmaxNormalizer(FeatureTransformer):
    r"""
    :author: Alberto M. Esmoris Pena

    Class for transforming features by subtracting the min and dividing by the
    range, i.e., the difference between max and min. Min-max normalized
    features will be in :math:`[0, 1]`.

    Let :math:`x'` be a min-max normalized version of the feature
    :math:`x \in X` where X is the set representing the feature's value for
    many points. Thus, the min-max normalized feature can be
    computed as:

    .. math::
        x' = \dfrac{x - \min X}{\max X - \min X}

    :ivar target_range: The (a, b) interval such that features will be
        normalized to be inside (a, b). By default, it is (0, 1).
    :vartype target_range: :class:`np.ndarray`
    :ivar clip: Whether to clip potential values from held-out data to respect
        the normalization interval (True) or not (False).
    :vartype clip: bool
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a MinmaxNormalizer.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a MinmaxNormalizer.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of MinmaxNormalizer
        kwargs['target_range'] = spec.get('target_range', None)
        kwargs['clip'] = spec.get('clip', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a MinmaxNormalizer.

        :param kwargs: The attributes for the MinmaxNormalizer.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.target_range = kwargs.get('target_range', np.array([0, 1]))
        self.clip = kwargs.get('clip', True)
        self.minmaxer = None  # By default, no normalizer model has been fit

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the MinmaxNormalizer.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Report feature names
        LOGGING.LOGGER.debug(
            f'MinmaxNormalizer considers the following {len(self.fnames)} '
            f'features:\n{self.fnames}'
        )
        # Transform
        plot_and_report = False
        start = time.perf_counter()
        if self.minmaxer is None:
            self.minmaxer = MinMaxScaler(
                feature_range=tuple(self.target_range),
                clip=self.clip
            ).fit(F)
            plot_and_report = True  # Plot and report only when fit
        F = self.minmaxer.transform(F)
        end = time.perf_counter()
        if plot_and_report:
            # Report feature-wise min and max
            self.report(
                MinmaxNormalizationReport(
                    self.get_names_of_transformed_features(fnames=self.fnames),
                    self.minmaxer.data_min_,
                    self.minmaxer.data_max_,
                    range=self.minmaxer.data_range_
                ),
                out_prefix=out_prefix
            )
        # Log transformation
        LOGGING.LOGGER.info(
            f'MinmaxNormalizer transformed {F.shape[0]} points with '
            f'{F.shape[1]} features each to be inside the interval [ '
            f'{self.target_range[0]:.3f}, {self.target_range[1]:.3f}] in '
            f'{end-start:.3f} seconds.'
        )
        # Return
        return F

    def get_names_of_transformed_features(self, **kwargs):
        """
        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.get_names_of_transformed_features`
        """
        return self.fnames
