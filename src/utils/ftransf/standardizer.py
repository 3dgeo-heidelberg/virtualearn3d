# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.report.standardization_report import StandardizationReport
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from sklearn.preprocessing import StandardScaler
import time


# ---   CLASS   --- #
# ----------------- #
class Standardizer(FeatureTransformer):
    r"""
    :author: Alberto M. Esmoris Pena

    Class for transforming features by subtracting the mean and dividing by
    the standard deviation.

    Let :math:`z` be a standardized version of the feature :math:`x` with mean
    :math:`\mu` and standard deviation :math:`\sigma`. Then, :math:`z` can be
    computed as:

    .. math::
        z = \dfrac{x-\mu}{\sigma}

    :ivar center: Flag to control whether to center (i.e., subtract the mean).
    :vartype center: bool
    :ivar scale: Flag to control whether to scale (i.e., divide by the standard
        deviation).
    :vartype scale: bool
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a Standardizer.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a Standardizer.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of Standardizer
        kwargs['center'] = spec.get('center', None)
        kwargs['scale'] = spec.get('scale', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a Standardizer.

        :param kwargs: The attributes for the Standardizer.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.center = kwargs.get('center', True)
        self.scale = kwargs.get('scale', True)
        self.stder = None  # By default, no standardizer model has been fit

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the Standardizer.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Transform
        plot_and_report = False
        start = time.perf_counter()
        if self.stder is None:
            self.stder = StandardScaler(
                with_mean=self.center,
                with_std=self.scale
            ).fit(F)
            plot_and_report = True  # Plot and report only when fit
        F = self.stder.transform(F)
        end = time.perf_counter()
        if plot_and_report:
            # Report feature-wise mean and stdev
            self.report(
                StandardizationReport(
                    self.get_names_of_transformed_features(fnames=self.fnames),
                    self.stder.scale_,
                    self.stder.mean_
                ),
                out_prefix=out_prefix
            )
        # Log transformation
        LOGGING.LOGGER.info(
            f'Standardizer transformed {F.shape[0]} points with {F.shape[1]} '
            'features each to have zero mean and unit variance in '
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
