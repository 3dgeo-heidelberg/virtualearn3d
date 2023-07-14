# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from src.report.variance_selection_report import VarianceSelectionReport
from sklearn.feature_selection import VarianceThreshold
import time


# ---   CLASS   --- #
# ----------------- #
class VarianceSelector(FeatureTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming features by discarding those which variance lies
    below a given threshold.

    :ivar var_th: The specified variance threshold.
    :vartype var_th: float
    :ivar vt: The internal variance threshold model.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a VarianceSelector
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a VarianceSelector.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of VarianceSelector
        kwargs['variance_threshold'] = spec.get('variance_threshold', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a VarianceSelector.

        :param kwargs: The attributes for the VarianceSelector.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.var_th = kwargs.get('variance_threshold', 0.0)
        self.vt = None  # By default, no variance threshold model has been fit

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the variance
        selector.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Transform
        old_num_features = F.shape[1]
        start = time.perf_counter()
        if self.vt is None:
            self.vt = VarianceThreshold(threshold=self.var_th).fit(F)
        F = self.vt.transform(F)
        end = time.perf_counter()
        new_num_features = F.shape[1]
        # Register selected features (either as mask or indices)
        self.selected_features = self.vt.get_support()
        # Report variances
        self.report(
            VarianceSelectionReport(
                self.fnames if fnames is None else fnames,
                self.vt.variances_,
                selected_features=self.selected_features
            ),
            out_prefix=out_prefix
        )
        # Log transformation
        LOGGING.LOGGER.info(
            'VarianceSelector transformed {n1} features into {n2} features in '
            '{texec:.3f} seconds.'.format(
                n1=old_num_features,
                n2=new_num_features,
                texec=end-start
            )
        )
        # Return
        return F

