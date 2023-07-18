# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer, \
    FeatureTransformerException
from src.utils.ftransf.kbest_selector import KBestSelector
from src.utils.dict_utils import DictUtils
from src.report.best_score_selection_report import BestScoreSelectionReport
import src.main.main_logger as LOGGING
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, f_regression
import time


# ---   CLASS   --- #
# ----------------- #
class PercentileSelector(FeatureTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming features by preserving only a given percentile of
    the highest scores for a particular task.

    :ivar percentile: The percentage of features that must be selected given
        as an integer in [0, 100].
    :vartype percentile: int
    :ivar scoref: The score function f(F, y) to evaluate the features F to
        predict the values of y.
    :vartype scoref: callable
    :ivar score_name: The name of the score used for the evaluations.
    :vartype score_name: path
    :ivar sp: The internal percentile selection model.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a PercentileSelector.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a PercentileSelector.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of KBestSelector
        kwargs['percentile'] = spec.get('percentile')
        # Extract score function from type
        valid_score = KBestSelector.handle_score_from_type(spec, kwargs)
        if not valid_score:
            raise ValueError(
                'The PercentileSelector specification contains an invalid '
                f'type: "{type}"'
            )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a PercentileSelector.

        :param kwargs: The attributes for the PercentileSelector.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.percentile = kwargs.get('percentile', None)
        if self.percentile is None:
            raise FeatureTransformerException(
                'PercentileSelector requires a percentile is specified. '
                'None was given.'
            )
        self.scoref = kwargs.get('scoref', None)
        if self.scoref is None:
            raise FeatureTransformerException(
                'PercentileSelector requires a valid score function. None was '
                'given.'
            )
        self.score_name = kwargs.get('score_name', 'SCORE')
        self.sp = None

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the percentile
        selector.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Check selection is possible
        if y is None:
            raise FeatureTransformerException(
                'PercentileSelector cannot trasnform features without the '
                'corresponding classes.'
            )
        # Transform
        old_num_features = F.shape[1]
        plot_and_report = False
        start = time.perf_counter()
        if self.sp is None:
            self.sp = SelectPercentile(
                score_func=self.scoref,
                percentile=self.percentile
            ).fit(F, y)
            plot_and_report = True  # Plot and reporty only when fit
        F = self.sp.transform(F)
        end = time.perf_counter()
        new_num_features = F.shape[1]
        # Register selected features
        self.selected_features = self.sp.get_support()
        if plot_and_report:
            # Report scores
            self.report(
                BestScoreSelectionReport(
                    self.fnames if fnames is None else fnames,
                    self.sp.scores_,
                    self.score_name,
                    pvalues=self.sp.pvalues_,
                    selected_features=self.selected_features
                ),
                out_prefix=out_prefix
            )
        # Log transformation
        LOGGING.LOGGER.info(
            'PercentileSelector transformed {n1} features into {n2} features '
            'in {texec:.3f} seconds'.format(
                n1=old_num_features,
                n2=new_num_features,
                texec=end-start
            )
        )
        # Return
        return F
